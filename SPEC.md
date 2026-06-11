# Inference Server Spec

Continuous batching inference server for Llama 3.1 on Apple Silicon (macOS). Targets high throughput multi-request serving using iteration-level scheduling, a paged KV cache, and MLX's built-in flash attention.

---

## Goals

- Serve Llama 3.1 (8B target, 70B stretch) with competitive tokens/sec on M-series hardware
- Support concurrent requests without head-of-line blocking
- Keep KV cache memory fragmentation low via paged allocation
- Expose an OpenAI-compatible HTTP API with streaming

## Non-Goals

- Custom Metal attention kernels (use `mlx.core.fast.scaled_dot_product_attention`)
- Prefix caching / KV block sharing across requests
- Speculative decoding
- Multi-node or multi-chip serving
- Training or fine-tuning

---

## Architecture Overview

```
HTTP Client
     │
     ▼
┌─────────────┐
│  API Server │  (FastAPI, OpenAI-compatible, SSE streaming)
└──────┬──────┘
       │ Request queue
       ▼
┌─────────────────┐
│    Scheduler    │  Iteration-level: selects sequences for each forward pass
└──────┬──────────┘
       │ Batch
       ▼
┌─────────────────┐      ┌──────────────────────┐
│   Model Runner  │ ◄──► │  KV Cache Manager    │
│  (MLX, Llama)   │      │  (Block Allocator +  │
└──────┬──────────┘      │   Page Tables)       │
       │                 └──────────────────────┘
       ▼
┌─────────────────┐
│    Sampler      │  (greedy / top-p / top-k / temperature)
└─────────────────┘
```

---

## Component Specifications

### 1. KV Cache Manager

**Responsibility**: Manage a pool of fixed-size memory blocks that hold KV tensors. Assign and release blocks to sequences. Provide each sequence's page table to the model runner.

#### Block

The atomic unit of KV storage.

```
shape: [2, num_kv_heads, block_size, head_dim]
  - dim 0: key vs value
  - num_kv_heads: 8 (Llama 3.1 8B/70B)
  - block_size: 16 tokens (configurable, power of 2)
  - head_dim: 128

dtype: float16
```

For the 8B model at block_size=16:
`2 × 8 × 16 × 128 × 2 bytes = 65,536 bytes (~64KB) per block`

The full block pool is pre-allocated as a single MLX array:
`[num_blocks, 2, num_kv_heads, block_size, head_dim]`

#### BlockAllocator

- Maintains a free list of block indices
- `allocate(n) → list[int]`: returns n free block indices, raises `OOM` if unavailable
- `free(block_ids: list[int])`: returns blocks to the free list
- `num_free_blocks → int`: available capacity

#### Sequence KV State

Each sequence holds:
- `block_table: list[int]` — ordered list of physical block indices (logical block 0 → block_table[0], etc.)
- `num_kv_tokens: int` — how many token positions have been written so far

The current write position within the last block is `num_kv_tokens % block_size`.
A new block is needed when `num_kv_tokens % block_size == 0`.

---

### 2. Scheduler

**Responsibility**: At each iteration, decide which sequences run and in what phase (prefill or decode). Allocate KV blocks for new token positions. Preempt sequences when memory is exhausted.

#### Sequence State Machine

```
WAITING ──► PREFILL ──► DECODE ──► FINISHED
                │                     ▲
                └──► PREEMPTED ───────┘
                         │
                         ▼
                      WAITING  (re-queued, KV blocks freed)
```

- **WAITING**: in queue, no KV blocks allocated
- **PREFILL**: processing prompt tokens; compute-bound
- **DECODE**: autoregressive generation; memory-bandwidth-bound
- **PREEMPTED**: evicted from active set due to KV cache pressure; blocks freed, sequence re-queued
- **FINISHED**: stop token emitted or max_tokens reached

#### Scheduling Policy

Each iteration the scheduler produces a `SchedulerOutput`:

```python
@dataclass
class SchedulerOutput:
    prefill_sequences: list[Sequence]   # sequences in prompt phase this step
    decode_sequences: list[Sequence]    # sequences in generation phase this step
    blocks_to_free: list[int]           # blocks released this step
```

Rules (applied in order each iteration):
1. Promote all WAITING sequences that fit in available KV blocks into PREFILL
2. Advance all PREFILL sequences that completed their prompt into DECODE
3. Run all DECODE sequences (one token each)
4. If KV cache is full and new sequences are waiting, preempt the lowest-priority DECODE sequence (default: FCFS → preempt most recently started)
5. Allocate exactly the blocks needed before returning the output

#### Chunked Prefill (v1 simplification)

In v1, process the entire prompt in one step (no chunking). If a prompt is too long to fit available KV blocks, the sequence stays WAITING.

---

### 3. Model Runner

**Responsibility**: Execute one forward pass given a batch from the scheduler. Gather KV blocks from the page tables, run the model, write new KV entries back.

#### Batch Representation

```python
@dataclass
class Batch:
    # Prefill sequences: process full prompt in parallel
    prefill_token_ids: list[list[int]]     # [num_prefill_seqs, prompt_len]
    prefill_positions: list[list[int]]     # absolute token positions for RoPE
    prefill_block_tables: list[list[int]]  # block_table per prefill seq

    # Decode sequences: one new token each
    decode_token_ids: list[int]            # [num_decode_seqs], the last generated token
    decode_positions: list[int]            # current position in each decode seq
    decode_block_tables: list[list[int]]   # block_table per decode seq
```

Prefill and decode sequences can be batched together in a single forward pass (they differ only in how Q is shaped and how attention masking works).

#### Paged Attention

The model uses `mlx.core.fast.scaled_dot_product_attention` as the attention primitive. Before calling it, the runner gathers K and V tensors from the block pool using each sequence's page table:

```
For each sequence:
  gathered_K = block_pool[block_table, 0, :, :num_valid_tokens, :]  # gather + reshape
  gathered_V = block_pool[block_table, 1, :, :num_valid_tokens, :]
```

After the attention projection, new K and V slices are written back into the current block at the correct offset.

The gather/scatter ops are the primary cost of paging vs. contiguous KV — this is acceptable because the memory flexibility it provides is worth it.

#### RoPE

Applied to Q and K after projection, before attention. Uses absolute positions from `prefill_positions` / `decode_positions`. Pre-compute the cos/sin tables up to max_seq_len at model load time.

#### Model Architecture (Llama 3.1 8B)

| Param | Value |
|---|---|
| Layers | 32 |
| d_model | 4096 |
| Q heads | 32 |
| KV heads | 8 |
| Head dim | 128 |
| FFN hidden | 14336 |
| Vocab | 128256 |
| Max seq len | 131072 |
| Norm | RMSNorm |
| Activation | SwiGLU |

Weights loaded from HuggingFace safetensors format, converted to MLX float16.

---

### 4. Sampler

**Responsibility**: Convert logits to next tokens.

Supported strategies (selected per-request):
- **Greedy**: `argmax(logits)`
- **Temperature + top-p**: scale logits by `1/T`, softmax, sample from nucleus
- **Temperature + top-k**: scale logits by `1/T`, restrict to top-k, sample

Output: `list[int]` of sampled token ids, one per sequence in the decode batch.

---

### 5. API Server

**Responsibility**: Accept HTTP requests, manage the request lifecycle, stream tokens back to clients.

#### Endpoints

`POST /v1/chat/completions` — OpenAI chat completions format
`POST /v1/completions` — OpenAI legacy completions format
`GET /v1/models` — list available models
`GET /health` — liveness check

#### Request Lifecycle

1. Request arrives → tokenize prompt → create `Sequence` object → push to scheduler WAITING queue
2. Background engine loop runs continuously: `scheduler.step()` → `model_runner.step()` → `sampler.step()`
3. Each generated token is pushed to the sequence's output queue
4. Streaming response reads from that queue and yields SSE events
5. When sequence hits FINISHED, response stream closes

#### Streaming

Server-Sent Events (SSE) for streaming. Non-streaming requests buffer all tokens then return.

---

## Data Flow: Single Iteration

```
1. scheduler.step()
   ├── inspect WAITING queue
   ├── allocate KV blocks for newly admitted sequences
   ├── build SchedulerOutput (prefill_seqs, decode_seqs)
   └── free blocks for finished/preempted sequences

2. model_runner.build_batch(scheduler_output) → Batch

3. model_runner.forward(batch)
   ├── embed tokens
   ├── for each layer:
   │   ├── RMSNorm
   │   ├── QKV projection
   │   ├── apply RoPE to Q, K
   │   ├── gather K, V from block pool via page tables
   │   ├── write new K, V into current block slot
   │   ├── scaled_dot_product_attention(Q, gathered_K, gathered_V)
   │   ├── output projection
   │   ├── RMSNorm
   │   └── SwiGLU FFN
   └── LM head → logits [num_decode_seqs, vocab_size]

4. sampler.sample(logits, sampling_params) → next_token_ids

5. scheduler.update(next_token_ids)
   ├── append tokens to sequence outputs
   ├── check stop conditions
   └── transition PREFILL → DECODE, DECODE → FINISHED as appropriate
```

---

## Configuration

```python
@dataclass
class ServerConfig:
    model_path: str                  # path to HF weights dir
    block_size: int = 16             # tokens per KV block
    max_num_blocks: int = 2048       # total KV cache blocks (~128MB for 8B at block_size=16)
    max_num_seqs: int = 64           # max concurrent sequences
    max_seq_len: int = 8192          # max tokens per sequence (prompt + generation)
    dtype: str = "float16"
    host: str = "0.0.0.0"
    port: int = 8000
```

`max_num_blocks` should be sized to fill available unified memory after model weights are loaded. For the 8B model at float16: weights ≈ 16GB, leaving ~8GB on a 24GB M-series chip for KV cache (~131K blocks, ~8M tokens of KV capacity).

---

## Performance Targets (8B model, M4 MacBook Pro)

| Metric | Target |
|---|---|
| Throughput (batch=8) | ≥ 400 tokens/sec |
| Time to first token | ≤ 200ms for prompts ≤ 512 tokens |
| KV cache utilization | ≥ 85% under steady load |
| Memory overhead (scheduler + allocator) | < 50MB |

---

## File Structure

```
inference_server/
├── model/
│   ├── llama.py          # Llama 3.1 forward pass (MLX)
│   ├── weights.py        # HF safetensors → MLX weight loading
│   └── rope.py           # RoPE cos/sin table + application
├── engine/
│   ├── block_allocator.py  # BlockAllocator, block pool
│   ├── sequence.py         # Sequence dataclass + state machine
│   ├── scheduler.py        # Scheduler, SchedulerOutput
│   └── runner.py           # ModelRunner, Batch construction
├── sampling/
│   └── sampler.py          # greedy, top-p, top-k
├── server/
│   ├── app.py              # FastAPI app, endpoints
│   └── protocol.py         # OpenAI-compatible request/response types
├── config.py               # ServerConfig
└── main.py                 # entrypoint
```

---

## Dependencies

```
mlx               # model compute + flash attention primitive
mlx-lm            # reference for weight loading patterns
fastapi           # HTTP server
uvicorn           # ASGI runner
transformers      # tokenizer only (HF tokenizer for Llama 3.1)
safetensors       # weight loading
```

---

## Decisions

1. **Mixed prefill+decode batching**: Prefill and decode sequences are batched together in a single forward pass per iteration. Prefill sequences attend only to their own prompt tokens (causal mask); decode sequences attend to their full KV history via the page table. Q is shaped differently for each but the layer logic is shared.

2. **Preemption strategy**: v1 uses recompute — on preemption, KV blocks are freed and the sequence is re-queued; it re-runs prefill when rescheduled. Future work: swap-to-CPU, saving KV blocks to system RAM and restoring them on reschedule. This avoids redundant prefill compute for long sequences and is the preferred long-term approach given Apple Silicon's unified memory (CPU↔GPU copies are cheap).

3. **Tokenizer process**: Tokenization runs in the request handler (FastAPI), not in the engine loop, so it does not block generation steps.