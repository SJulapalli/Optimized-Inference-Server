"""
Continuous batching v2 tests — chunked prefill and scheduling.

Validates:
  1.  num_computed_tokens advances one chunk at a time, not all at once
  2.  Admission allocates only the first chunk's blocks (not the full prompt)
  3.  Blocks are allocated incrementally per chunk
  4.  Decode sequences run alongside every prefill chunk (no head-of-line blocking)
  5.  Intermediate chunks produce no output tokens
  6.  Preemption during prefill resets num_computed_tokens to 0
  7.  build_batch produces correct chunk slices, positions, and kv offsets
  8.  Full end-to-end loop with chunked prefill completes correctly

Config:
  block_size = 4   — one block holds 4 token positions
  chunk_size = 4   — one chunk processes 4 tokens (aligned with block_size for clean arithmetic)
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import mlx.core as mx

from config import ModelConfig, ServerConfig
from engine.block_allocator import BlockAllocator
from engine.sequence import Sequence, SequenceStatus, SamplingParams
from engine.runner import BatchPagedKVCache, build_batch
from scheduler import Scheduler, SchedulerOutput
from sampling import sampler

# ── Config ────────────────────────────────────────────────────────────────────

BLOCK_SIZE = 4
CHUNK_SIZE = 4
EOS        = 999
NORMAL     = 1

MODEL_CFG = ModelConfig(
    num_layers=2, num_kv_heads=2, head_dim=4, eos_token_id=EOS,
)

def make_cfg(max_num_blocks: int = 20, max_num_seqs: int = 4, max_seq_len: int = 200):
    return ServerConfig(
        model_path="",
        block_size=BLOCK_SIZE,
        max_num_blocks=max_num_blocks,
        max_num_seqs=max_num_seqs,
        max_seq_len=max_seq_len,
        prefill_chunk_size=CHUNK_SIZE,
    )

_id = 0
def make_seq(prompt_len: int, max_tokens: int = 20) -> Sequence:
    global _id
    _id += 1
    return Sequence(
        seq_id=_id,
        prompt_token_ids=list(range(prompt_len)),
        sampling_params=SamplingParams(max_tokens=max_tokens),
    )

def run(name: str, fn):
    try:
        fn()
        print(f"  PASS  {name}")
    except AssertionError as e:
        print(f"  FAIL  {name}: {e}")
    except Exception as e:
        import traceback
        print(f"  ERROR {name}: {type(e).__name__}: {e}")
        traceback.print_exc()


# ── Scheduler tests ───────────────────────────────────────────────────────────

def test_chunked_prefill_advances_per_step():
    """
    A 12-token prompt (3 chunks of 4) stays in PREFILL for 3 update() calls.
    num_computed_tokens grows by chunk_size each update. No output tokens
    until the final chunk, which transitions the sequence to DECODE.
    """
    s = Scheduler(MODEL_CFG, make_cfg())
    seq = make_seq(prompt_len=12)
    s.add_sequence(seq)

    # Chunk 1
    out = s.step()
    assert seq in out.prefill_sequences, "seq should be in prefill on chunk 1"
    assert seq.num_computed_tokens == 0, "step() must not advance num_computed_tokens"
    s.update({seq.seq_id: NORMAL})
    assert seq.num_computed_tokens == 4
    assert seq.status == SequenceStatus.PREFILL
    assert seq.output_token_ids == [], "no output tokens on intermediate chunk"

    # Chunk 2
    out = s.step()
    assert seq in out.prefill_sequences
    s.update({seq.seq_id: NORMAL})
    assert seq.num_computed_tokens == 8
    assert seq.status == SequenceStatus.PREFILL
    assert seq.output_token_ids == []

    # Chunk 3 — final
    out = s.step()
    assert seq in out.prefill_sequences
    s.update({seq.seq_id: NORMAL})
    assert seq.num_computed_tokens == 12
    assert seq.status == SequenceStatus.DECODE
    assert seq.output_token_ids == [NORMAL], "first generated token expected after last chunk"


def test_admission_allocates_first_chunk_only():
    """
    A 12-token prompt needs 3 blocks total, but at admission only 1 block
    is allocated — enough for the first 4-token chunk. The remaining blocks
    are added incrementally as prefill progresses.
    """
    s = Scheduler(MODEL_CFG, make_cfg())
    seq = make_seq(prompt_len=12)
    s.add_sequence(seq)
    s.step()

    assert len(seq.block_table) == 1, \
        f"expected 1 block at admission (first chunk only), got {len(seq.block_table)}"
    assert s.allocator.num_free_blocks == make_cfg().max_num_blocks - 1


def test_blocks_allocated_incrementally():
    """
    block_table grows by exactly one block per chunk for a 12-token prompt
    with block_size=chunk_size=4. Allocation is lazy, not upfront.

      After chunk 1 step(): 1 block
      After chunk 2 step(): 2 blocks
      After chunk 3 step(): 3 blocks
    """
    s = Scheduler(MODEL_CFG, make_cfg())
    seq = make_seq(prompt_len=12)
    s.add_sequence(seq)

    s.step()
    assert len(seq.block_table) == 1, f"chunk 1: expected 1 block, got {len(seq.block_table)}"

    s.update({seq.seq_id: NORMAL})
    s.step()
    assert len(seq.block_table) == 2, f"chunk 2: expected 2 blocks, got {len(seq.block_table)}"

    s.update({seq.seq_id: NORMAL})
    s.step()
    assert len(seq.block_table) == 3, f"chunk 3: expected 3 blocks, got {len(seq.block_table)}"


def test_decode_runs_alongside_every_prefill_chunk():
    """
    Maximum throughput guarantee: once a sequence reaches DECODE, it must
    appear in decode_sequences for every subsequent step — even while another
    sequence is still processing prefill chunks.

    seq_a: 20-token prompt → 5 chunks (chunk_size=4)
    seq_b: 2-token prompt  → 1 chunk, then immediately enters DECODE

    After seq_b transitions to DECODE at iteration 1, it must appear in
    decode_sequences for all 4 remaining prefill chunks of seq_a. Any step
    where seq_b is absent from decode_sequences is a head-of-line blocking bug.
    """
    s = Scheduler(MODEL_CFG, make_cfg())
    seq_a = make_seq(prompt_len=20)
    seq_b = make_seq(prompt_len=2)
    s.add_sequence(seq_a)
    s.add_sequence(seq_b)

    # Iteration 1: both admitted, both in prefill
    out = s.step()
    assert seq_a in out.prefill_sequences, "seq_a not admitted"
    assert seq_b in out.prefill_sequences, "seq_b not admitted"
    assert out.decode_sequences == [], "no decode sequences yet"

    s.update({seq_a.seq_id: NORMAL, seq_b.seq_id: NORMAL})
    assert seq_a.status == SequenceStatus.PREFILL, "seq_a has 4 chunks remaining"
    assert seq_b.status == SequenceStatus.DECODE,  "seq_b done with prefill after 1 chunk"

    # Iterations 2–5: seq_a still prefilling; seq_b must decode every single step
    for chunk in range(2, 6):
        out = s.step()
        assert seq_a in out.prefill_sequences, \
            f"seq_a missing from prefill at chunk {chunk}"
        assert seq_b in out.decode_sequences, \
            f"seq_b missing from decode_sequences at chunk {chunk} — head-of-line blocking!"
        s.update({seq_a.seq_id: NORMAL, seq_b.seq_id: NORMAL})


def test_no_output_tokens_on_intermediate_chunks():
    """
    output_token_ids must remain empty throughout all intermediate prefill
    chunks. The first generated token only appears after the last chunk.
    An intermediate chunk that appends a token is a control-flow bug.

    8-token prompt → chunk 1 is intermediate, chunk 2 is final.
    """
    s = Scheduler(MODEL_CFG, make_cfg())
    seq = make_seq(prompt_len=8)
    s.add_sequence(seq)

    s.step()
    s.update({seq.seq_id: NORMAL})
    assert seq.output_token_ids == [], \
        f"expected no output after chunk 1, got {seq.output_token_ids}"
    assert seq.status == SequenceStatus.PREFILL

    s.step()
    s.update({seq.seq_id: NORMAL})
    assert seq.output_token_ids == [NORMAL], \
        f"expected exactly one token after final chunk, got {seq.output_token_ids}"
    assert seq.status == SequenceStatus.DECODE


def test_preemption_resets_num_computed_tokens():
    """
    When a DECODE sequence is preempted, num_computed_tokens resets to 0.
    A third sequence (seq_c) consumes the freed block within the active loop,
    preventing seq_b from being immediately re-admitted in the same step.

    4 blocks, 3 sequences:
      seq_a: prompt=8, 2 chunks
      seq_b: prompt=4, 1 chunk → enters DECODE → preempted
      seq_c: prompt=8, 2 chunks (consumes seq_b's freed block)
    """
    s = Scheduler(MODEL_CFG, make_cfg(max_num_blocks=4))
    seq_a = make_seq(prompt_len=8)
    seq_b = make_seq(prompt_len=4)
    seq_c = make_seq(prompt_len=8)
    s.add_sequence(seq_a)
    s.add_sequence(seq_b)
    s.add_sequence(seq_c)

    # Step 1: each admitted with 1 block, 1 free
    s.step()
    assert s.allocator.num_free_blocks == 1

    s.update({seq_a.seq_id: NORMAL, seq_b.seq_id: NORMAL, seq_c.seq_id: NORMAL})
    assert seq_a.status == SequenceStatus.PREFILL
    assert seq_b.status == SequenceStatus.DECODE
    assert seq_c.status == SequenceStatus.PREFILL

    # Step 2:
    #   seq_a (PREFILL chunk 2): takes the 1 free block → pool empty
    #   seq_b (DECODE, num_kv_tokens=5 > capacity=4): OOM → preempted, frees 1 block
    #   seq_c (PREFILL chunk 2): takes seq_b's freed block → pool empty again
    #   Admission loop: 0 free blocks → seq_b cannot be re-admitted
    s.step()

    assert seq_b.status == SequenceStatus.WAITING, \
        "preempted seq_b should be WAITING (pool exhausted by seq_c)"
    assert seq_b.num_computed_tokens == 0, \
        "num_computed_tokens must reset — KV state is gone, must re-run full prefill"
    assert seq_b.block_table == []
    assert seq_b in s.waiting_sequences



# ── Runner / build_batch tests ────────────────────────────────────────────────

def test_build_batch_mid_chunk_slice():
    """
    A PREFILL sequence on its second chunk (num_computed_tokens=4, prompt_len=12):
      token_ids  = prompt[4:8]   → [4, 5, 6, 7]
      positions  = [4, 5, 6, 7]  (absolute positions in the sequence)
      kv_offset  = 4             (bytes already in cache from chunk 1)
      seq_lens   = [4]
    """
    seq = Sequence(
        seq_id=1,
        prompt_token_ids=list(range(12)),
        sampling_params=SamplingParams(),
        status=SequenceStatus.PREFILL,
        num_computed_tokens=4,
        block_table=[0, 1],
    )
    schedule = SchedulerOutput(prefill_sequences=[seq], decode_sequences=[])
    batch = build_batch(schedule, prefill_chunk_size=4)

    assert batch.token_ids          == [4, 5, 6, 7], f"wrong tokens: {batch.token_ids}"
    assert batch.positions          == [4, 5, 6, 7], f"wrong positions: {batch.positions}"
    assert batch.kv_cache_offsets[0] == 4,           f"wrong kv offset: {batch.kv_cache_offsets[0]}"
    assert batch.seq_lens           == [4],           f"wrong seq_lens: {batch.seq_lens}"
    assert batch.num_prefill_seqs   == 1


def test_build_batch_first_chunk():
    """
    First chunk (num_computed_tokens=0): starts at token 0 with positions [0..3].
    kv_offset must be 0 — nothing has been cached yet.
    """
    seq = Sequence(
        seq_id=2,
        prompt_token_ids=list(range(12)),
        sampling_params=SamplingParams(),
        status=SequenceStatus.PREFILL,
        num_computed_tokens=0,
        block_table=[0],
    )
    schedule = SchedulerOutput(prefill_sequences=[seq], decode_sequences=[])
    batch = build_batch(schedule, prefill_chunk_size=4)

    assert batch.token_ids           == [0, 1, 2, 3], f"wrong tokens: {batch.token_ids}"
    assert batch.positions           == [0, 1, 2, 3], f"wrong positions: {batch.positions}"
    assert batch.kv_cache_offsets[0] == 0,            "kv_offset should be 0 for first chunk"


def test_build_batch_partial_last_chunk():
    """
    Final chunk that is shorter than chunk_size: 11-token prompt with chunk_size=4.
    Last chunk covers tokens [8, 9, 10] only (3 tokens, not 4).
    """
    seq = Sequence(
        seq_id=3,
        prompt_token_ids=list(range(11)),
        sampling_params=SamplingParams(),
        status=SequenceStatus.PREFILL,
        num_computed_tokens=8,
        block_table=[0, 1, 2],
    )
    schedule = SchedulerOutput(prefill_sequences=[seq], decode_sequences=[])
    batch = build_batch(schedule, prefill_chunk_size=4)

    assert batch.token_ids  == [8, 9, 10],  f"wrong tokens: {batch.token_ids}"
    assert batch.positions  == [8, 9, 10],  f"wrong positions: {batch.positions}"
    assert batch.seq_lens   == [3],         f"wrong seq_lens: {batch.seq_lens}"
    assert batch.kv_cache_offsets[0] == 8


def test_build_batch_decode_token_and_position():
    """
    Decode sequence with prompt=[1,2,3] and output=[7,8]:
      The input token is the last output token (8, at position 4).
      kv_offset = num_kv_tokens - 1 = 4.
    """
    seq = Sequence(
        seq_id=4,
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(),
        status=SequenceStatus.DECODE,
        output_token_ids=[7, 8],
        block_table=[0],
    )
    schedule = SchedulerOutput(prefill_sequences=[], decode_sequences=[seq])
    batch = build_batch(schedule, prefill_chunk_size=4)

    assert batch.token_ids           == [8],  f"wrong decode token: {batch.token_ids}"
    assert batch.positions           == [4],  f"wrong position (expected num_kv_tokens-1=4): {batch.positions}"
    assert batch.kv_cache_offsets[0] == 4,   f"wrong kv offset: {batch.kv_cache_offsets[0]}"
    assert batch.seq_lens            == [1]
    assert batch.num_prefill_seqs    == 0


def test_build_batch_mixed_prefill_and_decode():
    """
    One PREFILL sequence (chunk 2 of 12-token prompt) + one DECODE sequence
    packed into a single batch. Prefill tokens come first, decode token last.

    prefill_seq: num_computed=4, prompt=[0..11] → tokens [4,5,6,7], positions [4,5,6,7]
    decode_seq:  prompt=[50,51], output=[99]    → token [99], position [2]
    """
    prefill_seq = Sequence(
        seq_id=10,
        prompt_token_ids=list(range(12)),
        sampling_params=SamplingParams(),
        status=SequenceStatus.PREFILL,
        num_computed_tokens=4,
        block_table=[0, 1],
    )
    decode_seq = Sequence(
        seq_id=11,
        prompt_token_ids=[50, 51],
        sampling_params=SamplingParams(),
        status=SequenceStatus.DECODE,
        output_token_ids=[99],
        block_table=[2],
    )
    schedule = SchedulerOutput(prefill_sequences=[prefill_seq], decode_sequences=[decode_seq])
    batch = build_batch(schedule, prefill_chunk_size=4)

    assert batch.token_ids        == [4, 5, 6, 7, 99], f"wrong token order: {batch.token_ids}"
    assert batch.positions        == [4, 5, 6, 7, 2],  f"wrong positions: {batch.positions}"
    assert batch.seq_lens         == [4, 1],            f"wrong seq_lens: {batch.seq_lens}"
    assert batch.kv_cache_offsets == [4, 2],            f"wrong kv offsets: {batch.kv_cache_offsets}"
    assert batch.num_prefill_seqs == 1


# ── Integrated end-to-end test ────────────────────────────────────────────────

def test_integrated_chunked_prefill_loop():
    """
    Full engine loop with chunked prefill — scheduler + KV cache + sampler,
    no real model forward pass.

    seq0: 8-token prompt (2 chunks of 4), generates [3, EOS]
    seq1: 3-token prompt (1 chunk),        generates [5, 6, EOS]

    Key events:
      iter 0: both in prefill. seq1 completes its only chunk → enters DECODE.
      iter 1: seq0 still on chunk 2 (PREFILL); seq1 decoding alongside it.
              This is the critical continuous-batching step — verified explicitly.
      iter 2: both in DECODE. seq0 samples EOS (finishes). seq1 still running.
      iter 3: seq1 decodes its last token → EOS → finishes.

    Verifies:
      - KV gathered shape is correct at every step and every layer
      - seq1 decodes on iter 1 while seq0 is still prefilling
      - output_token_ids for both sequences are exactly correct
      - All blocks are freed after both sequences finish
    """
    VOCAB = 1000

    model_cfg  = ModelConfig(num_layers=2, num_kv_heads=2, head_dim=4, eos_token_id=EOS)
    server_cfg = make_cfg(max_num_blocks=10)
    scheduler  = Scheduler(model_cfg, server_cfg)
    allocator  = scheduler.allocator

    seq0 = Sequence(seq_id=0, prompt_token_ids=list(range(8)), sampling_params=SamplingParams(max_tokens=10))
    seq1 = Sequence(seq_id=1, prompt_token_ids=list(range(3)), sampling_params=SamplingParams(max_tokens=10))
    scheduler.add_sequence(seq0)
    scheduler.add_sequence(seq1)

    token_schedule = {0: [3, EOS], 1: [5, 6, EOS]}
    all_completed  = []

    for iteration in range(20):
        schedule = scheduler.step()
        if not schedule.prefill_sequences and not schedule.decode_sequences:
            break

        batch    = build_batch(schedule, prefill_chunk_size=CHUNK_SIZE)
        all_seqs = schedule.prefill_sequences + schedule.decode_sequences

        # Fake KV write: K value for each packed token = that token's id
        keys_np = np.zeros((1, 2, len(batch.token_ids), 4), dtype=np.float16)
        for t, tid in enumerate(batch.token_ids):
            keys_np[0, :, t, :] = float(tid)
        keys = mx.array(keys_np)

        # Verify gathered KV shape at every layer
        expected_kv = sum(off + sl for off, sl in zip(batch.kv_cache_offsets, batch.seq_lens))
        for layer in range(model_cfg.num_layers):
            cache = BatchPagedKVCache(
                batch.block_tables, batch.kv_cache_offsets, layer, allocator, batch.seq_lens
            )
            gathered_k, _ = cache.update_and_fetch(keys, keys)
            assert gathered_k.shape == (1, 2, expected_kv, 4), \
                f"iter {iteration} layer {layer}: shape {gathered_k.shape}, expected total_kv={expected_kv}"

        # iter 1: seq0 must still be prefilling, seq1 must be decoding alongside it
        if iteration == 1:
            assert seq0 in schedule.prefill_sequences, \
                "seq0 should still be on chunk 2 at iter 1"
            assert seq1 in schedule.decode_sequences, \
                "seq1 must decode while seq0 is still prefilling — continuous batching broken"

        # Build fake logits: spike at the next predetermined token for each sequence.
        # output_token_ids length tells us how many tokens have been committed —
        # which is the index into token_schedule.
        logits_np = np.zeros((len(all_seqs), VOCAB), dtype=np.float32)
        for idx, seq in enumerate(all_seqs):
            next_tok = token_schedule[seq.seq_id][len(seq.output_token_ids)]
            logits_np[idx, next_tok] = 100.0

        next_tokens = sampler.sample(mx.array(logits_np), all_seqs)
        completed   = scheduler.update(next_tokens)
        all_completed.extend(completed)

    assert len(all_completed) == 2, f"expected 2 completed sequences, got {len(all_completed)}"
    result = {s.seq_id: s for s in all_completed}
    assert result[0].output_token_ids == [3, EOS],    f"seq0: {result[0].output_token_ids}"
    assert result[1].output_token_ids == [5, 6, EOS], f"seq1: {result[1].output_token_ids}"
    assert allocator.num_free_blocks == 10, \
        f"expected all 10 blocks free after completion, got {allocator.num_free_blocks}"


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Continuous batching v2 tests\n")

    scheduler_tests = [
        ("chunked prefill: num_computed_tokens advances per step",   test_chunked_prefill_advances_per_step),
        ("chunked prefill: admission allocates first chunk only",     test_admission_allocates_first_chunk_only),
        ("chunked prefill: blocks allocated incrementally",           test_blocks_allocated_incrementally),
        ("throughput: decode runs alongside every prefill chunk",     test_decode_runs_alongside_every_prefill_chunk),
        ("throughput: no output tokens on intermediate chunks",       test_no_output_tokens_on_intermediate_chunks),
        ("preemption: num_computed_tokens resets to 0",               test_preemption_resets_num_computed_tokens),
    ]
    runner_tests = [
        ("build_batch: mid-chunk slice and positions",                test_build_batch_mid_chunk_slice),
        ("build_batch: first chunk (offset=0)",                       test_build_batch_first_chunk),
        ("build_batch: partial last chunk",                           test_build_batch_partial_last_chunk),
        ("build_batch: decode token and position",                    test_build_batch_decode_token_and_position),
        ("build_batch: mixed prefill + decode ordering",              test_build_batch_mixed_prefill_and_decode),
    ]
    e2e_tests = [
        ("integrated: chunked prefill end-to-end loop",              test_integrated_chunked_prefill_loop),
    ]

    print("── Scheduler ────────────────────────────────────")
    for name, fn in scheduler_tests:
        run(name, fn)

    print("\n── Runner / build_batch ─────────────────────────")
    for name, fn in runner_tests:
        run(name, fn)

    print("\n── End-to-end ───────────────────────────────────")
    for name, fn in e2e_tests:
        run(name, fn)