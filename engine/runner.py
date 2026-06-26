from dataclasses import dataclass
from scheduler import SchedulerOutput, Sequence, SequenceStatus
from .block_allocator import BlockAllocator
from model.llama import Model
from sampling.sampler import sample
import mlx.core as mx
import numpy as np

@dataclass
class Batch:
    token_ids: list[int]          # all tokens packed: [*prefill_0_tokens, ..., *prefill_N_tokens, decode_0_token, ...]
    positions: list[int]          # absolute position of each token in its own sequence (for RoPE)
    block_tables: list[list[int]] # one block_table per sequence, in the same order as sequences appear in token_ids
    seq_lens: list[int]           # number of tokens contributed by each sequence (prompt_len for prefill, 1 for decode)
    num_prefill_seqs: int         # how many sequences at the front are prefill; the rest are decode
    kv_cache_offsets: list[int]
    
def build_batch(schedule: SchedulerOutput, prefill_chunk_size: int):
    # v2 PREFILL path
    seq_lens = []
    token_ids = []
    positions = []
    kv_cache_offsets = []
    
    for seq in schedule.prefill_sequences:
        chunk_len = min(prefill_chunk_size, seq.num_prompt_tokens - seq.num_computed_tokens)
        token_ids  += seq.prompt_token_ids[seq.num_computed_tokens : seq.num_computed_tokens + chunk_len]
        positions  += list(range(seq.num_computed_tokens, seq.num_computed_tokens + chunk_len))
        seq_lens.append(chunk_len)
        kv_cache_offsets.append(seq.num_computed_tokens)   # was always 0 in v1

    for seq in schedule.decode_sequences:
        token_ids.append(seq.get_last_token_id())
        positions.append(seq.num_kv_tokens - 1)
        seq_lens.append(1)
        kv_cache_offsets.append(seq.num_kv_tokens - 1)

    block_tables = [sequence.block_table for sequence in schedule.prefill_sequences + schedule.decode_sequences]
    num_prefill_seqs = len(schedule.prefill_sequences)
        
    return Batch(token_ids=token_ids, positions=positions, block_tables=block_tables, seq_lens=seq_lens, num_prefill_seqs=num_prefill_seqs, kv_cache_offsets=kv_cache_offsets)

class BatchPagedKVCache:
    
    # All inputs are expected to be in prompt sequence order
    def __init__(self, block_tables: list[list[int]], kv_cache_offsets: list[int], layer: int, block_allocator: BlockAllocator, seq_lens: list[int], block_indices):
        self.block_tables = block_tables            # Array of block tables
        self.block_allocator = block_allocator      # The allocator managing the global cache memory
        self.kv_cache_offsets = kv_cache_offsets    # The current offset of the sequence in its local KV Cache (basically just 2 * num_kv_vectors I think)
        self.layer = layer                          # The attention layer being cached for
        self.seq_lens = seq_lens                    # The lengths of the incoming prompts
        self.num_sequences = len(block_tables)      # The number of sequences.
        self.block_indices = block_indices
        
    # Keys Shape: (B, num_heads, L, head_dim)
    def update_and_fetch(self, keys: mx.array, values: mx.array):        
        # seq_starts = np.cumsum(self.seq_lens) - self.seq_lens
        
        # (L, num_heads, D)
        reshaped_keys = keys[0].transpose(1, 0, 2)
        reshaped_values = values[0].transpose(1, 0, 2)
        
        # Pool Shape: (num_layers, 2, max_num_blocks * block_size, num_kv_heads, D)
        
        pool = self.block_allocator.pool[self.layer]            # (2, slots, H, D)
        for k in range(len(self.block_indices)):
            slot = int(self.block_indices[k])
            pool = mx.slice_update(pool, reshaped_keys[k][None, None],   mx.array([0, slot, 0, 0]), axes=[0,1,2,3])
            pool = mx.slice_update(pool, reshaped_values[k][None, None], mx.array([1, slot, 0, 0]), axes=[0,1,2,3])
        self.block_allocator.pool[self.layer] = pool            # rebind, before the gather
                
        keys = []
        values = []
        
        for i, block_table in enumerate(self.block_tables):
            slots = (mx.array(block_table)[:, None] * self.block_allocator.block_size + mx.arange(self.block_allocator.block_size)).reshape(-1)[:self.kv_cache_offsets[i] + self.seq_lens[i]]
            keys.append(self.block_allocator.pool[self.layer][0, slots])     # (n_kv, H, D)
            values.append(self.block_allocator.pool[self.layer][1, slots])
            
        keys = (mx.concatenate(keys)).transpose(1, 0, 2)
        values = (mx.concatenate(values)).transpose(1, 0, 2)
        
        return keys[None], values[None]
    
def compute_kv_offsets(sequences: list[Sequence]) -> list[int]:
    return [0 if seq.status == SequenceStatus.PREFILL else seq.num_kv_tokens - 1 for seq in sequences]

def build_block_diagonal_mask(seq_lens: list[int], kv_offsets: list[int], num_prefill_seqs: int) -> mx.array:
    # Derive kv_lens, total_tokens, total_kv_tokens
    total_tokens = np.sum(seq_lens)
    total_kv_tokens = total_tokens + np.sum(kv_offsets)
    # Initialize mask as all -inf, shape [total_tokens, total_kv_tokens]
    mask = np.full((total_tokens, total_kv_tokens), -np.inf)

    q_start = 0
    kv_start = 0

    for i, (seq_len, kv_offset) in enumerate(zip(seq_lens, kv_offsets)):
        kv_len = kv_offset + seq_len
        q_end = q_start + seq_len
        kv_end = kv_start + kv_len

        if i < num_prefill_seqs:
            # Fill causal triangular block at mask[q_start:q_end, kv_start:kv_end]
            # print(q_start, q_end, kv_start, kv_end, mask.shape)
            # print(mask[q_start:q_end, kv_start:kv_end])
            mask[q_start:q_end, kv_start:kv_end] = np.triu(mask[q_start:q_end, kv_start:kv_end], k=kv_offset + 1)
        else:
            # Fill full block at mask[q_start:q_end, kv_start:kv_end] with zeros
            mask[q_start:q_end, kv_start:kv_end] = 0

        q_start = q_end
        kv_start = kv_end

    return mx.array(mask[None, None].astype(np.float16))

class ModelRunner:
    def __init__(self, model:Model, block_allocator: BlockAllocator):
        self.model = model
        self.block_allocator = block_allocator
        self.num_layers = model.args.num_hidden_layers
    
    def forward(self, batch: Batch):
        # Build the per-layer BatchedPagedKVCache
        kv_cache_offsets = batch.kv_cache_offsets
        
        block_indices = []
        for i in range(len(batch.seq_lens)):
            p = np.arange(batch.seq_lens[i]) + batch.kv_cache_offsets[i]
            cache_block = p // self.block_allocator.block_size
            inner_index = p % self.block_allocator.block_size
            block_table = np.array(batch.block_tables[i])
            
            block_indices.append(block_table[cache_block] * self.block_allocator.block_size + inner_index)
        
        block_indices = np.concatenate(block_indices)
        
        cache = [BatchPagedKVCache(batch.block_tables, kv_cache_offsets=kv_cache_offsets, layer=i, block_allocator=self.block_allocator, seq_lens=batch.seq_lens, block_indices=block_indices) for i in range(self.num_layers)]
        mask = build_block_diagonal_mask(seq_lens=batch.seq_lens, kv_offsets=kv_cache_offsets, num_prefill_seqs=batch.num_prefill_seqs)
        
        out_logits = self.model(inputs=mx.array(batch.token_ids)[None, :], cache=cache, positions=mx.array(batch.positions), block_mask=mask)
        
        seq_offset = -1
        outputs = []
        for seq_len in batch.seq_lens:
            seq_offset += seq_len
            outputs += [out_logits[0, seq_offset, :]]
            
        return mx.stack(outputs)