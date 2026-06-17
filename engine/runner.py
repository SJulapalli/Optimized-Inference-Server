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
    
# Positions aren't deeply managed yet, but will need to be managed once chunked inputs start being handled.
def build_batch(schedule: SchedulerOutput):
    sequences = schedule.prefill_sequences + schedule.decode_sequences
    token_ids = [token for prefill_sequence in schedule.prefill_sequences for token in prefill_sequence.prompt_token_ids]
    token_ids += [decode_sequence.get_last_token_id() for decode_sequence in schedule.decode_sequences]
    
    positions = [i for sequence in schedule.prefill_sequences for i in range(sequence.num_kv_tokens)] + [sequence.num_kv_tokens - 1 for sequence in schedule.decode_sequences]
    seq_lens = [sequence.num_prompt_tokens for sequence in schedule.prefill_sequences] + [1 for _ in schedule.decode_sequences]
    
    block_tables = [sequence.block_table for sequence in sequences]
    num_prefill_seqs = len(schedule.prefill_sequences)
    
    kv_cache_offsets = [0 if seq.status == SequenceStatus.PREFILL else seq.num_kv_tokens - 1 for seq in sequences]
    
    return Batch(token_ids=token_ids, positions=positions, block_tables=block_tables, seq_lens=seq_lens, num_prefill_seqs=num_prefill_seqs, kv_cache_offsets=kv_cache_offsets)

class BatchPagedKVCache:
    
    # All inputs are expected to be in prompt sequence order
    def __init__(self, block_tables: list[list[int]], kv_cache_offsets: list[int], layer: int, block_allocator: BlockAllocator, seq_lens: list[int]):
        self.block_tables = block_tables            # Array of block tables
        self.block_allocator = block_allocator      # The allocator managing the global cache memory
        self.kv_cache_offsets = kv_cache_offsets    # The current offset of the sequence in its local KV Cache (basically just 2 * num_kv_vectors I think)
        self.layer = layer                          # The attention layer being cached for
        self.seq_lens = seq_lens                    # The lengths of the incoming prompts
        self.num_sequences = len(block_tables)      # The number of sequences.
        
    def update_and_fetch(self, keys: mx.array, values: mx.array):        
        seq_starts = np.cumsum(self.seq_lens) - self.seq_lens
        
        for i, seq_start in enumerate(seq_starts):
            for t in range(self.seq_lens[i]):
                cache_block = (self.kv_cache_offsets[i] + t) // self.block_allocator.block_size
                inner_index = (self.kv_cache_offsets[i] + t) % self.block_allocator.block_size
                mem_block_id = self.block_tables[i][cache_block]

                self.block_allocator.pool[mem_block_id, self.layer, 0, :, inner_index, :] = keys[0, :, seq_start + t, :]
                self.block_allocator.pool[mem_block_id, self.layer, 1, :, inner_index, :] = values[0, :, seq_start + t, :]
        
        keys = []
        values = []
        
        for i, block_table in enumerate(self.block_tables):
            # Blocks, kv, heads, block_size, hidden_dim
            cached_kv = self.block_allocator.pool[block_table, self.layer, :, :, :, :]
            num_kv_tokens = self.kv_cache_offsets[i] + self.seq_lens[i]
             
            # kv, blocks, heads, block_size, hidden_dim
            cached_kv_flattened = cached_kv.transpose((1, 2, 0, 3, 4)).flatten(start_axis=2, end_axis=3)
            
            # kv, heads, num_kv_tokens, head_dim
            cached_kv_flattened = cached_kv_flattened[:,:, : num_kv_tokens, :]
            
            # heads, sequence_len, head_dim
            keys += [cached_kv_flattened[0]]
            values += [cached_kv_flattened[1]]
            
        keys = mx.concatenate(keys, axis=1)
        values = mx.concatenate(values, axis=1)
        
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
        cache = [BatchPagedKVCache(batch.block_tables, kv_cache_offsets=kv_cache_offsets, layer=i, block_allocator=self.block_allocator, seq_lens=batch.seq_lens) for i in range(self.num_layers)]
        mask = build_block_diagonal_mask(seq_lens=batch.seq_lens, kv_offsets=kv_cache_offsets, num_prefill_seqs=batch.num_prefill_seqs)
        
        out_logits = self.model(inputs=mx.array(batch.token_ids)[None, :], cache=cache, positions=mx.array(batch.positions), block_mask=mask)
        
        seq_offset = -1
        outputs = []
        for seq_len in batch.seq_lens:
            seq_offset += seq_len
            outputs += [out_logits[0, seq_offset, :]]
            
        return mx.stack(outputs)