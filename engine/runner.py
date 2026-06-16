from dataclasses import dataclass
from scheduler import SchedulerOutput
from block_allocator import BlockAllocator
import mlx.core as mx
import numpy as np

@dataclass
class Batch:
    token_ids: list[int]          # all tokens packed: [*prefill_0_tokens, ..., *prefill_N_tokens, decode_0_token, ...]
    positions: list[int]          # absolute position of each token in its own sequence (for RoPE)
    block_tables: list[list[int]] # one block_table per sequence, in the same order as sequences appear in token_ids
    seq_lens: list[int]           # number of tokens contributed by each sequence (prompt_len for prefill, 1 for decode)
    num_prefill_seqs: int         # how many sequences at the front are prefill; the rest are decode
    
# Positions aren't deeply managed yet, but will need to be managed once chunked inputs start being handled.
def build_batch(schedule: SchedulerOutput):
    sequences = schedule.prefill_sequences + schedule.decode_sequences
    token_ids = [token for prefill_sequence in schedule.prefill_sequences for token in prefill_sequence.prompt_token_ids]
    token_ids += [decode_sequence.get_last_token_id() for decode_sequence in schedule.decode_sequences]
    
    positions = [i for sequence in schedule.prefill_sequences for i in range(sequence.num_kv_tokens)] + [sequence.num_kv_tokens - 1 for sequence in schedule.decode_sequences]
    seq_lens = [sequence.num_prompt_tokens for sequence in schedule.prefill_sequences] + [1 for _ in schedule.decode_sequences]
    
    block_tables = [sequence.block_table for sequence in sequences]
    num_prefill_seqs = len(schedule.prefill_sequences)
    
    return Batch(token_ids=token_ids, positions=positions, block_tables=block_tables, seq_lens=seq_lens, num_prefill_seqs=num_prefill_seqs)

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