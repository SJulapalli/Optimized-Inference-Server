from dataclasses import dataclass
from scheduler import SchedulerOutput

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
    def __init__(self):
        super()