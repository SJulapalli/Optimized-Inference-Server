from dataclasses import dataclass, field
from math import ceil
from engine.sequence import Sequence, SequenceStatus
from engine.block_allocator import BlockAllocator
from config import ModelConfig, ServerConfig

@dataclass
class SchedulerOutput:
    prefill_sequences: list[Sequence] = field(default_factory=list)
    decode_sequences: list[Sequence] = field(default_factory=list)
    blocks_to_free: list[int] = field(default_factory=list)

class Scheduler:
    def __init__(self, model_config: ModelConfig, server_config: ServerConfig):
        self.waiting_sequences: list[Sequence] = []
        self.active_sequences: list[Sequence] = []
        self.model_config = model_config
        self.server_config = server_config
        self.allocator : BlockAllocator = BlockAllocator(model_config, server_config)
        
    def add_sequence(self, sequence: Sequence):
        if len(sequence.prompt_token_ids) > self.server_config.max_seq_len:
            raise ValueError(f"Sequence {sequence.seq_id} is larger than possible to store in allocated KV Cache")
        else:
            self.waiting_sequences.append(sequence)
    
    def step(self):
        i = 0 
        k = 0
        prefill_sequences = []
        decode_sequences = []
        blocks_to_free = []
        
        while k < len(self.active_sequences):
            sequence = self.active_sequences[k]
            if sequence.num_kv_tokens - len(sequence.block_table) * self.server_config.block_size >= 0:
                try:
                    sequence.block_table.extend(self.allocator.allocate(1))
                except MemoryError:
                    self.active_sequences.pop(k)
                    sequence.status = SequenceStatus.WAITING
                    self.allocator.free(sequence.block_table)
                    sequence.block_table = []
                    self.waiting_sequences.insert(0, sequence)
                    continue
            
            if sequence.status == SequenceStatus.PREFILL:
                prefill_sequences.append(sequence)
            elif sequence.status == SequenceStatus.DECODE:
                decode_sequences.append(sequence)
                
            k += 1

        # Loop over waiting sequences
        while i < len(self.waiting_sequences):
            # Check if server settings allow for more sequences to be processed
            if len(self.active_sequences) >= self.server_config.max_num_seqs:
                break
            
            sequence = self.waiting_sequences[i]
            required_blocks = ceil((sequence.num_kv_tokens + 1) / self.server_config.block_size)
            
            # Check if sufficient KV cache blocks are available
            if self.allocator.num_free_blocks >= required_blocks:
                # Allocate blocks and move sequence into active list while removing from waiting list
                allocated_blocks = self.allocator.allocate(required_blocks)
                self.active_sequences.append(sequence)
                self.active_sequences[-1].status = SequenceStatus.PREFILL
                self.active_sequences[-1].block_table = allocated_blocks
                self.waiting_sequences.pop(i)
                
                prefill_sequences.append(self.active_sequences[-1])
            else:
                # Only iterate if sequence doesn't fit to see if other sequences fit. Goal here is to maximize throughput, not fairness.
                i += 1
        
        return SchedulerOutput(prefill_sequences=prefill_sequences, decode_sequences=decode_sequences, blocks_to_free=blocks_to_free)
    
    def update(self, next_tokens: dict[int, int]):
        completed_sequences = []
        i = 0
        while i < len(self.active_sequences):
            sequence = self.active_sequences[i]
            
            # If sequence is preempted, deallocate its blocks and move it to the top of the waiting list (for fairness so the request is considered first).
            if sequence.status == SequenceStatus.PREEMPTED:
                self.active_sequences.pop(i)
                sequence.status = SequenceStatus.WAITING
                self.allocator.free(sequence.block_table)
                sequence.block_table = []
                self.waiting_sequences.insert(0, sequence)
                continue
            
            # Otherwise, all other cases will have the sequence getting its next token added
            sequence.output_token_ids.append(next_tokens[sequence.seq_id])
                
            # Always happens if the token is EOS, seq_len is too long for server or model config, or max token generation limit reached.
            if next_tokens[sequence.seq_id] == self.model_config.eos_token_id or sequence.num_kv_tokens >= self.model_config.max_seq_len or sequence.num_kv_tokens >= self.server_config.max_seq_len or sequence.num_output_tokens >= sequence.sampling_params.max_tokens:
                self.active_sequences.pop(i) # Do we remove or return a sequence once complete? The outputs need to get back to the user somehow so returning it seems most logical
                self.allocator.free(sequence.block_table)
                sequence.block_table = []
                sequence.status = SequenceStatus.FINISHED  # Probably not needed
                completed_sequences.append(sequence)
                continue
            
            # Should no longer be necessary.
            # Check if the sequence needs more space to continue, if it does and we can't provide it then we preempt.
            # if sequence.num_kv_tokens - len(sequence.block_table) * self.server_config.block_size > 0:
            #     try:
            #         sequence.block_table.extend(self.allocator.allocate(1))
            #     except MemoryError:
            #         self.active_sequences.pop(i)
            #         sequence.status = SequenceStatus.WAITING
            #         self.allocator.free(sequence.block_table)
            #         sequence.block_table = []
            #         self.waiting_sequences.insert(0, sequence)
            #         continue
            
            if sequence.status == SequenceStatus.PREFILL:
                sequence.status = SequenceStatus.DECODE
            
            i += 1
        
        return completed_sequences
                