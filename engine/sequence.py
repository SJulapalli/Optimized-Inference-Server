from enum import Enum, auto
from dataclasses import dataclass, field



class SequenceStatus(Enum):
    WAITING    = auto()
    PREFILL    = auto()
    DECODE     = auto()
    PREEMPTED  = auto()
    FINISHED   = auto()
    
@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1        # -1 means disabled
    max_tokens: int = 256

@dataclass
class Sequence:
    seq_id: int                          # unique identifier
    prompt_token_ids: list[int]          # tokenized prompt, never mutated
    sampling_params: SamplingParams

    status: SequenceStatus = SequenceStatus.WAITING

    output_token_ids: list[int] = field(default_factory=list)  # generated tokens so far
    block_table: list[int] = field(default_factory=list)       # physical block indices, in order

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self.output_token_ids)

    @property
    def num_kv_tokens(self) -> int:
        # how many positions have been written into the KV cache
        # during prefill this grows from 0 to num_prompt_tokens
        # during decode it grows by 1 each step
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    @property
    def next_token_position(self) -> int:
        # the absolute position index of the next token to be generated
        return self.num_kv_tokens

    def get_last_token_id(self) -> int:
        if self.output_token_ids:
            return self.output_token_ids[-1]
        return self.prompt_token_ids[-1]
