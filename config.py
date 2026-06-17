from dataclasses import dataclass

# Current defaults are based on Llama 3.1
@dataclass
class ModelConfig:
    num_layers: int = 28
    d_model: int = 3072
    num_q_heads: int = 24
    num_kv_heads: int = 8
    head_dim: int = 128
    ffn_hidden: int = 8192
    vocab_size: int = 128256
    max_seq_len: int = 131072
    rms_norm_eps: float = 1e-5
    eos_token_id: int = 128009


# These values need to be based on actual server configurations.
@dataclass
class ServerConfig:
    model_path: str
    block_size: int = 16
    max_num_blocks: int = 2048
    max_num_seqs: int = 64
    max_seq_len: int = 8192
    host: str = "0.0.0.0"
    port: int = 8000
