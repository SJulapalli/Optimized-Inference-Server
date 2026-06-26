import mlx.core as mx
from config import ModelConfig, ServerConfig

def create_block_pool(model_config, server_config):
    N = server_config.max_num_blocks * server_config.block_size
    return [mx.zeros((2, N, model_config.num_kv_heads, model_config.head_dim), dtype=mx.float16)
            for _ in range(model_config.num_layers)]

class BlockAllocator:
    def __init__(self, model_config: ModelConfig, server_config: ServerConfig):
        self.block_size = server_config.block_size
        self.pool = create_block_pool(model_config, server_config)
        self.free_blocks: list[int] = list(range(server_config.max_num_blocks))
        
    # Returns allocated block indices and removes them from the free_blocks list
    def allocate(self, num_blocks:int) -> list[int]:
        if num_blocks > len(self.free_blocks):
            raise MemoryError(f"KV cache OOM: requested {num_blocks} blocks, {len(self.free_blocks)} available")
        allocated = self.free_blocks[:num_blocks]
        self.free_blocks = self.free_blocks[num_blocks:]
        
        return allocated
    
    def free(self, block_ids: list[int]) -> None:
        self.free_blocks.extend(block_ids)
    
    @property
    def num_free_blocks(self) -> int:
        return len(self.free_blocks)