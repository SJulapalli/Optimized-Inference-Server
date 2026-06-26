"""
Isolated test for the per-layer paged-KV-cache design.

Duplicates the scatter/gather logic of BatchPagedKVCache.update_and_fetch, but
the pool is a plain list of per-layer arrays held by this object, with NO
BlockAllocator involved. Verifies:
  1. correctness  -- paged store+gather round-trips values vs a contiguous ref
  2. flatness     -- per-step time is independent of pool size (no full copy)

Run: python test_paged_cache_isolated.py
"""
import time
import numpy as np
import mlx.core as mx

H, D, BLOCK, NUM_LAYERS = 8, 128, 16, 28


class PagedCache:
    """Per-layer pools as separate arrays. block tables / offsets passed in,
    exactly as the real BatchPagedKVCache receives them."""

    def __init__(self, max_num_blocks: int):
        N = max_num_blocks * BLOCK
        self.block_size = BLOCK
        self.pool = [mx.zeros((2, N, H, D), dtype=mx.float16) for _ in range(NUM_LAYERS)]

    def update_and_fetch(self, layer, keys, values, block_tables, kv_offsets, seq_lens):
        # keys/values: (1, H, L, D), packed across sequences
        reshaped_keys = keys[0].transpose(1, 0, 2)      # (L, H, D)
        reshaped_values = values[0].transpose(1, 0, 2)

        # --- scatter (in-place via slice_update chained through a local var) ---
        block_indices = []
        for i in range(len(seq_lens)):
            p = np.arange(seq_lens[i]) + kv_offsets[i]
            bt = np.array(block_tables[i])
            block_indices.append(bt[p // self.block_size] * self.block_size + p % self.block_size)
        block_indices = np.concatenate(block_indices)

        pool = self.pool[layer]                          # (2, slots, H, D)
        for k in range(len(block_indices)):
            slot = int(block_indices[k])
            pool = mx.slice_update(pool, reshaped_keys[k][None, None],   mx.array([0, slot, 0, 0]), axes=[0, 1, 2, 3])
            pool = mx.slice_update(pool, reshaped_values[k][None, None], mx.array([1, slot, 0, 0]), axes=[0, 1, 2, 3])
        self.pool[layer] = pool

        # --- gather ---
        ks, vs = [], []
        for i, bt in enumerate(block_tables):
            n_kv = kv_offsets[i] + seq_lens[i]
            slots = (np.array(bt)[:, None] * self.block_size + np.arange(self.block_size)).reshape(-1)[:n_kv]
            slots = mx.array(slots)
            ks.append(self.pool[layer][0, slots])        # (n_kv, H, D)
            vs.append(self.pool[layer][1, slots])
        ks = mx.concatenate(ks).transpose(1, 0, 2)       # (H, total_kv, D)
        vs = mx.concatenate(vs).transpose(1, 0, 2)
        return ks[None], vs[None]


def _blocks_for(n_tokens):
    return (n_tokens + BLOCK - 1) // BLOCK


def test_correctness():
    """Single + multi sequence, spanning several blocks. Paged gather must
    equal a contiguous reference exactly (it's a pure store/retrieve)."""
    print("=== correctness ===")
    rng = mx.random.normal

    for label, prompts in [("single-seq", [37]), ("multi-seq", [20, 9, 31])]:
        cache = PagedCache(max_num_blocks=64)
        free = list(range(64))
        nseq = len(prompts)
        block_tables = [[] for _ in range(nseq)]
        num_kv = [0] * nseq
        ref_k = [[[] for _ in range(NUM_LAYERS)] for _ in range(nseq)]
        ref_v = [[[] for _ in range(NUM_LAYERS)] for _ in range(nseq)]

        def grow(i, total):
            while len(block_tables[i]) < _blocks_for(total):
                block_tables[i].append(free.pop(0))

        # ---- prefill all sequences (one packed step) ----
        for i in range(nseq):
            grow(i, prompts[i])
        for layer in range(NUM_LAYERS):
            # packed keys: concat each seq's prompt tokens
            per_seq_k = [rng((1, H, prompts[i], D)).astype(mx.float16) for i in range(nseq)]
            per_seq_v = [rng((1, H, prompts[i], D)).astype(mx.float16) for i in range(nseq)]
            keys = mx.concatenate(per_seq_k, axis=2)
            values = mx.concatenate(per_seq_v, axis=2)
            for i in range(nseq):
                for t in range(prompts[i]):
                    ref_k[i][layer].append(per_seq_k[i][0, :, t, :])
                    ref_v[i][layer].append(per_seq_v[i][0, :, t, :])
            gk, gv = cache.update_and_fetch(
                layer, keys, values, block_tables,
                kv_offsets=[0] * nseq, seq_lens=list(prompts))
            # verify per-sequence slices of the packed gather
            off = 0
            for i in range(nseq):
                exp_k = mx.stack(ref_k[i][layer], axis=1)[None]   # (1,H,Li,D)
                got_k = gk[:, :, off:off + prompts[i], :]
                assert mx.allclose(exp_k, got_k), f"{label} prefill keys mismatch seq{i} layer{layer}"
                off += prompts[i]
        for i in range(nseq):
            num_kv[i] = prompts[i]

        # ---- 25 decode steps (crosses block boundaries) ----
        for step in range(25):
            for i in range(nseq):
                grow(i, num_kv[i] + 1)
            for layer in range(NUM_LAYERS):
                per_seq_k = [rng((1, H, 1, D)).astype(mx.float16) for _ in range(nseq)]
                per_seq_v = [rng((1, H, 1, D)).astype(mx.float16) for _ in range(nseq)]
                keys = mx.concatenate(per_seq_k, axis=2)
                values = mx.concatenate(per_seq_v, axis=2)
                for i in range(nseq):
                    ref_k[i][layer].append(per_seq_k[i][0, :, 0, :])
                    ref_v[i][layer].append(per_seq_v[i][0, :, 0, :])
                gk, gv = cache.update_and_fetch(
                    layer, keys, values, block_tables,
                    kv_offsets=[num_kv[i] for i in range(nseq)], seq_lens=[1] * nseq)
                off = 0
                for i in range(nseq):
                    li = num_kv[i] + 1
                    exp_k = mx.stack(ref_k[i][layer], axis=1)[None]
                    exp_v = mx.stack(ref_v[i][layer], axis=1)[None]
                    assert mx.allclose(exp_k, gk[:, :, off:off + li, :]), f"{label} decode keys seq{i} layer{layer} step{step}"
                    assert mx.allclose(exp_v, gv[:, :, off:off + li, :]), f"{label} decode vals seq{i} layer{layer} step{step}"
                    off += li
            for i in range(nseq):
                num_kv[i] += 1
        print(f"  {label}: OK  ({nseq} seq, final lens={[num_kv[i] for i in range(nseq)]})")


def test_flat():
    """Per-step decode time must not scale with pool size."""
    print("=== flatness (decode, 1 seq, all 28 layers/step) ===")

    def bench(max_num_blocks):
        cache = PagedCache(max_num_blocks)
        bt = [list(range(8))]                 # 8 blocks pre-allocated for the seq
        num_kv = 100
        k = mx.random.normal((1, H, 1, D)).astype(mx.float16)
        v = mx.random.normal((1, H, 1, D)).astype(mx.float16)

        def step():
            outs = []
            for layer in range(NUM_LAYERS):
                gk, gv = cache.update_and_fetch(layer, k, v, bt, [num_kv], [1])
                outs.append(gk)
            mx.eval(outs)
        step()  # warmup
        s = time.perf_counter()
        for _ in range(20):
            step()
        mx.synchronize()
        return (time.perf_counter() - s) / 20 * 1000

    t128 = bench(128)
    t2048 = bench(2048)
    pool_gb = NUM_LAYERS * 2 * (2048 * BLOCK) * H * D * 2 / 1e9
    print(f"  N=128  -> {t128:.2f} ms/step")
    print(f"  N=2048 -> {t2048:.2f} ms/step   (pool would be {pool_gb:.2f} GB if monolithic)")
    print(f"  ratio  -> {t2048 / t128:.2f}x  ({'FLAT - copy is gone' if t2048 / t128 < 2 else 'STILL SCALING'})")


if __name__ == "__main__":
    test_correctness()
    test_flat()
