# Paged KV Cache — Copy Pathology Postmortem

How the paged KV cache went from a 5× deficit, through a catastrophic ~4.8s/token
regression, to a flat in-place design — and the one MLX behavior that caused all of it.

## 0. The one MLX fact everything traces back to

MLX arrays are **immutable in the graph sense**. Operations don't mutate — they build a
lazy graph node, and values are computed at `mx.eval()`. There is **no in-place mutation API**:

- `a[idx] = b` (`__setitem__`), `mx.slice_update(...)`, and `mx.scatter(...)` all produce a
  **new array node**. Conceptually: "copy `a`, apply the update."
- The only thing that saves you from a literal copy is **buffer donation**: at `eval` time,
  MLX may reuse an input's buffer for the output — *but only if that input has no other live
  consumer in the graph*. If anything else still needs the old value, MLX must keep it intact,
  so it allocates a fresh buffer and copies.

So "in-place" in MLX is not a property of the call you write — it's an *emergent* property of
whether the old array is dead at eval time. **This single rule explains every number below.**

## 1. The original implementation — slow, for ordinary reasons

The pre-Stage-1 cache (one monolithic 6-D pool):

```python
# pool shape: (num_blocks, num_layers, 2, num_kv_heads, block_size, head_dim)  -- ONE giant array
def update_and_fetch(self, keys, values):
    seq_starts = np.cumsum(self.seq_lens) - self.seq_lens

    # --- SCATTER: a Python double loop, one tiny op per token, per layer ---
    for i, seq_start in enumerate(seq_starts):
        for t in range(self.seq_lens[i]):                       # <-- per-token Python loop
            cache_block = (self.kv_cache_offsets[i] + t) // self.block_allocator.block_size
            inner_index = (self.kv_cache_offsets[i] + t) %  self.block_allocator.block_size
            mem_block_id = self.block_tables[i][cache_block]
            self.block_allocator.pool[mem_block_id, self.layer, 0, :, inner_index, :] = keys[0, :, seq_start + t, :]
            self.block_allocator.pool[mem_block_id, self.layer, 1, :, inner_index, :] = values[0, :, seq_start + t, :]

    # --- GATHER: rebuild the ENTIRE history every layer, every step ---
    keys, values = [], []
    for i, block_table in enumerate(self.block_tables):
        cached_kv = self.block_allocator.pool[block_table, self.layer, :, :, :, :]   # gather this seq's blocks
        num_kv_tokens = self.kv_cache_offsets[i] + self.seq_lens[i]
        cached_kv_flattened = cached_kv.transpose((1, 2, 0, 3, 4)).flatten(start_axis=2, end_axis=3)
        cached_kv_flattened = cached_kv_flattened[:, :, :num_kv_tokens, :]
        keys  += [cached_kv_flattened[0]]
        values += [cached_kv_flattened[1]]
    keys   = mx.concatenate(keys,   axis=1)
    values = mx.concatenate(values, axis=1)
    return keys[None], values[None]
```

The *visible* problems:
- The scatter launches two tiny ops per token — fine for decode (~8 tokens), bad for prefill (thousands).
- The gather **re-reads and re-assembles the entire KV history on every layer of every step**
  (`transpose → flatten → truncate → concat`). O(total_kv) of redundant work per layer — the real
  reason it was ~5× slower than mlx_lm.

The profiler showed `fwd` was ~100% of the step, and the ablation (`benchmark.py --mode contiguous`)
proved the model and orchestration were fine — all the deficit was this method. Stage 1 set out to
vectorize it, and MLX's donation rule ambushed us.

## 2. Stage 1, attempt #1 — fancy-index scatter into the giant pool

Flat-slot pool `(num_layers, 2, num_slots, H, D)`, one vectorized scatter:

```python
block_indices = mx.concatenate(block_indices)            # mx.array of every new token's physical slot
self.block_allocator.pool[self.layer, 0, block_indices] = reshaped_keys      # <-- fancy (array) index
self.block_allocator.pool[self.layer, 1, block_indices] = reshaped_values
```

**Symptom:** a *single* decode token took **4.8 seconds**, and step time scaled with `--max-num-blocks`.

**Why:** array-index assignment lowers to the `scatter` primitive. To preserve untouched elements it
needs the whole input pool; the pool is persistent (referenced for the next step), so it **cannot
donate → it copies all 3.76 GB**, twice per layer (K and V) × 28 layers.

## 3. Stage 1, attempt #2 — `slice_update` into the giant pool (the trap that looked right)

Contiguous slice writes were *not* in-place either. The microbench is the receipt:

```python
for k, slot in enumerate(block_indices):
    self.block_allocator.pool[self.layer, 0, slot:slot+1] = reshaped_keys[k:k+1]   # __setitem__ slice
    self.block_allocator.pool[self.layer, 1, slot:slot+1] = reshaped_values[k:k+1]
```

```
# cost of ONE write + eval, at two pool sizes
slice_write (__setitem__):  0.44ms @128  →  4.98ms @2048    # scales with pool size = COPYING
gather (read):              0.16ms @128  →  0.16ms @2048    # flat = innocent
```

The *functional* form with a rebind looked flat in isolation:

```python
pool = self.block_allocator.pool
for k, slot in enumerate(block_indices):
    pool = mx.slice_update(pool, reshaped_keys[k][None,None,None],   mx.array([self.layer,0,slot,0,0]), axes=[0,1,2,3,4])
    pool = mx.slice_update(pool, reshaped_values[k][None,None,None], mx.array([self.layer,1,slot,0,0]), axes=[0,1,2,3,4])
self.block_allocator.pool = pool
```

```
# chained slice_updates, NO interleaved gather:  FLAT (1.13ms @2048)
# but the REAL forward still scaled: 3949ms @2048 vs 190ms @128
```

**The decisive insight — it was never just the write, it was the read+write *interleaving* on a
shared array.** Each layer does scatter then gather on the *same* pool. Model that and the catastrophe
reappears:

```
# per layer: write then gather, one eval
giant pool    : N=128 ->   62ms     N=2048 -> 10508ms     # 170× — every layer copies all 3.76 GB
per-layer pool: N=128 -> 0.86ms     N=2048 ->  1.26ms     # FLAT
```

Why the giant pool can't donate: the layers form **one dependency chain through a single array** —
`P0 → P1 → … → P27`. Each version `P_L` is consumed by **two** things: layer L's *gather* (reads `P_L`)
and layer L+1's *scatter* (chains from `P_L`). Two live consumers ⇒ donation forbidden ⇒ `P_L` is
copied. And because it's the monolithic pool, each copy duplicates **all 28 layers' data**.

## 4. The fix — a list of per-layer pools

```python
# block_allocator.py — a LIST of small arrays, not one block
def create_block_pool(model_config, server_config):
    N = server_config.max_num_blocks * server_config.block_size
    return [mx.zeros((2, N, num_kv_heads, head_dim), dtype=mx.float16)   # (2, slots, H, D) per layer
            for _ in range(model_config.num_layers)]
```

```python
# runner.py — scatter chains slice_update on THIS layer's array only, rebinds once
pool = self.block_allocator.pool[self.layer]            # (2, slots, H, D)  -- small
for k in range(len(block_indices)):
    slot = int(block_indices[k])                        # numpy/host int -> no device sync
    pool = mx.slice_update(pool, reshaped_keys[k][None, None],   mx.array([0, slot, 0, 0]), axes=[0,1,2,3])
    pool = mx.slice_update(pool, reshaped_values[k][None, None], mx.array([1, slot, 0, 0]), axes=[0,1,2,3])
self.block_allocator.pool[self.layer] = pool            # rebind once, before the gather reads it
```

**Why this donates and the monolith doesn't:** splitting per layer **breaks the cross-layer chain**.
Layer L reads and writes `pools[L]`, a *separate* array from `pools[L+1]`. So `pools[L]`'s previous
version is consumed by exactly one transforming op — layer L's scatter — which lets MLX donate its
buffer and write in place. The gather then *reads* the result (reads never block donation of the
producer, and produce only a small output). No array version has two live consumers anymore.

```
this_step:  135.7ms @128   vs   137.0ms @2048      # FLAT — the copy is dead
```

This is the same reason **mlx_lm** keeps a separate cache array per layer: a per-layer array is the
unit MLX can actually donate, and even if it *did* copy, a per-layer array is small.

## 5. Where the implementation goes from here

The copy is gone, but the cache is not yet optimal — remaining levers, in priority order:

1. **The gather still re-reads all KV every step** (O(total_kv) per layer). Flat in *pool size* now,
   but still grows with *sequence length*. A Stage-2 `mx.fast.metal_kernel` paged-attention kernel
   would remove it (read the pool through the block table, streaming-softmax, no gather). Only worth
   it if the sub-timers show the gather dominating at concurrency.
2. **Packed block-diagonal decode is O(B²·len).** The packed `(1, H, total_kv, D)` + block-diagonal
   mask makes SDPA compute a `B × total_kv` score matrix, mostly masked. Moving decode to a true
   batched `(B, H, L_max, D)` + padding mask makes it O(B·len). The lever for high concurrency.
3. **Prefill op-launch count.** Per-token `slice_update` is fine for decode (B writes) but launches
   `L` ops for a big prefill chunk. Coalesce each sequence's chunk into per-block contiguous runs
   (one `slice_update` per run) — purely fewer launches, same in-place behavior.
4. **Hoist `block_indices`/`slots`** — recomputed every layer though they're pure functions of block
   tables + offsets. Cheap now (host NumPy), but trivially hoistable to once per step.
5. **Keep `test_paged_cache_isolated.py` as the regression guard.** The correctness oracle is
   *logit-closeness*, not exact text — fp16 ties legitimately flip greedy argmax between the paged and
   contiguous paths.

## The transferable lesson

In MLX, any persistent buffer you both read and write within one eval — especially a large shared one
touched by every layer — will silently copy. Keep mutable state in the **smallest independent arrays**
that match your access granularity (here: per layer), so each update is a single-consumer node MLX can
donate.
