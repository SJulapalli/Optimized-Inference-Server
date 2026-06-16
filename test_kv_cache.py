"""
Tests for BatchPagedKVCache: write, gather, and end-to-end.

Uses small dims for easy manual reasoning:
  num_layers   = 2
  num_kv_heads = 2
  block_size   = 4
  head_dim     = 4
  max_blocks   = 8
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'engine'))

import numpy as np
import mlx.core as mx

from config import ModelConfig, ServerConfig
from engine.block_allocator import BlockAllocator
from engine.runner import BatchPagedKVCache, build_batch
from engine.sequence import Sequence, SamplingParams
from scheduler import Scheduler
from sampling import sampler

NUM_LAYERS   = 2
NUM_KV_HEADS = 2
BLOCK_SIZE   = 4
HEAD_DIM     = 4
MAX_BLOCKS   = 8


def make_allocator():
    model_cfg  = ModelConfig(num_layers=NUM_LAYERS, num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM)
    server_cfg = ServerConfig(model_path="", block_size=BLOCK_SIZE, max_num_blocks=MAX_BLOCKS)
    return BlockAllocator(model_cfg, server_cfg)


def make_keys(total_tokens, fill):
    """All tokens get the same fill value across all heads and dims."""
    arr = np.full((1, NUM_KV_HEADS, total_tokens, HEAD_DIM), fill, dtype=np.float16)
    return mx.array(arr)


def make_keys_per_token(total_tokens):
    """Token t gets value float(t + 1) across all heads and dims."""
    arr = np.zeros((1, NUM_KV_HEADS, total_tokens, HEAD_DIM), dtype=np.float16)
    for t in range(total_tokens):
        arr[0, :, t, :] = float(t + 1)
    return mx.array(arr)


# ---------------------------------------------------------------------------
# Write tests
# ---------------------------------------------------------------------------

def test_write_single_decode_token():
    """A decode token at offset 3 lands in slot 3 of the correct block."""
    allocator = make_allocator()
    # offset=3 → slot = 3 % 4 = 3, block_idx = 3 // 4 = 0, physical = block_table[0][0] = 0
    cache = BatchPagedKVCache([[0]], [3], 0, allocator, [1])

    keys   = make_keys(1, fill=7.0)
    values = make_keys(1, fill=3.0)
    cache.update_and_fetch(keys, values)

    assert bool(mx.all(allocator.pool[0, 0, 0, :, 3, :] == 7.0)), "key not written to correct slot"
    assert bool(mx.all(allocator.pool[0, 0, 1, :, 3, :] == 3.0)), "value not written to correct slot"
    for slot in [0, 1, 2]:
        assert bool(mx.all(allocator.pool[0, 0, 0, :, slot, :] == 0.0)), f"slot {slot} was unexpectedly written"


def test_write_crosses_block_boundary():
    """5-token prefill: first 4 tokens go to block 0, 5th goes to block 1 slot 0."""
    allocator = make_allocator()
    cache = BatchPagedKVCache([[0, 1]], [0], 0, allocator, [5])

    keys   = make_keys_per_token(5)   # keys[0, :, t, :] = t + 1
    values = make_keys(5, fill=0.0)
    cache.update_and_fetch(keys, values)

    for t in range(4):
        assert bool(mx.all(allocator.pool[0, 0, 0, :, t, :] == float(t + 1))), \
            f"token {t} not in block 0 slot {t}"
    assert bool(mx.all(allocator.pool[1, 0, 0, :, 0, :] == 5.0)), "token 4 not in block 1 slot 0"
    for slot in range(1, 4):
        assert bool(mx.all(allocator.pool[1, 0, 0, :, slot, :] == 0.0)), \
            f"block 1 slot {slot} was unexpectedly written"


def test_write_two_sequences_no_cross_contamination():
    """Two sequences writing to separate blocks do not bleed into each other."""
    allocator = make_allocator()
    # Seq 0 → block 2, seq 1 → block 3
    cache = BatchPagedKVCache([[2], [3]], [0, 0], 0, allocator, [2, 2])

    keys_np = np.zeros((1, NUM_KV_HEADS, 4, HEAD_DIM), dtype=np.float16)
    keys_np[0, :, 0, :] = 1.0   # seq 0, local token 0
    keys_np[0, :, 1, :] = 2.0   # seq 0, local token 1
    keys_np[0, :, 2, :] = 9.0   # seq 1, local token 0
    keys_np[0, :, 3, :] = 8.0   # seq 1, local token 1
    cache.update_and_fetch(mx.array(keys_np), make_keys(4, fill=0.0))

    assert bool(mx.all(allocator.pool[2, 0, 0, :, 0, :] == 1.0)), "seq 0 token 0 not in block 2"
    assert bool(mx.all(allocator.pool[2, 0, 0, :, 1, :] == 2.0)), "seq 0 token 1 not in block 2"
    assert bool(mx.all(allocator.pool[3, 0, 0, :, 0, :] == 9.0)), "seq 1 token 0 not in block 3"
    assert bool(mx.all(allocator.pool[3, 0, 0, :, 1, :] == 8.0)), "seq 1 token 1 not in block 3"
    # Unwritten slots in each block should be zero — real contamination check
    for slot in range(2, BLOCK_SIZE):
        assert bool(mx.all(allocator.pool[2, 0, 0, :, slot, :] == 0.0)), f"block 2 slot {slot} unexpectedly non-zero"
        assert bool(mx.all(allocator.pool[3, 0, 0, :, slot, :] == 0.0)), f"block 3 slot {slot} unexpectedly non-zero"


def test_write_layer_isolation():
    """Writing to layer 0 does not affect layer 1."""
    allocator = make_allocator()
    cache = BatchPagedKVCache([[0]], [0], 0, allocator, [2])

    cache.update_and_fetch(make_keys(2, fill=99.0), make_keys(2, fill=99.0))

    assert bool(mx.all(allocator.pool[0, 0, 0, :, 0, :] == 99.0)), "layer 0 not written"
    assert bool(mx.all(allocator.pool[0, 1, :, :, :, :] == 0.0)), "layer 1 was incorrectly modified"


# ---------------------------------------------------------------------------
# Gather tests
# ---------------------------------------------------------------------------

def test_gather_partial_last_block():
    """3 tokens in a block of size 4: gathered tensor has 3 time steps, not 4."""
    allocator = make_allocator()
    cache = BatchPagedKVCache([[0]], [0], 0, allocator, [3])

    gathered_k, gathered_v = cache.update_and_fetch(make_keys(3, fill=1.0), make_keys(3, fill=2.0))

    assert gathered_k.shape == (1, NUM_KV_HEADS, 3, HEAD_DIM), \
        f"expected shape (1, {NUM_KV_HEADS}, 3, {HEAD_DIM}), got {gathered_k.shape}"
    assert gathered_v.shape == (1, NUM_KV_HEADS, 3, HEAD_DIM)
    assert bool(mx.all(allocator.pool[0, 0, 0, :, 3, :] == 0.0)), "slot 3 should be untouched"


def test_gather_two_sequences_concatenated_in_order():
    """Gather from two sequences: output is seq0 tokens followed by seq1 tokens."""
    allocator = make_allocator()
    # Seq 0: 2 tokens → block 0; Seq 1: 3 tokens → block 1
    cache = BatchPagedKVCache([[0], [1]], [0, 0], 0, allocator, [2, 3])

    keys_np = np.zeros((1, NUM_KV_HEADS, 5, HEAD_DIM), dtype=np.float16)
    keys_np[0, :, :2, :] = 10.0   # seq 0
    keys_np[0, :, 2:, :] = 20.0   # seq 1
    gathered_k, _ = cache.update_and_fetch(mx.array(keys_np), make_keys(5, fill=0.0))

    assert gathered_k.shape == (1, NUM_KV_HEADS, 5, HEAD_DIM)
    assert bool(mx.all(gathered_k[0, :, :2, :] == 10.0)), "seq 0 tokens not first in gather output"
    assert bool(mx.all(gathered_k[0, :, 2:, :] == 20.0)), "seq 1 tokens not second in gather output"


# ---------------------------------------------------------------------------
# End-to-end tests
# ---------------------------------------------------------------------------

def test_end_to_end_write_then_gather():
    """Values written in the same call are immediately returned by gather."""
    allocator = make_allocator()
    cache = BatchPagedKVCache([[0]], [0], 0, allocator, [4])

    keys   = make_keys_per_token(4)   # token t has value t + 1
    values = make_keys(4, fill=5.0)
    gathered_k, gathered_v = cache.update_and_fetch(keys, values)

    assert gathered_k.shape == (1, NUM_KV_HEADS, 4, HEAD_DIM)
    for t in range(4):
        assert bool(mx.all(gathered_k[0, :, t, :] == float(t + 1))), \
            f"token {t} value mismatch in gathered keys"
    assert bool(mx.all(gathered_v == 5.0)), "gathered values do not match written values"


def test_end_to_end_decode_appends_to_prefill_history():
    """Prefill writes 3 tokens; subsequent decode gather returns all 4 (prefill + decode)."""
    allocator = make_allocator()

    # Step 1: prefill
    prefill_cache = BatchPagedKVCache([[0]], [0], 0, allocator, [3])
    prefill_cache.update_and_fetch(make_keys(3, fill=1.0), make_keys(3, fill=1.0))

    # Step 2: decode — offset=3 places the new token in slot 3 of block 0
    decode_cache = BatchPagedKVCache([[0]], [3], 0, allocator, [1])
    gathered_k, gathered_v = decode_cache.update_and_fetch(make_keys(1, fill=2.0), make_keys(1, fill=2.0))

    assert gathered_k.shape == (1, NUM_KV_HEADS, 4, HEAD_DIM), \
        "decode gather should include all 4 KV tokens"
    assert bool(mx.all(gathered_k[0, :, :3, :] == 1.0)), "prefill tokens not present in decode gather"
    assert bool(mx.all(gathered_k[0, :, 3:, :] == 2.0)), "decode token not appended correctly"


def test_complex_multi_cycle_multi_layer():
    """
    Simulates a realistic inference scenario across all layers:

    Setup:
      Seq 0: 3-token prompt  → block 0  (3 prefill + 1 decode fills the block exactly)
      Seq 1: 2-token prompt  → block 1  (2 prefill + 2 decodes fills the block exactly)

    Token values are chosen to be distinct per sequence and per layer so
    any misrouting (wrong block, wrong layer, wrong slot) will produce
    a wrong value rather than accidentally matching.

      Layer 0 seq 0: prefill [1, 2, 3],   decode1 = 4
      Layer 0 seq 1: prefill [5, 6],       decode1 = 7,  decode2 = 8
      Layer 1 seq 0: prefill [11, 12, 13], decode1 = 14
      Layer 1 seq 1: prefill [15, 16],     decode1 = 17, decode2 = 18

    Steps:
      1. Prefill both sequences simultaneously (both layers)
      2. Decode step 1: both sequences (both layers) — verify full history gathered
      3. Decode step 2: seq 1 only (seq 0 finished) — verify seq 0's blocks are undisturbed
    """
    allocator = make_allocator()

    L0_S0 = [1.0, 2.0, 3.0, 4.0]    # layer 0, seq 0 token values
    L0_S1 = [5.0, 6.0, 7.0, 8.0]    # layer 0, seq 1 token values
    L1_S0 = [11.0, 12.0, 13.0, 14.0]
    L1_S1 = [15.0, 16.0, 17.0, 18.0]
    layer_vals = [(L0_S0, L0_S1), (L1_S0, L1_S1)]

    # --- Step 1: prefill both sequences ---
    # Packed order: [seq0_tok0, seq0_tok1, seq0_tok2, seq1_tok0, seq1_tok1]
    for layer in range(NUM_LAYERS):
        s0, s1 = layer_vals[layer]
        keys_np = np.zeros((1, NUM_KV_HEADS, 5, HEAD_DIM), dtype=np.float16)
        keys_np[0, :, 0, :] = s0[0]
        keys_np[0, :, 1, :] = s0[1]
        keys_np[0, :, 2, :] = s0[2]
        keys_np[0, :, 3, :] = s1[0]
        keys_np[0, :, 4, :] = s1[1]
        cache = BatchPagedKVCache([[0], [1]], [0, 0], layer, allocator, [3, 2])
        cache.update_and_fetch(mx.array(keys_np), make_keys(5, fill=0.0))

    # Verify pool contents for both layers
    for layer in range(NUM_LAYERS):
        s0, s1 = layer_vals[layer]
        for t in range(3):
            assert bool(mx.all(allocator.pool[0, layer, 0, :, t, :] == s0[t])), \
                f"prefill: layer {layer} seq 0 token {t}"
        for t in range(2):
            assert bool(mx.all(allocator.pool[1, layer, 0, :, t, :] == s1[t])), \
                f"prefill: layer {layer} seq 1 token {t}"

    # --- Step 2: decode step 1, both sequences running ---
    # Seq 0 gets decode token at offset 3 (slot 3 of block 0)
    # Seq 1 gets decode token at offset 2 (slot 2 of block 1)
    for layer in range(NUM_LAYERS):
        s0, s1 = layer_vals[layer]
        keys_np = np.zeros((1, NUM_KV_HEADS, 2, HEAD_DIM), dtype=np.float16)
        keys_np[0, :, 0, :] = s0[3]   # seq 0 decode token
        keys_np[0, :, 1, :] = s1[2]   # seq 1 decode token
        cache = BatchPagedKVCache([[0], [1]], [3, 2], layer, allocator, [1, 1])
        gathered_k, _ = cache.update_and_fetch(mx.array(keys_np), make_keys(2, fill=0.0))

        # Gathered output: seq 0's 4 tokens then seq 1's 3 tokens = 7 total
        assert gathered_k.shape == (1, NUM_KV_HEADS, 7, HEAD_DIM), \
            f"decode1: layer {layer} wrong shape {gathered_k.shape}"
        for t in range(4):
            assert bool(mx.all(gathered_k[0, :, t, :] == s0[t])), \
                f"decode1: layer {layer} seq 0 token {t} mismatch"
        for t in range(3):
            assert bool(mx.all(gathered_k[0, :, 4 + t, :] == s1[t])), \
                f"decode1: layer {layer} seq 1 token {t} mismatch"

    # --- Step 3: decode step 2, seq 1 only (seq 0 finished) ---
    for layer in range(NUM_LAYERS):
        s0, s1 = layer_vals[layer]
        keys_np = np.full((1, NUM_KV_HEADS, 1, HEAD_DIM), s1[3], dtype=np.float16)
        cache = BatchPagedKVCache([[1]], [3], layer, allocator, [1])
        gathered_k, _ = cache.update_and_fetch(mx.array(keys_np), make_keys(1, fill=0.0))

        # Seq 1 now has 4 KV tokens
        assert gathered_k.shape == (1, NUM_KV_HEADS, 4, HEAD_DIM), \
            f"decode2: layer {layer} wrong shape {gathered_k.shape}"
        for t in range(4):
            assert bool(mx.all(gathered_k[0, :, t, :] == s1[t])), \
                f"decode2: layer {layer} seq 1 token {t} mismatch"

        # Seq 0's block (block 0) must be completely undisturbed
        for t in range(4):
            assert bool(mx.all(allocator.pool[0, layer, 0, :, t, :] == s0[t])), \
                f"decode2: layer {layer} seq 0 block was corrupted at slot {t}"


def test_integrated_engine_loop():
    """
    Simulates the full engine loop — scheduler, KV cache, and sampler working together —
    without a real model forward pass.

    Two sequences run concurrently through prefill then decode until EOS:
      Seq 0: prompt=[1, 2]    → generates [3, 4, EOS]  (3 sampled tokens)
      Seq 1: prompt=[6, 7, 8] → generates [5, EOS]     (2 sampled tokens)

    The fake model sets K/V = token_id for each token, and produces logits
    with a spike at the predetermined next token so greedy sampling is deterministic.

    Verifies:
      - Gathered KV shape grows correctly at every step across all layers
      - Sampler selects the correct token at every step
      - KV pool contains correct token values at the right block locations (spot-check)
      - Final output_token_ids match the predetermined sequences
      - All blocks are freed once both sequences finish
    """
    EOS        = 9
    VOCAB_SIZE = 10

    model_cfg = ModelConfig(
        num_layers=NUM_LAYERS, num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM, eos_token_id=EOS,
    )
    server_cfg = ServerConfig(
        model_path="", block_size=BLOCK_SIZE,
        max_num_blocks=10, max_num_seqs=4, max_seq_len=20,
    )

    scheduler = Scheduler(model_cfg, server_cfg)
    allocator = scheduler.allocator   # shared reference — same pool the cache writes into

    seq0 = Sequence(seq_id=0, prompt_token_ids=[1, 2],    sampling_params=SamplingParams(max_tokens=10))
    seq1 = Sequence(seq_id=1, prompt_token_ids=[6, 7, 8], sampling_params=SamplingParams(max_tokens=10))
    scheduler.add_sequence(seq0)
    scheduler.add_sequence(seq1)

    assert allocator.num_free_blocks == 10, "all blocks should be free before admission"

    token_schedule = {0: [3, 4, EOS], 1: [5, EOS]}
    step_counter   = {0: 0, 1: 0}
    all_completed  = []

    for iteration in range(10):
        schedule = scheduler.step()
        if not schedule.prefill_sequences and not schedule.decode_sequences:
            break

        all_seqs = schedule.prefill_sequences + schedule.decode_sequences
        batch    = build_batch(schedule)

        # kv_offsets: 0 for prefill (nothing cached yet),
        #             num_kv_tokens - 1 for decode (all prior tokens already cached)
        kv_offsets = (
            [0] * len(schedule.prefill_sequences) +
            [seq.num_kv_tokens - 1 for seq in schedule.decode_sequences]
        )

        # Fake model: K/V value for each token = that token's id
        keys_np = np.zeros((1, NUM_KV_HEADS, len(batch.token_ids), HEAD_DIM), dtype=np.float16)
        for flat_t, tid in enumerate(batch.token_ids):
            keys_np[0, :, flat_t, :] = float(tid)
        keys = mx.array(keys_np)

        # Run KV cache for every layer; verify gathered shape at each
        expected_total_kv = sum(off + sl for off, sl in zip(kv_offsets, batch.seq_lens))
        for layer in range(NUM_LAYERS):
            cache = BatchPagedKVCache(batch.block_tables, kv_offsets, layer, allocator, batch.seq_lens)
            gathered_k, _ = cache.update_and_fetch(keys, keys)
            assert gathered_k.shape == (1, NUM_KV_HEADS, expected_total_kv, HEAD_DIM), \
                f"iter {iteration} layer {layer}: shape {gathered_k.shape}"

        # Spot-check pool values on the prefill iteration
        if iteration == 0:
            bt0 = batch.block_tables[0]
            bt1 = batch.block_tables[1]
            for layer in range(NUM_LAYERS):
                for slot, tid in enumerate([1.0, 2.0]):
                    assert bool(mx.all(allocator.pool[bt0[0], layer, 0, :, slot, :] == tid)), \
                        f"prefill layer {layer}: seq 0 token {slot} (id={tid}) missing from pool"
                for slot, tid in enumerate([6.0, 7.0, 8.0]):
                    assert bool(mx.all(allocator.pool[bt1[0], layer, 0, :, slot, :] == tid)), \
                        f"prefill layer {layer}: seq 1 token {slot} (id={tid}) missing from pool"

        # Fake logits: spike at the predetermined next token for each sequence
        desired   = {seq.seq_id: token_schedule[seq.seq_id][step_counter[seq.seq_id]] for seq in all_seqs}
        logits_np = np.zeros((len(all_seqs), VOCAB_SIZE), dtype=np.float32)
        for i, seq in enumerate(all_seqs):
            logits_np[i, desired[seq.seq_id]] = 100.0
        for seq in all_seqs:
            step_counter[seq.seq_id] += 1

        next_tokens = sampler.sample(mx.array(logits_np), all_seqs)

        for seq in all_seqs:
            assert next_tokens[seq.seq_id] == desired[seq.seq_id], \
                f"iter {iteration} seq {seq.seq_id}: sampler chose {next_tokens[seq.seq_id]}, expected {desired[seq.seq_id]}"

        completed = scheduler.update(next_tokens)
        all_completed.extend(completed)

    assert len(all_completed) == 2, f"expected 2 completed sequences, got {len(all_completed)}"

    result = {s.seq_id: s for s in all_completed}
    assert result[0].output_token_ids == [3, 4, EOS], f"seq 0: {result[0].output_token_ids}"
    assert result[1].output_token_ids == [5, EOS],    f"seq 1: {result[1].output_token_ids}"
    assert allocator.num_free_blocks == 10, \
        f"expected all 10 blocks free after completion, got {allocator.num_free_blocks}"


if __name__ == "__main__":
    tests = [
        test_write_single_decode_token,
        test_write_crosses_block_boundary,
        test_write_two_sequences_no_cross_contamination,
        test_write_layer_isolation,
        test_gather_partial_last_block,
        test_gather_two_sequences_concatenated_in_order,
        test_end_to_end_write_then_gather,
        test_end_to_end_decode_appends_to_prefill_history,
        test_complex_multi_cycle_multi_layer,
        test_integrated_engine_loop,
    ]
    for test in tests:
        try:
            test()
            print(f"PASS  {test.__name__}")
        except AssertionError as e:
            print(f"FAIL  {test.__name__}: {e}")
        except Exception as e:
            print(f"ERROR {test.__name__}: {type(e).__name__}: {e}")
