"""
Standalone scheduler tests. No model or forward pass required.
Simulates the engine loop by calling step() and update() with fake tokens.

Uses small configs to make block boundaries easy to reason about:
  block_size = 4   → each block holds 4 token positions
  max_num_blocks = 10  → total KV capacity of 40 token positions
  max_num_seqs = 3
  max_seq_len = 20
  eos_token_id = 999  → easy-to-spot sentinel, not a real token
"""

from config import ModelConfig, ServerConfig
from engine.sequence import Sequence, SequenceStatus, SamplingParams
from scheduler import Scheduler, SchedulerOutput

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_CFG = ModelConfig(eos_token_id=999)

SERVER_CFG = ServerConfig(
    model_path="fake/path",
    block_size=4,
    max_num_blocks=10,
    max_num_seqs=3,
    max_seq_len=20,
)

EOS = MODEL_CFG.eos_token_id
NORMAL_TOKEN = 1

# ── Helpers ───────────────────────────────────────────────────────────────────

_seq_counter = 0

def make_sequence(prompt_len: int, max_tokens: int = 10) -> Sequence:
    """Create a sequence with a fake prompt of the given length."""
    global _seq_counter
    _seq_counter += 1
    return Sequence(
        seq_id=_seq_counter,
        prompt_token_ids=list(range(prompt_len)),
        sampling_params=SamplingParams(max_tokens=max_tokens),
    )

def make_scheduler() -> Scheduler:
    """Fresh scheduler with small test config."""
    return Scheduler(MODEL_CFG, SERVER_CFG)

def fake_next_tokens(scheduler: Scheduler, token_id: int) -> dict[int, int]:
    """Build a next_tokens dict giving every active sequence the same token."""
    return {seq.seq_id: token_id for seq in scheduler.active_sequences}

def run(name: str, fn):
    try:
        fn()
        print(f"  PASS  {name}")
    except AssertionError as e:
        print(f"  FAIL  {name}: {e}")
    except Exception as e:
        print(f"  ERROR {name}: {type(e).__name__}: {e}")

# ── Tests ─────────────────────────────────────────────────────────────────────

def test_add_sequence_valid():
    """A short prompt lands in the waiting queue."""
    s = make_scheduler()
    seq = make_sequence(prompt_len=5)
    s.add_sequence(seq)
    assert len(s.waiting_sequences) == 1
    assert s.waiting_sequences[0].seq_id == seq.seq_id

def test_add_sequence_too_long():
    """A prompt exceeding max_seq_len raises ValueError before touching the queue."""
    s = make_scheduler()
    seq = make_sequence(prompt_len=SERVER_CFG.max_seq_len + 1)
    raised = False
    try:
        s.add_sequence(seq)
    except ValueError:
        raised = True
    assert raised, "Expected ValueError for oversized prompt"
    assert len(s.waiting_sequences) == 0

def test_step_admits_waiting_sequence():
    """step() moves a waiting sequence into active with PREFILL status."""
    s = make_scheduler()
    seq = make_sequence(prompt_len=5)
    s.add_sequence(seq)

    out = s.step()

    assert len(s.waiting_sequences) == 0
    assert len(s.active_sequences) == 1
    assert s.active_sequences[0].status == SequenceStatus.PREFILL
    assert seq in out.prefill_sequences

def test_step_allocates_correct_blocks():
    """
    A prompt of 5 tokens with block_size=4 requires ceil(5/4)=2 blocks.
    The sequence's block_table should have 2 entries after step().
    """
    s = make_scheduler()
    seq = make_sequence(prompt_len=5)
    s.add_sequence(seq)
    s.step()

    assert len(seq.block_table) == 2
    # KV capacity for this sequence is now 2*4=8 token positions
    assert s.allocator.num_free_blocks == SERVER_CFG.max_num_blocks - 2

def test_step_respects_max_num_seqs():
    """step() admits at most max_num_seqs sequences (3 in test config)."""
    s = make_scheduler()
    for _ in range(5):
        s.add_sequence(make_sequence(prompt_len=2))

    s.step()

    assert len(s.active_sequences) == SERVER_CFG.max_num_seqs
    assert len(s.waiting_sequences) == 5 - SERVER_CFG.max_num_seqs

def test_step_skips_large_but_admits_small():
    """
    If a large sequence at the front of the queue doesn't fit, smaller
    sequences behind it should still be admitted (throughput over fairness).

    Setup: fill 9 of 10 blocks, leaving 1 free.
      - seq A needs 2 blocks (prompt=5) → won't fit
      - seq B needs 1 block  (prompt=3) → should fit
    """
    s = make_scheduler()

    # Occupy 9 blocks by admitting a sequence that needs 9 blocks (prompt=35
    # tokens → ceil(35/4)=9 blocks). We manually bypass add_sequence's
    # max_seq_len check since this is just setup, so we directly push to waiting
    # after temporarily raising the seq_len cap.
    big_setup = make_sequence(prompt_len=9 * SERVER_CFG.block_size)  # exactly 9 blocks
    big_setup.prompt_token_ids = big_setup.prompt_token_ids[:9 * SERVER_CFG.block_size]
    s.waiting_sequences.append(big_setup)
    s.step()  # admits big_setup, 1 block remaining

    seq_a = make_sequence(prompt_len=5)   # needs 2 blocks
    seq_b = make_sequence(prompt_len=3)   # needs 1 block
    s.add_sequence(seq_a)
    s.add_sequence(seq_b)

    # Reset active cap so we can admit more (big_setup counts against max_num_seqs)
    # Use a fresh scheduler with big_setup already active to keep max_num_seqs clean.
    s2 = make_scheduler()
    s2.allocator.free_blocks = s2.allocator.free_blocks[1:]  # simulate 1 free block
    s2.active_sequences.append(big_setup)  # already "running"
    s2.add_sequence(seq_a)
    s2.add_sequence(seq_b)
    s2.step()

    admitted_ids = {seq.seq_id for seq in s2.active_sequences}
    assert seq_b.seq_id in admitted_ids, "seq_b (1 block) should have been admitted"
    assert seq_a.seq_id not in admitted_ids, "seq_a (2 blocks) should not fit"

def test_update_prefill_to_decode():
    """After update() with a normal token, PREFILL sequences become DECODE."""
    s = make_scheduler()
    seq = make_sequence(prompt_len=3)
    s.add_sequence(seq)
    s.step()

    assert seq.status == SequenceStatus.PREFILL
    completed = s.update({seq.seq_id: NORMAL_TOKEN})

    assert seq.status == SequenceStatus.DECODE
    assert seq.output_token_ids == [NORMAL_TOKEN]
    assert completed == []

def test_update_eos_finishes_sequence():
    """EOS token moves the sequence to FINISHED, frees its blocks, returns it."""
    s = make_scheduler()
    seq = make_sequence(prompt_len=3)
    s.add_sequence(seq)
    s.step()

    blocks_before = s.allocator.num_free_blocks
    completed = s.update({seq.seq_id: EOS})

    assert seq.seq_id not in {sq.seq_id for sq in s.active_sequences}
    assert seq.status == SequenceStatus.FINISHED
    assert seq in completed
    # Blocks should be returned to the allocator
    assert s.allocator.num_free_blocks > blocks_before

def test_update_max_tokens_finishes_sequence():
    """A sequence with max_tokens=2 finishes after 2 generated tokens."""
    s = make_scheduler()
    seq = make_sequence(prompt_len=3, max_tokens=2)
    s.add_sequence(seq)
    s.step()

    # First token — transitions to DECODE, not finished yet
    s.update({seq.seq_id: NORMAL_TOKEN})
    assert seq.status == SequenceStatus.DECODE

    # Second token — hits max_tokens, should finish
    completed = s.update({seq.seq_id: NORMAL_TOKEN})
    assert seq in completed
    assert seq.status == SequenceStatus.FINISHED

def test_update_allocates_block_at_boundary():
    """
    With block_size=4 and a prompt of 4 tokens (fills 1 block exactly),
    the first decode token pushes num_kv_tokens to 5 > capacity 4.
    A second block must be allocated automatically in update().
    """
    s = make_scheduler()
    seq = make_sequence(prompt_len=4)   # ceil(4/4) = 1 block
    s.add_sequence(seq)
    s.step()

    assert len(seq.block_table) == 1
    assert seq.num_kv_tokens == 4

    # update() appends token → num_kv_tokens becomes 5, capacity was 4
    s.update({seq.seq_id: NORMAL_TOKEN})

    assert len(seq.block_table) == 2, "A new block should have been allocated at the boundary"

def test_update_preempts_when_oom():
    """
    When a decode sequence needs a new block but the pool is full,
    it gets preempted: removed from active, re-queued at the front of waiting,
    blocks freed.
    """
    s = make_scheduler()

    # Fill all but 1 block with a background sequence (prompt = 9 blocks)
    filler = make_sequence(prompt_len=(SERVER_CFG.max_num_blocks - 1) * SERVER_CFG.block_size)
    s.waiting_sequences.append(filler)
    s.step()
    # 1 block remains free

    # Add a sequence that uses the last free block (prompt = 1 block = 4 tokens)
    victim = make_sequence(prompt_len=4)
    s.add_sequence(victim)
    s.step()
    # Pool now completely full

    assert s.allocator.num_free_blocks == 0

    # First update: victim transitions PREFILL → DECODE. num_kv_tokens = 5,
    # capacity = 4 → needs a new block → OOM → preempted.
    s.update({filler.seq_id: NORMAL_TOKEN, victim.seq_id: NORMAL_TOKEN})

    assert victim.seq_id not in {sq.seq_id for sq in s.active_sequences}
    assert victim in s.waiting_sequences
    assert victim.status == SequenceStatus.WAITING
    assert victim.block_table == []
    # Freeing victim's block should give the pool back 1 free block
    assert s.allocator.num_free_blocks == 1

# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running scheduler tests...\n")
    tests = [
        ("add_sequence: valid prompt queued",           test_add_sequence_valid),
        ("add_sequence: oversized prompt raises",       test_add_sequence_too_long),
        ("step: admits waiting sequence",               test_step_admits_waiting_sequence),
        ("step: allocates correct block count",         test_step_allocates_correct_blocks),
        ("step: respects max_num_seqs cap",             test_step_respects_max_num_seqs),
        ("step: skips large, admits smaller",           test_step_skips_large_but_admits_small),
        ("update: PREFILL → DECODE on normal token",   test_update_prefill_to_decode),
        ("update: EOS finishes and frees sequence",     test_update_eos_finishes_sequence),
        ("update: max_tokens limit finishes sequence",  test_update_max_tokens_finishes_sequence),
        ("update: new block allocated at boundary",     test_update_allocates_block_at_boundary),
        ("update: OOM preempts decode sequence",        test_update_preempts_when_oom),
    ]
    for name, fn in tests:
        run(name, fn)
