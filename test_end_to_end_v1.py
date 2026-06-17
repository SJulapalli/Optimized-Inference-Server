import unittest
import numpy as np
import mlx.core as mx
from transformers import AutoTokenizer

from config import ModelConfig, ServerConfig
from engine.sequence import Sequence, SequenceStatus, SamplingParams
from engine.block_allocator import BlockAllocator
from engine.runner import (
    build_batch, BatchPagedKVCache, ModelRunner,
    compute_kv_offsets, build_block_diagonal_mask
)
from model.llama import Model
from model.weights import load_config, load_weights
from scheduler import Scheduler, SchedulerOutput
from sampling.sampler import sample

MODEL_PATH = "/Users/Suhas/Llama-3.2-3B"
SEED = 42

LLAMA32_3B = ModelConfig(
    num_layers=28,
    d_model=3072,
    num_q_heads=24,
    num_kv_heads=8,
    head_dim=128,
    ffn_hidden=8192,
    vocab_size=128256,
    max_seq_len=131072,
    eos_token_id=128009,
)


# ── Mask ──────────────────────────────────────────────────────────────────────

class TestBlockDiagonalMask(unittest.TestCase):

    def test_shape(self):
        mask = build_block_diagonal_mask([3, 5, 1, 1], [0, 0, 7, 2], num_prefill_seqs=2)
        self.assertEqual(mask.shape, (1, 1, 10, 19))

    def test_cross_sequence_blocked(self):
        mask = np.array(build_block_diagonal_mask([3, 3], [0, 0], num_prefill_seqs=2))
        self.assertTrue(np.all(np.isinf(mask[0, 0, :3, 3:])))
        self.assertTrue(np.all(np.isinf(mask[0, 0, 3:, :3])))

    def test_prefill_block_is_causal(self):
        mask = np.array(build_block_diagonal_mask([4], [0], num_prefill_seqs=1))
        for i in range(4):
            for j in range(i + 1):
                self.assertEqual(mask[0, 0, i, j], 0.0)
            for j in range(i + 1, 4):
                self.assertTrue(np.isinf(mask[0, 0, i, j]))

    def test_decode_attends_full_history(self):
        mask = np.array(build_block_diagonal_mask([1], [10], num_prefill_seqs=0))
        self.assertEqual(mask.shape, (1, 1, 1, 11))
        self.assertTrue(np.all(mask[0, 0, 0, :] == 0.0))


# ── KV Offsets ────────────────────────────────────────────────────────────────

class TestKVOffsets(unittest.TestCase):

    def test_prefill_is_zero(self):
        seq = Sequence(seq_id=0, prompt_token_ids=[1, 2, 3, 4, 5], sampling_params=SamplingParams())
        seq.status = SequenceStatus.PREFILL
        self.assertEqual(compute_kv_offsets([seq])[0], 0)

    def test_decode_is_num_kv_tokens_minus_one(self):
        seq = Sequence(seq_id=1, prompt_token_ids=[1, 2, 3], sampling_params=SamplingParams())
        seq.output_token_ids = [10, 11, 12]
        seq.status = SequenceStatus.DECODE
        self.assertEqual(compute_kv_offsets([seq])[0], 5)


# ── Batch construction ────────────────────────────────────────────────────────

class TestBuildBatch(unittest.TestCase):

    def test_prefill_batch(self):
        seq = Sequence(seq_id=0, prompt_token_ids=[10, 20, 30], sampling_params=SamplingParams())
        seq.status = SequenceStatus.PREFILL
        seq.block_table = [0]
        batch = build_batch(SchedulerOutput(prefill_sequences=[seq], decode_sequences=[]))
        self.assertEqual(batch.token_ids, [10, 20, 30])
        self.assertEqual(batch.positions, [0, 1, 2])
        self.assertEqual(batch.seq_lens, [3])
        self.assertEqual(batch.num_prefill_seqs, 1)
        self.assertEqual(batch.kv_cache_offsets, [0])

    def test_decode_batch(self):
        seq = Sequence(seq_id=1, prompt_token_ids=[10, 20, 30], sampling_params=SamplingParams())
        seq.output_token_ids = [40, 50]
        seq.status = SequenceStatus.DECODE
        seq.block_table = [0]
        batch = build_batch(SchedulerOutput(prefill_sequences=[], decode_sequences=[seq]))
        self.assertEqual(batch.token_ids, [50])
        self.assertEqual(batch.positions, [4])
        self.assertEqual(batch.seq_lens, [1])
        self.assertEqual(batch.num_prefill_seqs, 0)
        self.assertEqual(batch.kv_cache_offsets, [4])


# ── Sampler ───────────────────────────────────────────────────────────────────

class TestSampler(unittest.TestCase):

    def test_greedy_picks_argmax(self):
        logits = mx.array([[0.1, 0.2, 5.0, 0.1]])
        seq = Sequence(seq_id=0, prompt_token_ids=[1], sampling_params=SamplingParams())
        result = sample(logits, [seq])
        self.assertEqual(result[0], 2)


# ── End-to-end ────────────────────────────────────────────────────────────────

class TestEndToEnd(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            cls.model_args = load_config(MODEL_PATH)
            cls.model = Model(cls.model_args)
            load_weights(MODEL_PATH, cls.model)
            cls.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
            cls.server_config = ServerConfig(model_path=MODEL_PATH, max_num_seqs=4)
            cls.model_loaded = True
        except Exception as e:
            cls.model_loaded = False
            cls.load_error = str(e)

    def setUp(self):
        if not self.model_loaded:
            self.skipTest(f"Model not available: {self.load_error}")
        mx.random.seed(SEED)
        self.allocator = BlockAllocator(LLAMA32_3B, self.server_config)
        self.scheduler = Scheduler(LLAMA32_3B, self.server_config, block_allocator=self.allocator)
        self.runner = ModelRunner(self.model, self.allocator)

    def _run_loop(self, max_steps=60, verbose=False):
        completed = []
        for step in range(max_steps):
            schedule = self.scheduler.step()
            if not schedule.prefill_sequences and not schedule.decode_sequences:
                break
            sequences = schedule.prefill_sequences + schedule.decode_sequences
            batch = build_batch(schedule)
            logits = self.runner.forward(batch)
            next_tokens = sample(logits, sequences)
            if verbose:
                for seq in sequences:
                    tok = next_tokens[seq.seq_id]
                    print(f"  [seq{seq.seq_id}] {self.tokenizer.decode([tok])!r}", flush=True)
            completed += self.scheduler.update(next_tokens)
        return completed


    def test_forward_output_shape(self):
        # Verifies the runner returns [num_seqs, vocab_size] after a prefill step
        ids = self.tokenizer.encode("Hello")
        self.scheduler.add_sequence(Sequence(seq_id=0, prompt_token_ids=ids, sampling_params=SamplingParams(max_tokens=1)))
        schedule = self.scheduler.step()
        logits = self.runner.forward(build_batch(schedule))
        self.assertEqual(logits.shape, (1, self.model_args.vocab_size))

    def test_single_sequence(self):
        ids = self.tokenizer.encode("The capital of France is")
        self.scheduler.add_sequence(Sequence(seq_id=0, prompt_token_ids=ids, sampling_params=SamplingParams(max_tokens=20)))
        completed = self._run_loop()
        self.assertEqual(len(completed), 1)
        output = self.tokenizer.decode(completed[0].output_token_ids)
        print(f"\n[single] '{output}'")

    def test_concurrent_sequences(self):
        prompts = ["The sky is", "Water is made of", "The largest planet is"]
        for i, p in enumerate(prompts):
            ids = self.tokenizer.encode(p)
            self.scheduler.add_sequence(Sequence(seq_id=i, prompt_token_ids=ids, sampling_params=SamplingParams(max_tokens=15)))
        completed = self._run_loop(max_steps=100)
        self.assertEqual(len(completed), len(prompts))
        for seq in completed:
            self.assertGreater(len(seq.output_token_ids), 0)
            print(f"\n[concurrent {seq.seq_id}] '{self.tokenizer.decode(seq.output_token_ids)}'")

    def test_max_tokens_respected(self):
        ids = self.tokenizer.encode("Count to ten:")
        self.scheduler.add_sequence(Sequence(seq_id=0, prompt_token_ids=ids, sampling_params=SamplingParams(max_tokens=5)))
        completed = self._run_loop()
        self.assertLessEqual(len(completed[0].output_token_ids), 5)

    def test_sequence_replacement_on_completion(self):
        """
        Fills scheduler to capacity with 3 sequences, one of which (seq 0) is
        deliberately short. Verifies seq 0 completes first, that its freed slot
        and blocks allow a queued replacement to be admitted in the subsequent
        scheduler step, and that all 4 sequences ultimately complete correctly.
        """
        local_config = ServerConfig(model_path=MODEL_PATH, max_num_seqs=3)
        local_allocator = BlockAllocator(LLAMA32_3B, local_config)
        local_scheduler = Scheduler(LLAMA32_3B, local_config, block_allocator=local_allocator)
        local_runner = ModelRunner(self.model, local_allocator)

        # Seq 0 finishes in exactly 3 tokens; 1 and 2 run much longer
        for seq_id, (prompt, max_tok) in enumerate([
            ("Hi",                        3),
            ("The capital of France is", 25),
            ("Water is made of",         25),
        ]):
            local_scheduler.add_sequence(Sequence(
                seq_id=seq_id,
                prompt_token_ids=self.tokenizer.encode(prompt),
                sampling_params=SamplingParams(max_tokens=max_tok),
            ))

        # Replacement waits in the queue until seq 0's slot frees
        replacement = Sequence(
            seq_id=3,
            prompt_token_ids=self.tokenizer.encode("The largest planet is"),
            sampling_params=SamplingParams(max_tokens=15),
        )

        all_completed      = []
        completion_order   = []
        replacement_added  = False
        first_completion_step     = None
        replacement_admitted_step = None
        free_blocks_after_completion = None

        for step in range(200):
            schedule = local_scheduler.step()
            active_ids = {s.seq_id for s in schedule.prefill_sequences + schedule.decode_sequences}

            # Record the first step the replacement appears in an active batch
            if replacement_added and replacement_admitted_step is None and 3 in active_ids:
                replacement_admitted_step = step

            if not schedule.prefill_sequences and not schedule.decode_sequences:
                break

            sequences = schedule.prefill_sequences + schedule.decode_sequences
            batch     = build_batch(schedule)
            logits    = local_runner.forward(batch)
            next_tokens = sample(logits, sequences)
            for seq in sequences:
                tok = next_tokens[seq.seq_id]
                print(f"  step {step:3d} [seq{seq.seq_id}] {self.tokenizer.decode([tok])!r}", flush=True)
            newly_completed = local_scheduler.update(next_tokens)

            if newly_completed:
                all_completed.extend(newly_completed)
                completion_order.extend(s.seq_id for s in newly_completed)

            # On the step seq 0 first completes: snapshot free blocks, enqueue replacement
            if not replacement_added and any(s.seq_id == 0 for s in newly_completed):
                first_completion_step        = step
                free_blocks_after_completion = local_allocator.num_free_blocks
                local_scheduler.add_sequence(replacement)
                replacement_added = True

            if len(all_completed) == 4:
                break

        # All 4 sequences ran to completion
        self.assertEqual(len(all_completed), 4, "Not all sequences completed")

        # Seq 0 (max_tokens=3) must have been the first to finish
        self.assertEqual(completion_order[0], 0, "Seq 0 did not complete first")

        # Blocks were actually freed when seq 0 completed
        self.assertIsNotNone(free_blocks_after_completion)
        self.assertGreater(free_blocks_after_completion, 0,
                        "No blocks freed after seq 0 completed")

        # Replacement was admitted, and only *after* seq 0 completed (not before)
        self.assertIsNotNone(replacement_admitted_step,
                            "Replacement sequence was never admitted to active scheduling")
        self.assertGreater(replacement_admitted_step, first_completion_step,
                        "Replacement was admitted before seq 0 freed its slot")

        # Every sequence produced at least one output token
        for seq in all_completed:
            self.assertGreater(len(seq.output_token_ids), 0,
                            f"Seq {seq.seq_id} produced no output tokens")

        # Replacement respects its own max_tokens bound
        replacement_result = next(s for s in all_completed if s.seq_id == 3)
        self.assertLessEqual(len(replacement_result.output_token_ids), 15)

        for seq in all_completed:
            print(f"\n[replacement_test seq{seq.seq_id}] "
                f"'{self.tokenizer.decode(seq.output_token_ids)}'")


if __name__ == "__main__":
    unittest.main(verbosity=2)
