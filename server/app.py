import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from config import ModelConfig, ServerConfig
from engine.block_allocator import BlockAllocator
from engine.runner import ModelRunner, build_batch, Batch
from engine.sequence import Sequence, SamplingParams
from model.llama import Model, ModelArgs
from sampling.sampler import sample
from scheduler import Scheduler
from server.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    ModelCard,
    ModelList,
)

import traceback
from fastapi.staticfiles import StaticFiles
import os
import mlx.core as mx


def _model_config_from_args(args: ModelArgs, tokenizer) -> ModelConfig:
    return ModelConfig(
        num_layers=args.num_hidden_layers,
        d_model=args.hidden_size,
        num_q_heads=args.num_attention_heads,
        num_kv_heads=args.num_key_value_heads,
        head_dim=args.head_dim or args.hidden_size // args.num_attention_heads,
        ffn_hidden=args.intermediate_size,
        vocab_size=args.vocab_size,
        max_seq_len=args.max_position_embeddings or 131072,
        rms_norm_eps=args.rms_norm_eps,
        eos_token_id=tokenizer.eos_token_id,
    )

class InferenceEngine:
    def __init__(
        self,
        server_config: ServerConfig,
        model_config: ModelConfig,
        model: Model,
        tokenizer,
    ):
        self.server_config = server_config
        self.model_config = model_config
        self.tokenizer = tokenizer
        self.model_name = server_config.model_path.rstrip("/").split("/")[-1]

        self.allocator = BlockAllocator(model_config, server_config)
        self.scheduler = Scheduler(model_config, server_config, self.allocator)
        self.runner = ModelRunner(model, self.allocator)

        self.output_queues: dict[int, asyncio.Queue] = {}
        self._seq_counter = 0
        self._loop_task: asyncio.Task | None = None

        # ── temporary TTFT-breakdown scaffolding (remove once diagnosed) ──
        self._ttft_arrival: dict[int, float] = {}    # add_request time
        self._ttft_scheduled: dict[int, float] = {}  # first time seq is in a batch
        self._ttft_done: set[int] = set()

    def _next_seq_id(self) -> int:
        seq_id = self._seq_counter
        self._seq_counter += 1
        return seq_id

    def warmup(self):
        """Run a dummy prefill + decode so MLX compiles its Metal kernels at
        startup instead of on the first real request."""
        prompt_len = 4
        bt = self.allocator.allocate(1)
        try:
            prefill = Batch(
                token_ids=list(range(prompt_len)),
                positions=list(range(prompt_len)),
                block_tables=[bt],
                seq_lens=[prompt_len],
                num_prefill_seqs=1,
                kv_cache_offsets=[0],
            )
            mx.eval(self.runner.forward(prefill))

            decode = Batch(
                token_ids=[0],
                positions=[prompt_len],
                block_tables=[bt],
                seq_lens=[1],
                num_prefill_seqs=0,
                kv_cache_offsets=[prompt_len],
            )
            mx.eval(self.runner.forward(decode))
        finally:
            self.allocator.free(bt)

    async def start(self):
        self._loop_task = asyncio.create_task(self._engine_loop())

    async def stop(self):
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass

    async def _engine_loop(self):

        loop = asyncio.get_event_loop()
        # ── temporary profiling scaffolding (remove once diagnosed) ──────────
        prof = {"schedule": 0.0, "build": 0.0, "forward": 0.0,
                "sample": 0.0, "update": 0.0, "steps": 0}
        while True:
            try:
                t0 = time.perf_counter()
                schedule = self.scheduler.step()
                if not schedule.prefill_sequences and not schedule.decode_sequences:
                    await asyncio.sleep(0.001)
                    continue
                t1 = time.perf_counter()

                sequences = schedule.prefill_sequences + schedule.decode_sequences
                batch = build_batch(schedule, self.server_config.prefill_chunk_size)
                t2 = time.perf_counter()

                # ttft: stamp the first step a sequence appears in a batch
                for seq in sequences:
                    if seq.seq_id not in self._ttft_scheduled:
                        self._ttft_scheduled[seq.seq_id] = t1

                def _compute(batch, sequences):
                    logits = self.runner.forward(batch)
                    mx.eval(logits)  # force the lazy graph so forward time isn't smeared into sample
                    return sample(logits, sequences)
                t3 = time.perf_counter()

                next_tokens = await loop.run_in_executor(None, _compute, batch, sequences)
                t4 = time.perf_counter()

                completed = self.scheduler.update(next_tokens)
                t5 = time.perf_counter()

                prof["schedule"] += t1 - t0
                prof["build"] += t2 - t1
                prof["forward"] += t3 - t2
                prof["sample"] += t4 - t3
                prof["update"] += t5 - t4
                prof["steps"] += 1
                if prof["steps"] % 25 == 0:
                    n = prof["steps"]
                    tot = sum(prof[k] for k in ("schedule", "build", "forward", "sample", "update"))
                    print(
                        f"[prof] step={n} avg_step={tot / n * 1000:6.1f}ms | "
                        f"sched={prof['schedule'] / n * 1000:5.1f} "
                        f"build={prof['build'] / n * 1000:5.1f} "
                        f"fwd={prof['forward'] / n * 1000:6.1f} "
                        f"samp={prof['sample'] / n * 1000:5.1f} "
                        f"upd={prof['update'] / n * 1000:5.1f} || "
                        f"this_step={(t5 - t0) * 1000:6.1f}ms "
                        f"toks={len(batch.token_ids)} "
                        f"nprefill={batch.num_prefill_seqs} "
                        f"ndecode={len(schedule.decode_sequences)} "
                        f"preempt={self.scheduler.num_preemptions} "
                        f"free_blocks={self.allocator.num_free_blocks}",
                        flush=True,
                    )
                # ttft: first token produced for a sequence (prefill complete)
                for seq in sequences:
                    sid = seq.seq_id
                    if sid not in self._ttft_done and len(seq.output_token_ids) >= 1:
                        self._ttft_done.add(sid)
                        arr = self._ttft_arrival.get(sid)
                        sch = self._ttft_scheduled.get(sid, t5)
                        if arr is not None:
                            print(
                                f"[ttft] seq={sid:<3} queue={(sch - arr) * 1000:7.1f}ms "
                                f"prefill={(t5 - sch) * 1000:7.1f}ms "
                                f"server_total={(t5 - arr) * 1000:8.1f}ms || "
                                f"step nprefill={batch.num_prefill_seqs} "
                                f"ndecode={len(schedule.decode_sequences)}",
                                flush=True,
                            )
                # ────────────────────────────────────────────────────────────

                for seq in sequences:
                    q = self.output_queues.get(seq.seq_id)
                    if q is not None:
                        q.put_nowait(next_tokens[seq.seq_id])
                for seq in completed:
                    q = self.output_queues.get(seq.seq_id)
                    if q is not None:
                        q.put_nowait(None)
                
                await asyncio.sleep(0)

            except Exception:
                traceback.print_exc()


    def add_request(
        self, prompt_token_ids: list[int], sampling_params: SamplingParams
    ) -> int:
        seq_id = self._next_seq_id()
        queue: asyncio.Queue = asyncio.Queue()
        self.output_queues[seq_id] = queue  # queue before scheduler so engine loop never misses it
        seq = Sequence(
            seq_id=seq_id,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
        )
        self._ttft_arrival[seq_id] = time.perf_counter()  # ttft scaffolding
        self.scheduler.add_sequence(seq)
        return seq_id

    async def stream_tokens(self, seq_id: int) -> AsyncGenerator[int, None]:
        queue = self.output_queues[seq_id]
        try:
            while True:
                token_id = await queue.get()
                if token_id is None:
                    break
                yield token_id
        finally:
            self.output_queues.pop(seq_id, None)

def create_app(
    server_config: ServerConfig,
    model_args: ModelArgs,
    model: Model,
    tokenizer,
) -> FastAPI:
    model_config = _model_config_from_args(model_args, tokenizer)
    engine = InferenceEngine(server_config, model_config, model, tokenizer)

    print("Warming up (compiling kernels)...", flush=True)
    _t = time.time()
    engine.warmup()
    print(f"Warmup complete in {time.time() - _t:.1f}s.", flush=True)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await engine.start()
        yield
        await engine.stop()

    app = FastAPI(title="Inference Server", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── helpers ──────────────────────────────────────────────────────────────

    def _sampling_params(req) -> SamplingParams:
        return SamplingParams(
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            max_tokens=req.max_tokens,
            repetition_penalty=req.repetition_penalty,   # add this
        )

    def _sse(data: str) -> str:
        return f"data: {data}\n\n"

    async def _completion_stream(seq_id: int, req_id: str, model_name: str):
        async for token_id in engine.stream_tokens(seq_id):
            text = tokenizer.decode([token_id], skip_special_tokens=True)
            event = {
                "id": req_id,
                "object": "text_completion",
                "created": int(time.time()),
                "model": model_name,
                "choices": [{"text": text, "index": 0, "finish_reason": None}],
            }
            yield _sse(json.dumps(event))
        yield _sse("[DONE]")

    async def _chat_stream(seq_id: int, req_id: str, model_name: str):
        async for token_id in engine.stream_tokens(seq_id):
            text = tokenizer.decode([token_id], skip_special_tokens=True)
            event = {
                "id": req_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_name,
                "choices": [
                    {"index": 0, "delta": {"content": text}, "finish_reason": None}
                ],
            }
            yield _sse(json.dumps(event))
        yield _sse("[DONE]")

    # ── routes ───────────────────────────────────────────────────────────────

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models():
        return ModelList(data=[ModelCard(id=engine.model_name)])

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest):
        token_ids = tokenizer.encode(req.prompt)
        seq_id = engine.add_request(token_ids, _sampling_params(req))
        req_id = f"cmpl-{uuid.uuid4().hex[:8]}"

        if req.stream:
            return StreamingResponse(
                _completion_stream(seq_id, req_id, engine.model_name),
                media_type="text/event-stream",
            )

        # Non-streaming: collect all tokens then return
        output_tokens: list[int] = []
        async for token_id in engine.stream_tokens(seq_id):
            output_tokens.append(token_id)
        text = tokenizer.decode(output_tokens, skip_special_tokens=True)
        return {
            "id": req_id,
            "object": "text_completion",
            "created": int(time.time()),
            "model": engine.model_name,
            "choices": [{"text": text, "index": 0, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": len(token_ids), "completion_tokens": len(output_tokens)},
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest):
        messages = [{"role": m.role, "content": m.content} for m in req.messages]
        token_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        seq_id = engine.add_request(token_ids, _sampling_params(req))
        req_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

        if req.stream:
            return StreamingResponse(
                _chat_stream(seq_id, req_id, engine.model_name),
                media_type="text/event-stream",
            )

        output_tokens: list[int] = []
        async for token_id in engine.stream_tokens(seq_id):
            output_tokens.append(token_id)
        text = tokenizer.decode(output_tokens, skip_special_tokens=True)
        return {
            "id": req_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": engine.model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": len(token_ids), "completion_tokens": len(output_tokens)},
        }

    frontend = os.path.join(os.path.dirname(__file__), "..", "frontend")
    if os.path.isdir(frontend):
        app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")


    return app
