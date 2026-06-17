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
from engine.runner import ModelRunner, build_batch
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

    def _next_seq_id(self) -> int:
        seq_id = self._seq_counter
        self._seq_counter += 1
        return seq_id

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
        while True:
            try:
                schedule = self.scheduler.step()
                if not schedule.prefill_sequences and not schedule.decode_sequences:
                    await asyncio.sleep(0.001)
                    continue

                sequences = schedule.prefill_sequences + schedule.decode_sequences
                batch = build_batch(schedule)

                logits = self.runner.forward(batch)
                next_tokens = sample(logits, sequences)
                completed = self.scheduler.update(next_tokens)

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
