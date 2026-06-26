import argparse, asyncio, json, time, statistics

PROMPTS = [
    "Explain what a KV cache is in one paragraph.",
    "Write a haiku about GPUs.",
    "List three benefits of continuous batching.",
    "What is the capital of France, and why is it famous?",
    "Describe the difference between prefill and decode in LLM inference.",
    "Give me a recipe for a simple pasta dish.",
    "Summarize the theory of relativity for a 10-year-old.",
    "Translate 'good morning' into five languages.",
]


def make_prompts(n):
    return [PROMPTS[i % len(PROMPTS)] for i in range(n)]


# ---------- Server (continuous batching) ----------
async def _server_one(client, url, model, prompt, max_tokens, idx, gt0):
    submit = time.perf_counter() - gt0          # when this request was actually sent (from global start)
    payload = {"model": model, "prompt": prompt,
               "max_tokens": max_tokens, "temperature": 0.0, "stream": True}
    t0, ttft, n_tok = time.perf_counter(), None, 0
    async with client.stream("POST", f"{url}/v1/completions", json=payload) as resp:
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            text = json.loads(data)["choices"][0].get("text", "")
            if text == "":
                continue
            if ttft is None:
                ttft = time.perf_counter() - t0
            n_tok += 1
    return idx, submit, ttft, n_tok, time.perf_counter() - t0


async def run_server(url, model, prompts, max_tokens, concurrency):
    import httpx
    sem = asyncio.Semaphore(concurrency)
    gt0 = time.perf_counter()

    async def _worker(client, idx, prompt):
        async with sem:  # cap requests in flight so the batch stays full, not drained
            return await _server_one(client, url, model, prompt, max_tokens, idx, gt0)

    async with httpx.AsyncClient(timeout=None) as client:
        detailed = await asyncio.gather(
            *[_worker(client, i, p) for i, p in enumerate(prompts)])
    wall = time.perf_counter() - gt0
    return detailed, wall


def report_per_request(detailed):
    print(f"\n  {'idx':>3} {'submit(s)':>10} {'ttft(ms)':>9} {'1st-tok@(s)':>11} {'toks':>5}")
    for idx, submit, ttft, n_tok, _ in sorted(detailed, key=lambda r: r[0]):
        first_at = submit + (ttft or 0)          # absolute time of first token from global start
        ttft_ms = f"{ttft * 1000:.0f}" if ttft is not None else "-"
        print(f"  {idx:>3} {submit:>10.2f} {ttft_ms:>9} {first_at:>11.2f} {n_tok:>5}")


# ---------- Ablation: YOUR model + mlx_lm contiguous cache, single sequence ----------
def run_contiguous(model_path, prompts, max_tokens):
    import mlx.core as mx
    from model.llama import Model
    from model.weights import load_config, load_weights
    from transformers import AutoTokenizer

    args = load_config(model_path)
    model = Model(args)
    load_weights(model_path, model)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    eos = tokenizer.eos_token_id

    def generate(ids):
        cache = model.make_cache()
        ts, ttft = time.perf_counter(), None
        pos = len(ids)
        logits = model(mx.array(ids)[None], cache=cache,
                       positions=mx.arange(len(ids)), block_mask=None)
        tok = mx.argmax(logits[0, -1]).item()
        ttft = time.perf_counter() - ts
        n_tok = 1
        while n_tok < max_tokens and tok != eos:
            logits = model(mx.array([tok])[None], cache=cache,
                           positions=mx.array([pos]), block_mask=None)
            tok = mx.argmax(logits[0, -1]).item()
            pos += 1
            n_tok += 1
        return ttft, n_tok, time.perf_counter() - ts

    generate(tokenizer.encode("warmup"))  # absorb cold kernel-compile off the clock
    detailed, gt0 = [], time.perf_counter()
    for idx, p in enumerate(prompts):
        submit = time.perf_counter() - gt0          # sequential: request idx starts after idx-1 finishes
        ttft, n_tok, elapsed = generate(tokenizer.encode(p))
        detailed.append((idx, submit, ttft, n_tok, elapsed))
    return detailed, time.perf_counter() - gt0


# ---------- Baseline (sequential mlx_lm) ----------
def run_baseline(model_path, prompts, max_tokens):
    from mlx_lm import load, stream_generate
    model, tokenizer = load(model_path)
    detailed, gt0 = [], time.perf_counter()
    for idx, p in enumerate(prompts):
        submit = time.perf_counter() - gt0          # sequential: request idx starts after idx-1 finishes
        ts, ttft, n_tok = time.perf_counter(), None, 0
        for _ in stream_generate(model, tokenizer, p, max_tokens=max_tokens):
            if ttft is None:
                ttft = time.perf_counter() - ts
            n_tok += 1
        detailed.append((idx, submit, ttft, n_tok, time.perf_counter() - ts))
    return detailed, time.perf_counter() - gt0


def report(name, results, wall):
    ttfts = [r[0] for r in results if r[0] is not None]
    total_tok = sum(r[1] for r in results)
    print(f"\n=== {name} ===")
    print(f"requests:     {len(results)}")
    print(f"wall time:    {wall:.2f} s")
    print(f"total tokens: {total_tok}")
    print(f"throughput:   {total_tok / wall:.1f} tok/s (aggregate)")
    print(f"TTFT mean:    {statistics.mean(ttfts) * 1000:.0f} ms")
    print(f"TTFT median:  {statistics.median(ttfts) * 1000:.0f} ms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["server", "baseline", "contiguous"], required=True)
    ap.add_argument("--num-prompts", type=int, default=32,
                    help="total requests sent (keep >> concurrency for steady state)")
    ap.add_argument("--concurrency", type=int, default=8,
                    help="server mode: max requests in flight at once")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--model", default="llama")
    ap.add_argument("--model-path", help="weights path (baseline mode)")
    args = ap.parse_args()

    prompts = make_prompts(args.num_prompts)
    if args.mode == "server":
        detailed, wall = asyncio.run(
            run_server(args.url, args.model, prompts, args.max_tokens, args.concurrency))
        # detailed: (idx, submit, ttft, n_tok, elapsed) -> adapt to report's (ttft, n_tok, elapsed)
        results = [(ttft, n_tok, elapsed) for _, _, ttft, n_tok, elapsed in detailed]
        report(f"SERVER (continuous batching, concurrency={args.concurrency})", results, wall)
        report_per_request(detailed)
    elif args.mode == "contiguous":
        if not args.model_path:
            ap.error("--model-path is required for contiguous mode")
        detailed, wall = run_contiguous(args.model_path, prompts, args.max_tokens)
        results = [(ttft, n_tok, elapsed) for _, _, ttft, n_tok, elapsed in detailed]
        report("ABLATION (your model + mlx_lm contiguous cache, single seq)", results, wall)
        report_per_request(detailed)
    else:
        if not args.model_path:
            ap.error("--model-path is required for baseline mode")
        detailed, wall = run_baseline(args.model_path, prompts, args.max_tokens)
        results = [(ttft, n_tok, elapsed) for _, _, ttft, n_tok, elapsed in detailed]
        report("BASELINE (sequential mlx_lm)", results, wall)
        report_per_request(detailed)


if __name__ == "__main__":
    main()
