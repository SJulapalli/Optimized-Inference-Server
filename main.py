import argparse
import uvicorn
from model.llama import Model
from model.weights import load_config, load_weights
from transformers import AutoTokenizer
from config import ServerConfig
from server.app import create_app

def main():
    parser = argparse.ArgumentParser(description="Inference Server")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--max-num-blocks", type=int, default=2048)
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--prefill-chunk-size", type=int, default=512)
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    model_args = load_config(args.model_path)
    model = Model(model_args)
    load_weights(args.model_path, model)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    print("Model loaded.")

    server_config = ServerConfig(
        model_path=args.model_path,
        host=args.host,
        port=args.port,
        max_num_seqs=args.max_num_seqs,
        max_num_blocks=args.max_num_blocks,
        max_seq_len=args.max_seq_len,
        block_size=args.block_size,
        prefill_chunk_size=args.prefill_chunk_size
    )

    app = create_app(server_config, model_args, model, tokenizer)
    uvicorn.run(app, host=server_config.host, port=server_config.port)

if __name__ == "__main__":
    main()
