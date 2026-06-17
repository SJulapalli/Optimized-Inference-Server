import mlx.core as mx
import mlx_lm
from model.llama import Model, ModelArgs
import json
from pathlib import Path
import glob

def load_config(model_path:str) -> ModelArgs:
    with open(Path(model_path) / "config.json") as f:
        config = json.load(f)
        
    return ModelArgs.from_dict(config)

def load_weights(model_path: str, model:Model) -> None:
    weight_files = glob.glob(str(Path(model_path) / "*.safetensors"))
    weights = {}
    
    for path in sorted(weight_files):
        weights.update(mx.load(path))
    
    weights = model.sanitize(weights)
    model.load_weights(list(weights.items()))
    mx.eval(model.parameters())