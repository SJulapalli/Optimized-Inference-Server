import mlx.core as mx
from engine.sequence import SamplingParams, Sequence

def _apply_repetition_penalty(logits: mx.array, token_ids: list[int], penalty: float) -> mx.array:
    if not token_ids:
        return logits
    unique_ids = mx.array(list(set(token_ids)))
    penalty_arr = mx.ones(logits.shape)
    penalty_arr = penalty_arr.at[unique_ids].set(penalty)
    return mx.where(logits > 0, logits / penalty_arr, logits * penalty_arr)

def sample(logits_arrays: mx.array, sequences: list[Sequence]) -> dict[int, int]:
    sampled_tokens = {}
    
    for logits, sequence in zip(logits_arrays, sequences):
        sample_params = sequence.sampling_params
        if sample_params.repetition_penalty != 1.0:
            seen = sequence.prompt_token_ids + sequence.output_token_ids
            logits = _apply_repetition_penalty(logits, seen, sample_params.repetition_penalty)
            
        if sample_params.top_k != -1:
            sampled_tokens[sequence.seq_id] = _top_k_sample(logits, sample_params)
        elif sample_params.top_p != -1:
            sampled_tokens[sequence.seq_id] = _top_p_sample(logits, sample_params)
        else:
            sampled_tokens[sequence.seq_id] = _greedy(logits)
        
    return sampled_tokens

# All of these functions return the index of the selected logit, not the logit itself.
def _greedy(logits: mx.array) -> int:
    return mx.argmax(logits).item()

# Selects based on top k sampling, highest k probs are used as the distribution
def _top_k_sample(logits: mx.array, params: SamplingParams) -> int:
    scaled_logits = logits / params.temperature
    top_k = mx.sort(scaled_logits)[-params.top_k]
    masked_logits = mx.where(scaled_logits >= top_k, scaled_logits, -float('inf'))
    
    return mx.random.categorical(masked_logits).item()
    
# Nucleus sampling.
def _top_p_sample(logits: mx.array, params: SamplingParams) -> int:
    # Scale logits by temperature and construct sorted logit array
    temperature_scaled_logits = logits / params.temperature
    sorted_indices = mx.argsort(-logits)
    sorted_logits = temperature_scaled_logits[sorted_indices]
    
    # Form probabilities and cumsum to find breakpoint
    sorted_probabilities = mx.softmax(sorted_logits)
    cummulative_probs = mx.cumsum(sorted_probabilities)
    
    # Mask sorted logits by breakpoint
    masked_probs = mx.where(cummulative_probs <= params.top_p, sorted_logits, -float('inf'))
    
    # Sample from sorted logits to get sorted index, then map back to original logits index.
    return sorted_indices[mx.random.categorical(masked_probs).item()].item()