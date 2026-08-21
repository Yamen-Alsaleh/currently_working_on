from llm_sdk import Small_LLM_Model
from vocab import TokenVocab
from schema import FunctionRegistry
from grammar import GenerationState


def generate_function_call(
    prompt: str,
    model: Small_LLM_Model,
    vocab: TokenVocab,
    registry: FunctionRegistry,
    max_steps: int = 500,
) -> dict:
    tensor = model.encode(prompt)
    input_ids = tensor[0].tolist()
    state = GenerationState.start(prompt=prompt, registry=registry)
    current_step = 0
    while not state.is_done() and current_step < max_steps:
        logits = model.get_logits_from_input_ids(input_ids)
        valid_ids = state.valid_token_ids(vocab)
        token_id = max(valid_ids, key=lambda i: logits[i])
        input_ids.append(token_id)
        state.advance(token_id, vocab)
        current_step += 1
    return state.result()
