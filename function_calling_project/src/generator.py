from llm_sdk import Small_LLM_Model
from vocab import TokenVocab
from schema import FunctionRegistry
from grammar import GenerationState, GrammarError


def _build_framing_text(prompt: str, registry: FunctionRegistry) -> str:
    lines = [
        "You are a function-calling assistant. Given a user request, "
        "select the single best matching function and extract only its "
        "required arguments. Output exactly one JSON object, then stop "
        "immediately -- no extra text, no further examples.",
        "",
        "Available functions:",
    ]
    for name in registry.function_names():
        fn = registry.get(name)
        params = ", ".join(
            f"{p} ({fn.parameter_type(p)})" for p in fn.required_parameter_names()
        )
        lines.append(f"- {name}({params}): {fn.description}")
    lines.append("")
    lines.append("Example:")
    lines.append("Request: Greet john")
    lines.append(
        '{"prompt": "Greet john", "name": "fn_greet", "parameters": {"name": "john"}}'
    )
    lines.append("")
    lines.append(f"Request: {prompt}")
    return "\n".join(lines)


def generate_function_call(
    prompt: str,
    model: Small_LLM_Model,
    vocab: TokenVocab,
    registry: FunctionRegistry,
    max_steps: int = 500,
) -> dict:
    framing_text = _build_framing_text(prompt, registry)
    tensor = model.encode(framing_text)
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
    if not state.is_done():
        raise GrammarError(
            f'couldn\'t generate function call for the prompt "{prompt}" within the max_steps'
        )
    return state.result()
