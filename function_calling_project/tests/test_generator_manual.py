# scratch_test_generator.py -- run manually, not part of pytest suite
import time
from llm_sdk import Small_LLM_Model
from vocab import TokenVocab
from schema import FunctionRegistry
from generator import generate_function_call

print("Loading model (first run downloads Qwen3-0.6B)...")
model = Small_LLM_Model()

print("Loading vocab...")
vocab = TokenVocab.load(model.get_path_to_vocab_file())

print("Loading function registry...")
registry = FunctionRegistry.load("data/input/functions_definition.json")

test_prompts = [
    "What is the sum of 2 and 3?",
    "Greet shrek",
    "Reverse the string 'hello'",
]
# add this temporarily, right after loading model/vocab/registry, before the normal test loop
from grammar import GenerationState

print("\n--- DEBUG RUN ---")
prompt = "What is the sum of 2 and 3?"
tensor = model.encode(prompt)
input_ids = tensor[0].tolist()
state = GenerationState.start(prompt=prompt, registry=registry)
step = 0
while not state.is_done() and step < 60:
    logits = model.get_logits_from_input_ids(input_ids)
    valid_ids = state.valid_token_ids(vocab)
    token_id = max(valid_ids, key=lambda i: logits[i])
    chosen_text = vocab.token_text(token_id)
    print(
        f"[{step}] phase={state.phase.name:26s} progress={state.progress!r:15s} chose={chosen_text!r}"
    )
    input_ids.append(token_id)
    state.advance(token_id, vocab)
    step += 1
print("final phase:", state.phase, "| is_done:", state.is_done())
for prompt in test_prompts:
    t0 = time.time()
    result = generate_function_call(prompt, model, vocab, registry)
    elapsed = time.time() - t0
    print(f"\n[{elapsed:.2f}s] {prompt!r}")
    print(f"  -> {result}")
