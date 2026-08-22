# tests/test_generator_full_batch.py -- run manually against the real model
import json
import time

from llm_sdk import Small_LLM_Model
from vocab import TokenVocab
from schema import FunctionRegistry
from generator import generate_function_call
from grammar import GrammarError

print("Loading model...")
model = Small_LLM_Model()

print("Loading vocab...")
vocab = TokenVocab.load(model.get_path_to_vocab_file())

print("Loading function registry...")
registry = FunctionRegistry.load("data/input/functions_definition.json")

with open("data/input/function_calling_tests.json") as f:
    prompts = [entry["prompt"] for entry in json.load(f)]

results = []
total_t0 = time.time()
for prompt in prompts:
    t0 = time.time()
    try:
        result = generate_function_call(prompt, model, vocab, registry)
        elapsed = time.time() - t0
        print(f"[{elapsed:.2f}s] OK   {prompt!r}")
        print(f"          -> {result}")
        results.append(result)
    except GrammarError as e:
        elapsed = time.time() - t0
        print(f"[{elapsed:.2f}s] FAIL {prompt!r}")
        print(f"          -> {e}")
        results.append(None)

total_elapsed = time.time() - total_t0
succeeded = sum(1 for r in results if r is not None)
print(f"\n{succeeded}/{len(prompts)} completed without error")
print(f"Total time: {total_elapsed:.2f}s")
