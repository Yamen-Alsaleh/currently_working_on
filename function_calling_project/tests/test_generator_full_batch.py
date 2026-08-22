# tests/test_generator_full_batch.py -- run manually against the real model
import json
import re
import time

from llm_sdk import Small_LLM_Model
from vocab import TokenVocab
from schema import FunctionRegistry
from generator import generate_function_call
from grammar import GrammarError

EXPECTED = {
    "What is the sum of 2 and 3?": ("fn_add_numbers", {"a": 2, "b": 3}),
    "What is the sum of 265 and 345?": (
        "fn_add_numbers", {"a": 265, "b": 345}
    ),
    "Greet shrek": ("fn_greet", {"name": "shrek"}),
    "Greet john": ("fn_greet", {"name": "john"}),
    "Reverse the string 'hello'": ("fn_reverse_string", {"s": "hello"}),
    "Reverse the string 'world'": ("fn_reverse_string", {"s": "world"}),
    "What is the square root of 16?": ("fn_get_square_root", {"a": 16}),
    "Calculate the square root of 144": ("fn_get_square_root", {"a": 144}),
    'Replace all numbers in "Hello 34 I\'m 233 years old" with NUMBERS': (
        "fn_substitute_string_with_regex",
        {
            "source_string": "Hello 34 I'm 233 years old",
            "regex": "[0-9]+",
            "replacement": "NUMBERS",
        },
    ),
    "Replace all vowels in 'Programming is fun' with asterisks": (
        "fn_substitute_string_with_regex",
        {
            "source_string": "Programming is fun",
            "regex": "[aeiou]",
            "replacement": "*",
        },
    ),
    (
        "Substitute the word 'cat' with 'dog' in "
        "'The cat sat on the mat with another cat'"
    ): (
        "fn_substitute_string_with_regex",
        {
            "source_string": "The cat sat on the mat with another cat",
            "regex": "cat",
            "replacement": "dog",
        },
    ),
}

EXPECTED_SUBSTITUTIONS = {
    'Replace all numbers in "Hello 34 I\'m 233 years old" with NUMBERS': (
        "Hello NUMBERS I'm NUMBERS years old"
    ),
    "Replace all vowels in 'Programming is fun' with asterisks": (
        "Pr*gr*mm*ng *s f*n"
    ),
    (
        "Substitute the word 'cat' with 'dog' in "
        "'The cat sat on the mat with another cat'"
    ): "The dog sat on the mat with another dog",
}


def is_correct_result(prompt, result, expected_name, expected_parameters):
    """Check exact values, or functional equivalence for regex patterns."""
    if result["name"] != expected_name:
        return False
    if expected_name != "fn_substitute_string_with_regex":
        return result["parameters"] == expected_parameters

    parameters = result["parameters"]
    if parameters.get("source_string") != expected_parameters["source_string"]:
        return False
    if parameters.get("replacement") != expected_parameters["replacement"]:
        return False
    try:
        actual = re.sub(
            parameters["regex"],
            parameters["replacement"],
            parameters["source_string"],
        )
    except (KeyError, re.error, TypeError):
        return False
    return actual == EXPECTED_SUBSTITUTIONS[prompt]

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
        expected_name, expected_parameters = EXPECTED[prompt]
        correct = is_correct_result(
            prompt,
            result,
            expected_name,
            expected_parameters,
        )
        status = "PASS" if correct else "WRONG"
        print(f"[{elapsed:.2f}s] {status:5s} {prompt!r}")
        print(f"          -> {result}")
        if not correct:
            print(
                "          expected -> "
                f"name={expected_name!r}, parameters={expected_parameters!r}"
            )
        results.append(result if correct else None)
    except GrammarError as e:
        elapsed = time.time() - t0
        print(f"[{elapsed:.2f}s] FAIL {prompt!r}")
        print(f"          -> {e}")
        results.append(None)

total_elapsed = time.time() - total_t0
passed = sum(1 for result in results if result is not None)
print(f"\n{passed}/{len(prompts)} semantically correct")
print(f"Total time: {total_elapsed:.2f}s")
