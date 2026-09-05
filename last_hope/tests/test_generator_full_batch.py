"""Validate the output produced by the real-model CLI integration run."""

import json
import re
from typing import Any

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


def is_correct_result(
    prompt: str,
    result: dict[str, Any],
    expected_name: str,
    expected_parameters: dict[str, Any],
) -> bool:
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


def main() -> None:
    """Validate every generated result without running the model again."""
    with open(
        "data/input/function_calling_tests.json",
        "r",
        encoding="utf-8",
    ) as stream:
        prompts = [entry["prompt"] for entry in json.load(stream)]

    with open(
        "data/output/function_calling_results.json",
        "r",
        encoding="utf-8",
    ) as stream:
        generated: Any = json.load(stream)

    if not isinstance(generated, list):
        raise SystemExit("FAIL: output root must be a JSON array")
    if len(generated) != len(prompts):
        raise SystemExit(
            f"FAIL: expected {len(prompts)} results, got {len(generated)}"
        )

    passed = 0
    for prompt, result in zip(prompts, generated):
        expected_name, expected_parameters = EXPECTED[prompt]
        correct = (
            isinstance(result, dict)
            and set(result) == {"prompt", "name", "parameters"}
            and result.get("prompt") == prompt
            and is_correct_result(
                prompt,
                result,
                expected_name,
                expected_parameters,
            )
        )
        status = "PASS" if correct else "WRONG"
        print(f"{status:5s} {prompt!r}")
        print(f"      -> {result}")
        if not correct:
            print(
                "      expected -> "
                f"name={expected_name!r}, "
                f"parameters={expected_parameters!r}"
            )
        passed += int(correct)

    print(f"\n{passed}/{len(prompts)} semantically correct")
    if passed != len(prompts):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
