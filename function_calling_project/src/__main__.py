"""Command-line entry point for the constrained function-calling tool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Sequence

from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from .generator import generate_function_call
from .grammar import GrammarError
from .schema import FunctionRegistry, SchemaError
from .vocab import TokenVocab, VocabError

DEFAULT_FUNCTIONS_PATH = Path("data/input/functions_definition.json")
DEFAULT_INPUT_PATH = Path("data/input/function_calling_tests.json")
DEFAULT_OUTPUT_PATH = Path("data/output/function_calling_results.json")


class InputError(Exception):
    """Report an invalid or unreadable prompt input file."""


class PromptEntry(BaseModel):
    """Validate one object from the prompt input array."""

    model_config = ConfigDict(extra="forbid", strict=True)

    prompt: str


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Translate natural-language prompts into schema-conforming "
            "function calls using constrained LLM decoding."
        )
    )
    parser.add_argument(
        "--functions_definition",
        type=Path,
        default=DEFAULT_FUNCTIONS_PATH,
        help="function schema JSON file (default: %(default)s)",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="prompt input JSON file (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="result JSON file (default: %(default)s)",
    )
    return parser


def load_prompts(path: Path) -> list[str]:
    """Load and validate the JSON array of prompt objects.

    Args:
        path: Location of the prompt test JSON file.

    Returns:
        Prompts in their original order.

    Raises:
        InputError: If the file cannot be read or does not match the expected
            array of prompt objects.
    """
    try:
        with path.open("r", encoding="utf-8") as stream:
            raw_data: Any = json.load(stream)
    except json.JSONDecodeError as error:
        raise InputError(f"{path} is not valid JSON: {error}") from error
    except (OSError, UnicodeDecodeError) as error:
        raise InputError(f"could not read {path}: {error}") from error

    if not isinstance(raw_data, list):
        raise InputError(
            f"{path} must contain a JSON array, "
            f"got {type(raw_data).__name__}"
        )

    adapter: TypeAdapter[list[PromptEntry]] = TypeAdapter(list[PromptEntry])
    try:
        entries = adapter.validate_python(raw_data, strict=True)
    except ValidationError as error:
        raise InputError(
            f"{path} does not match the expected prompt schema: {error}"
        ) from error
    return [entry.prompt for entry in entries]


def write_results(path: Path, results: list[dict[str, Any]]) -> None:
    """Create the output directory and write a formatted JSON result array.

    Args:
        path: Destination JSON path.
        results: Generated function calls.

    Raises:
        OSError: If the directory or output file cannot be written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(results, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def run_pipeline(
    functions_path: Path,
    input_path: Path,
    output_path: Path,
) -> None:
    """Load inputs, generate every function call, and write the output file."""
    registry = FunctionRegistry.load(functions_path)
    prompts = load_prompts(input_path)

    if not prompts:
        write_results(output_path, [])
        print(f"Wrote 0 function calls to {output_path}")
        return

    print("Loading Qwen/Qwen3-0.6B...", file=sys.stderr)
    model = Small_LLM_Model()
    vocab = TokenVocab.load(model.get_path_to_vocab_file())

    results: list[dict[str, Any]] = []
    total = len(prompts)
    started = time.perf_counter()
    for index, prompt in enumerate(prompts, start=1):
        prompt_started = time.perf_counter()
        try:
            result = generate_function_call(prompt, model, vocab, registry)
        except GrammarError as error:
            raise GrammarError(
                f"generation failed for prompt {index}/{total} "
                f"({prompt!r}): {error}"
            ) from error
        results.append(result)
        elapsed = time.perf_counter() - prompt_started
        print(
            f"[{index}/{total}] generated in {elapsed:.2f}s",
            file=sys.stderr,
        )

    write_results(output_path, results)
    total_elapsed = time.perf_counter() - started
    print(
        f"Wrote {total} function calls to {output_path} "
        f"in {total_elapsed:.2f}s"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and convert expected failures into clear error messages."""
    arguments = build_parser().parse_args(argv)
    try:
        run_pipeline(
            arguments.functions_definition,
            arguments.input,
            arguments.output,
        )
    except (InputError, SchemaError, VocabError, GrammarError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    except Exception as error:  # defensive boundary required by the subject
        print(
            f"error: unexpected {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
