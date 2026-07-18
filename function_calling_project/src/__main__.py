# ABOUTME: CLI entry point for the function calling tool.
# ABOUTME: Parses arguments, loads inputs, runs generation for each prompt, writes output.

"""
Usage:
    uv run python -m src [--functions_definition <path>] [--input <path>] [--output <path>]

Defaults:
    --functions_definition -> data/input/functions_definition.json
    --input                -> data/input/function_calling_tests.json
    --output                -> data/output/function_calling_results.json

TODO:
    1. Parse CLI args with argparse (all three optional, with the defaults above).
    2. Load functions_definition.json into a FunctionRegistry (see schema.py).
    3. Load function_calling_tests.json into a list of prompts.
    4. Instantiate the LLM (Small_LLM_Model) and the vocab lookup (see vocab.py).
    5. For each prompt, run the constrained generation loop (see generator.py)
       to obtain {"prompt", "name", "parameters"}.
    6. Write the collected results as a JSON array to the output path.
    7. Handle errors gracefully at every step (missing files, bad JSON, etc.)
       per the "General Rules" in the subject -- never let the program crash
       with an unhandled exception.
"""


def main() -> None:
    """Entry point: parse args, run the pipeline, write results."""
    raise NotImplementedError("TODO: wire up argparse + pipeline")


if __name__ == "__main__":
    main()
