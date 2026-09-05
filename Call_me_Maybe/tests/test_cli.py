"""Unit tests for CLI input validation and output writing."""

import json
from pathlib import Path

import pytest

from src.__main__ import InputError, load_prompts, main, write_results


def test_load_prompts_preserves_order_and_original_text(
    tmp_path: Path,
) -> None:
    """Accept a valid prompt array without modifying prompt text."""
    input_path = tmp_path / "input.json"
    input_path.write_text(
        json.dumps([{"prompt": "hello"}, {"prompt": "  keep spaces  "}]),
        encoding="utf-8",
    )
    assert load_prompts(input_path) == ["hello", "  keep spaces  "]


@pytest.mark.parametrize(
    "contents",
    [
        "{bad json",
        '{"prompt": "not an array"}',
        '[{"wrong_key": "hello"}]',
        '[{"prompt": 42}]',
        '[{"prompt": "hello", "extra": true}]',
        '[{"prompt": ""}]',
        '[{"prompt": "   "}]',
    ],
)
def test_load_prompts_rejects_invalid_input(
    tmp_path: Path,
    contents: str,
) -> None:
    """Convert malformed input formats into InputError."""
    input_path = tmp_path / "input.json"
    input_path.write_text(contents, encoding="utf-8")
    with pytest.raises(InputError):
        load_prompts(input_path)


def test_write_results_creates_parent_and_valid_json(tmp_path: Path) -> None:
    """Create a missing output directory and serialize exact result data."""
    output_path = tmp_path / "nested" / "results.json"
    results = [
        {
            "prompt": "Greet Ada",
            "name": "fn_greet",
            "parameters": {"name": "Ada"},
        }
    ]
    write_results(output_path, results)
    assert json.loads(output_path.read_text(encoding="utf-8")) == results


def test_main_reports_missing_definition_without_traceback(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Return failure and print a concise error for a missing schema file."""
    exit_code = main(
        [
            "--functions_definition",
            str(tmp_path / "missing.json"),
            "--input",
            str(tmp_path / "input.json"),
            "--output",
            str(tmp_path / "output.json"),
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.err.startswith("error:")
    assert "Traceback" not in captured.err


def test_main_reports_missing_prompt_file_before_loading_model(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Reject an invalid --input path with a clear non-crashing error."""
    definitions = tmp_path / "functions.json"
    definitions.write_text(
        json.dumps(
            [
                {
                    "name": "fn_test",
                    "description": "Test.",
                    "parameters": {},
                    "returns": {"type": "string"},
                }
            ]
        ),
        encoding="utf-8",
    )
    exit_code = main(
        [
            "--functions_definition",
            str(definitions),
            "--input",
            str(tmp_path / "missing-prompts.json"),
            "--output",
            str(tmp_path / "output.json"),
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "could not read" in captured.err
    assert "Traceback" not in captured.err
