# ABOUTME: Tests for schema.py -- loading and validating functions_definition.json.
# ABOUTME: Covers the happy path plus every graceful-failure case the subject requires.

import json

import pytest

from schema import FunctionRegistry, SchemaError

REAL_DEFINITIONS_PATH = "data/input/functions_definition.json"


def write_json(tmp_path, data) -> str:
    """Helper: dump `data` as JSON to a temp file, return its path as a string."""
    path = tmp_path / "functions_definition.json"
    path.write_text(json.dumps(data))
    return str(path)


def write_raw(tmp_path, text: str) -> str:
    """Helper: write raw (possibly invalid) text to a temp file, return its path."""
    path = tmp_path / "functions_definition.json"
    path.write_text(text)
    return str(path)


def test_happy_path_loads_real_definitions_file():
    registry = FunctionRegistry.load(REAL_DEFINITIONS_PATH)

    assert set(registry.function_names()) == {
        "fn_add_numbers",
        "fn_greet",
        "fn_reverse_string",
        "fn_get_square_root",
        "fn_substitute_string_with_regex",
    }
    assert registry.get("fn_greet").required_parameter_names() == ["name"]
    assert registry.get("fn_add_numbers").parameter_type("a") == "number"
    assert len(registry) == 5


def test_missing_file_raises_schema_error():
    with pytest.raises(SchemaError):
        FunctionRegistry.load("data/input/does_not_exist.json")


def test_malformed_json_raises_schema_error(tmp_path):
    bad_path = write_raw(tmp_path, '[{"name": "fn_x", "parameters": {},}]')

    with pytest.raises(SchemaError):
        FunctionRegistry.load(bad_path)


def test_non_list_json_raises_schema_error(tmp_path):
    bad_path = write_json(tmp_path, {"not": "a list"})

    with pytest.raises(SchemaError):
        FunctionRegistry.load(bad_path)


def test_missing_required_key_raises_schema_error(tmp_path):
    bad_path = write_json(tmp_path, [
        {
            "name": "fn_x",
            "description": "d",
            "returns": {"type": "number"},
            # "parameters" missing entirely
        }
    ])

    with pytest.raises(SchemaError):
        FunctionRegistry.load(bad_path)


def test_invalid_parameter_type_raises_schema_error(tmp_path):
    bad_path = write_json(tmp_path, [
        {
            "name": "fn_x",
            "description": "d",
            "parameters": {"a": {"type": "array"}},  # unsupported type
            "returns": {"type": "number"},
        }
    ])

    with pytest.raises(SchemaError):
        FunctionRegistry.load(bad_path)


def test_duplicate_function_names_raises_schema_error(tmp_path):
    bad_path = write_json(tmp_path, [
        {"name": "fn_dup", "description": "d1", "parameters": {}, "returns": {"type": "number"}},
        {"name": "fn_dup", "description": "d2", "parameters": {}, "returns": {"type": "number"}},
    ])

    with pytest.raises(SchemaError):
        FunctionRegistry.load(bad_path)


def test_get_unknown_function_name_raises_schema_error():
    registry = FunctionRegistry.load(REAL_DEFINITIONS_PATH)

    with pytest.raises(SchemaError):
        registry.get("fn_does_not_exist")


def test_contains_reflects_known_and_unknown_names():
    registry = FunctionRegistry.load(REAL_DEFINITIONS_PATH)

    assert registry.contains("fn_greet") is True
    assert registry.contains("fn_does_not_exist") is False
