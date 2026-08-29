"""Pydantic models and validation for function-definition files."""

import json
from pathlib import Path

from pydantic import (
    BaseModel,
    Field,
    PrivateAttr,
    ValidationError,
    field_validator,
    model_validator,
)


class SchemaError(Exception):
    """Report an invalid or unreadable function-definition file."""


class ParameterSchema(BaseModel):
    """Describe one supported function parameter or return value."""

    type: str

    @field_validator("type")
    @classmethod
    def check_type(cls, value: str) -> str:
        """Reject parameter types unsupported by the decoder."""
        if value not in ["number", "string", "boolean"]:
            raise ValueError(f"unsupported parameter type: {value!r}")
        return value


class FunctionDefinition(BaseModel):
    """Describe one function available for model selection."""

    name: str
    description: str
    parameters: dict[str, ParameterSchema]
    returns: ParameterSchema

    def required_parameter_names(self) -> list[str]:
        """Return parameter names in their schema-defined order."""
        return list(self.parameters.keys())

    def parameter_type(self, param_name: str) -> str:
        """Return a parameter's declared primitive type."""
        try:
            return self.parameters[param_name].type
        except KeyError as error:
            raise KeyError(
                f"{param_name} isn't one of the function's parameters"
            ) from error


class FunctionRegistry(BaseModel):
    """Validate functions and provide efficient name-based lookup."""

    functions: list[FunctionDefinition] = Field(min_length=1)
    _by_name: dict[str, FunctionDefinition] = PrivateAttr(default_factory=dict)

    @model_validator(mode="after")
    def build_lookup(self) -> "FunctionRegistry":
        """Build an idempotent lookup and reject duplicate names."""
        self._by_name = {}
        for f in self.functions:
            if f.name in self._by_name:
                raise ValueError(f"duplicate function name: {f.name!r}")
            else:
                self._by_name[f.name] = f
        return self

    @classmethod
    def load(cls, path: str | Path) -> "FunctionRegistry":
        """Load and validate a JSON function-definition array."""
        try:
            with open(path, "r", encoding="utf-8") as file:
                text = file.read()
                try:
                    parsed_list = json.loads(text)
                except json.JSONDecodeError as e:
                    raise SchemaError(f"{path} is not valid JSON: {e}") from e

                if not isinstance(parsed_list, list):
                    raise SchemaError(
                        f"{path} must contain a JSON array,"
                        + f" got {type(parsed_list).__name__}"
                    )

                try:
                    return cls(functions=parsed_list)
                except ValidationError as e:
                    raise SchemaError(
                        f"{path} does not match the expected schema: {e}"
                    ) from e

        except (OSError, UnicodeDecodeError) as e:
            raise SchemaError(f"could not read {path}: {e}") from e

    def function_names(self) -> list[str]:
        """Return all available function names in file order."""
        return list(self._by_name.keys())

    def get(self, name: str) -> FunctionDefinition:
        """Return a definition by name or raise SchemaError."""
        try:
            return self._by_name[name]
        except KeyError as e:
            raise SchemaError(f"unknown function name: {name!r}") from e

    def contains(self, name: str) -> bool:
        """Return whether a function name is registered."""
        return name in self._by_name

    def __len__(self) -> int:
        """Return the number of registered functions."""
        return len(self._by_name)
