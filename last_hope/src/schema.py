"""Pydantic models and validation for function-definition files."""

import json
from pathlib import Path
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    ValidationError,
    field_validator,
    model_validator,
)

PrimitiveValue = int | float | str | bool


class SchemaError(Exception):
    """Report an invalid or unreadable function-definition file."""


class ParameterSchema(BaseModel):
    """Describe one supported function parameter or return value."""

    model_config = ConfigDict(extra="forbid", strict=True)
    type: str
    enum: list[PrimitiveValue] | None = None

    @field_validator("type")
    @classmethod
    def check_type(cls, value: str) -> str:
        """Reject parameter types unsupported by the decoder."""
        if value not in ["number", "integer", "string", "boolean"]:
            raise ValueError(f"unsupported parameter type: {value!r}")
        return value

    @model_validator(mode="after")
    def check_enum(self) -> "ParameterSchema":
        """Require every optional enum value to match the declared type."""
        if self.enum is None:
            return self
        if not self.enum:
            raise ValueError("enum must contain at least one value")
        seen: set[tuple[type[object], str]] = set()
        for value in self.enum:
            if not self.accepts(value):
                raise ValueError(
                    f"enum value {value!r} does not match type {self.type!r}"
                )
            marker = (type(value), repr(value))
            if marker in seen:
                raise ValueError(f"duplicate enum value: {value!r}")
            seen.add(marker)
        return self

    def accepts(self, value: object) -> bool:
        """Return whether a Python value conforms to this parameter schema."""
        if self.type == "string":
            type_matches = type(value) is str
        elif self.type == "boolean":
            type_matches = type(value) is bool
        elif self.type == "integer":
            type_matches = type(value) is int
        else:
            type_matches = type(value) in (int, float)
        if not type_matches:
            return False
        return self.enum is None or value in self.enum


class FunctionDefinition(BaseModel):
    """Describe one function available for model selection."""

    model_config = ConfigDict(extra="forbid", strict=True)
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

    def parameter_schema(self, param_name: str) -> ParameterSchema:
        """Return the complete schema for one declared parameter."""
        try:
            return self.parameters[param_name]
        except KeyError as error:
            raise KeyError(
                f"{param_name} isn't one of the function's parameters"
            ) from error


class FunctionCall(BaseModel):
    """Validate the exact top-level shape of one generated function call."""

    model_config = ConfigDict(extra="forbid", strict=True)

    prompt: str
    name: str
    parameters: dict[str, PrimitiveValue]


class FunctionRegistry(BaseModel):
    """Validate functions and provide efficient name-based lookup."""

    model_config = ConfigDict(extra="forbid", strict=True)
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

    def validate_call(
        self,
        raw_call: dict[str, Any],
        expected_prompt: str,
    ) -> dict[str, Any]:
        """Validate a generated call against its selected function schema."""
        try:
            call = FunctionCall.model_validate(raw_call, strict=True)
        except ValidationError as error:
            raise SchemaError(
                f"generated call has an invalid output shape: {error}"
            ) from error
        if call.prompt != expected_prompt:
            raise SchemaError(
                "generated call did not preserve the input prompt"
            )

        function = self.get(call.name)
        required = set(function.required_parameter_names())
        received = set(call.parameters)
        if received != required:
            missing = sorted(required - received)
            extra = sorted(received - required)
            raise SchemaError(
                "generated parameters do not match the function schema: "
                f"missing={missing}, extra={extra}"
            )
        for name, value in call.parameters.items():
            parameter = function.parameter_schema(name)
            if not parameter.accepts(value):
                raise SchemaError(
                    f"generated parameter {name!r} has invalid value "
                    f"{value!r} for type {parameter.type!r}"
                )
        return call.model_dump()

    def __len__(self) -> int:
        """Return the number of registered functions."""
        return len(self._by_name)
