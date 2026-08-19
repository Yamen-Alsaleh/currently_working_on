from pathlib import Path
import json
from pydantic import (
    BaseModel,
    PrivateAttr,
    ValidationError,
    field_validator,
    model_validator,
)


class SchemaError(Exception):
    pass


class ParameterSchema(BaseModel):
    type: str

    @field_validator("type")
    @classmethod
    def check_type(cls, value: str) -> str:
        if value not in ["number", "string", "boolean"]:
            raise ValueError(f"unsupported parameter type: {value!r}")
        return value


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, ParameterSchema]
    returns: ParameterSchema

    def required_parameter_names(self) -> list[str]:
        return list(self.parameters.keys())

    def parameter_type(self, param_name: str) -> str:
        try:
            return self.parameters[param_name].type
        except KeyError:
            raise KeyError(
                f"{param_name} isn't one of the function's parameters"
            )


class FunctionRegistry(BaseModel):
    functions: list[FunctionDefinition]
    _by_name: dict[str, FunctionDefinition] = PrivateAttr(default_factory=dict)

    @model_validator(mode="after")
    def build_lookup(self) -> "FunctionRegistry":
        self._by_name = {}
        for f in self.functions:
            if f.name in self._by_name:
                raise ValueError(f"duplicate function name: {f.name!r}")
            else:
                self._by_name[f.name] = f
        return self

    @classmethod
    def load(cls, path: str | Path) -> "FunctionRegistry":
        try:
            with open(path, "r") as file:
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
        return list(self._by_name.keys())

    def get(self, name: str) -> FunctionDefinition:
        try:
            return self._by_name[name]
        except KeyError as e:
            raise SchemaError(f"unknown function name: {name!r}") from e

    def contains(self, name: str) -> bool:
        return name in self._by_name

    def __len__(self) -> int:
        return len(self._by_name)
