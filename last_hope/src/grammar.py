"""Schema-aware state machine used to constrain token generation."""

from enum import Enum
import json
from typing import Any
import re
from pydantic import BaseModel, Field

from .schema import FunctionDefinition, FunctionRegistry
from .vocab import TokenVocab

MAX_STRING_VALUE_LENGTH = 100
MAX_NUMBER_VALUE_LENGTH = 100
NO_MATCH_FUNCTION_NAME = "fn_no_matching_function"


class Phase(Enum):
    """Represent the next syntactic or semantic generation requirement."""

    EXPECT_OPEN_BRACE = 1
    EXPECT_KEY = 2
    EXPECT_COLON = 3
    EMIT_KNOWN_LITERAL = 4
    EXPECT_ENUM_VALUE = 5
    EXPECT_STRING_OPEN_QUOTE = 6
    EXPECT_FREE_STRING_VALUE = 7
    EXPECT_FREE_NUMBER_VALUE = 8
    EXPECT_FREE_BOOLEAN_VALUE = 9
    EXPECT_COMMA_OR_CLOSE = 10
    DONE = 11


class GrammarError(Exception):
    """Report an impossible or invalid grammar transition."""


class GenerationState(BaseModel):
    """Track constrained decoding progress for one function call."""

    prompt: str
    registry: FunctionRegistry
    phase: Phase
    candidate_menu: list[str] = Field(default_factory=list)
    progress: str
    chosen_function_name: str | None
    collected_parameters: dict[str, float | int | str | bool] = Field(
        default_factory=dict
    )
    top_level_key: str | None
    current_parameter_name: str | None
    string_escape_pending: bool = False

    def _has_repetition_tail(
        self, min_chunk: int = 3, repeats: int = 2
    ) -> bool:
        """Return whether the generated string ends in repeated chunks."""
        text = self.progress
        for chunk_len in range(min_chunk, len(text) // repeats + 1):
            tail = text[-chunk_len * repeats :]
            if len(tail) < chunk_len * repeats:
                continue
            chunk = tail[:chunk_len]
            if tail == chunk * repeats:
                return True
        return False

    def _prefix_match_token_ids(
        self, vocab: TokenVocab, candidates: list[str], quote: bool = True
    ) -> set[int]:
        """Return tokens that keep at least one exact candidate possible."""
        if quote:
            target_candidates = [f'"{cand}"' for cand in candidates]
        else:
            target_candidates = candidates
        survived_candidate = [
            cand
            for cand in target_candidates
            if cand.startswith(self.progress)
        ]
        remaining_suffix = [
            cand.removeprefix(self.progress) for cand in survived_candidate
        ]
        first_chars = {suffix[0] for suffix in remaining_suffix if suffix}
        candidate_ids: set[int] = set()
        for char in first_chars:
            candidate_ids.update(vocab.ids_starting_with(char))
        result = {
            token_id
            for token_id in candidate_ids
            for suffix in remaining_suffix
            if suffix.startswith(vocab.token_text(token_id))
        }
        return result

    def _all_required_parameters_collected(self) -> bool:
        """Return whether every required argument has been collected."""
        required = self._chosen_function().required_parameter_names()
        return set(self.collected_parameters.keys()) == set(required)

    def _chosen_function(self) -> FunctionDefinition:
        """Return the selected definition or report an invalid state."""
        if self.chosen_function_name is None:
            raise GrammarError("a function must be selected before parameters")
        return self.registry.get(self.chosen_function_name)

    def _current_parameter(self) -> str:
        """Return the current parameter name or report an invalid state."""
        if self.current_parameter_name is None:
            raise GrammarError("no current parameter is selected")
        return self.current_parameter_name

    def _parameter_type(self) -> str:
        """Return the current parameter's declared primitive type."""
        return self._chosen_function().parameter_type(
            self._current_parameter()
        )

    def _is_valid_number_prefix(self, candidate: str) -> bool:
        """Return whether text can still become the declared JSON number."""
        if candidate == "-":
            return True
        if self._parameter_type() == "integer":
            pattern = r"-?(?:0|[1-9][0-9]*)"
        else:
            pattern = r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]*)?"
        return re.fullmatch(pattern, candidate) is not None

    def _number_can_finish(self) -> bool:
        """Return whether the accumulated numeric prefix is complete."""
        if self._parameter_type() == "integer":
            pattern = r"-?(?:0|[1-9][0-9]*)"
        else:
            pattern = r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?"
        return re.fullmatch(pattern, self.progress) is not None

    def _number_terminator_ids(self, vocab: TokenVocab) -> set[int]:
        """Return the comma or brace that may terminate this parameter."""
        required = set(self._chosen_function().required_parameter_names())
        collected = set(self.collected_parameters) | {
            self._current_parameter()
        }
        raw = "}" if collected == required else ","
        token_id = vocab.id_of(raw)
        if token_id is None:
            raise GrammarError(f"{raw!r} not found in vocab")
        return {token_id}

    def valid_token_ids(self, vocab: TokenVocab) -> set[int]:
        """Compute token IDs allowed by the current grammar phase."""
        if self.phase == Phase.EXPECT_OPEN_BRACE:
            brace_id = vocab.id_of("{")
            if brace_id is None:
                raise GrammarError("'{' not found in vocab")
            return {brace_id}
        elif self.phase == Phase.EXPECT_COLON:
            colon_id = vocab.id_of(":")
            if colon_id is None:
                raise GrammarError("':' not found in vocab")
            return {colon_id}
        elif self.phase == Phase.EXPECT_KEY:
            return self._prefix_match_token_ids(vocab, self.candidate_menu)
        elif self.phase == Phase.EXPECT_ENUM_VALUE:
            return self._prefix_match_token_ids(vocab, self.candidate_menu)
        elif self.phase == Phase.EMIT_KNOWN_LITERAL:
            return self._prefix_match_token_ids(
                vocab, [json.dumps(self.prompt)], quote=False
            )
        elif self.phase == Phase.EXPECT_STRING_OPEN_QUOTE:
            quote_id = vocab.id_of('"')
            if quote_id is None:
                raise GrammarError("opening quote not found in vocab")
            return {quote_id}
        elif self.phase == Phase.EXPECT_COMMA_OR_CLOSE:
            if (
                self.top_level_key == "parameters"
                and self.current_parameter_name is not None
            ):
                if self._all_required_parameters_collected():
                    close_brace_id = vocab.id_of("}")
                    if close_brace_id is None:
                        raise GrammarError("'}' not found in vocab")
                    return {close_brace_id}
                else:
                    comma_id = vocab.id_of(",")
                    if comma_id is None:
                        raise GrammarError("',' is not found in vocab")
                    return {comma_id}
            elif (
                self.top_level_key == "parameters"
                and self.current_parameter_name is None
            ):
                close_brace_id = vocab.id_of("}")
                if close_brace_id is None:
                    raise GrammarError("'}' not found in vocab")
                return {close_brace_id}
            else:
                comma_id = vocab.id_of(",")
                if comma_id is None:
                    raise GrammarError("',' is not found in vocab")
                return {comma_id}
        elif self.phase == Phase.EXPECT_FREE_STRING_VALUE:
            if not self.string_escape_pending and (
                self._has_repetition_tail()
                or len(self.progress) >= MAX_STRING_VALUE_LENGTH
            ):
                close_id = vocab.id_of('"')
                if close_id is None:
                    raise GrammarError('"' + " not found in vocab")
                return {close_id}
            valid_token_ids = vocab.string_value_token_ids()
            if not valid_token_ids:
                raise GrammarError("no valid characters found!")
            return valid_token_ids
        elif self.phase == Phase.EXPECT_FREE_NUMBER_VALUE:
            if (
                len(self.progress) >= MAX_NUMBER_VALUE_LENGTH
                and self._number_can_finish()
            ):
                return self._number_terminator_ids(vocab)
            valid_token_ids = set()
            candidate_ids: set[int] = set()
            initial_chars = "0123456789-"
            if self._parameter_type() == "number":
                initial_chars += "."
            for char in initial_chars:
                candidate_ids.update(vocab.ids_starting_with(char))
            for token_id in candidate_ids:
                token_text = vocab.token_text(token_id)
                if self._is_valid_number_prefix(self.progress + token_text):
                    valid_token_ids.add(token_id)
            if self._number_can_finish():
                valid_token_ids.update(self._number_terminator_ids(vocab))
            if not valid_token_ids:
                raise GrammarError("no valid characters found!")
            return valid_token_ids
        elif self.phase == Phase.EXPECT_FREE_BOOLEAN_VALUE:
            return self._prefix_match_token_ids(
                vocab, ["true", "false"], quote=False
            )
        elif self.phase == Phase.DONE:
            raise GrammarError(
                "valid_token_ids called after generation is already DONE"
            )
        raise GrammarError(f"unsupported generation phase: {self.phase}")

    def advance(self, token_id: int, vocab: TokenVocab) -> None:
        """Consume one valid token and advance the grammar state."""
        if self.phase == Phase.EXPECT_OPEN_BRACE:
            if self.top_level_key == "parameters":
                function_def = self._chosen_function()
                param_names = function_def.required_parameter_names()
                if not param_names:
                    self.phase = Phase.EXPECT_COMMA_OR_CLOSE
                    self.current_parameter_name = None
                    self.progress = ""
                    return
                self.phase = Phase.EXPECT_KEY
                self.candidate_menu = param_names
                self.progress = ""
                return
            self.phase = Phase.EXPECT_KEY
            self.candidate_menu = ["prompt"]
            self.progress = ""
            return
        elif self.phase == Phase.EXPECT_COLON:
            if self.top_level_key == "prompt":
                self.phase = Phase.EMIT_KNOWN_LITERAL
            elif self.top_level_key == "name":
                self.phase = Phase.EXPECT_ENUM_VALUE
                self.candidate_menu = self.registry.function_names() + [
                    NO_MATCH_FUNCTION_NAME
                ]
                self.progress = ""
            elif self.top_level_key == "parameters":
                if self.current_parameter_name is None:
                    self.phase = Phase.EXPECT_OPEN_BRACE
                else:
                    param_type = self._chosen_function().parameter_type(
                        self.current_parameter_name
                    )
                    if param_type == "string":
                        self.phase = Phase.EXPECT_STRING_OPEN_QUOTE
                    elif param_type in ["number", "integer"]:
                        self.phase = Phase.EXPECT_FREE_NUMBER_VALUE
                    elif param_type == "boolean":
                        self.phase = Phase.EXPECT_FREE_BOOLEAN_VALUE
            return
        elif self.phase == Phase.EXPECT_KEY:
            self.progress = self.progress + vocab.token_text(token_id)
            quoted_candidates = [f'"{cand}"' for cand in self.candidate_menu]
            if self.progress in quoted_candidates:
                self.progress = self.progress.removeprefix('"')
                self.progress = self.progress.removesuffix('"')
                if self.top_level_key == "parameters":
                    self.current_parameter_name = self.progress
                else:
                    self.top_level_key = self.progress
                self.phase = Phase.EXPECT_COLON
                self.progress = ""
            return
        elif self.phase == Phase.EXPECT_ENUM_VALUE:
            self.progress = self.progress + vocab.token_text((token_id))
            quoted_candidates = [f'"{cand}"' for cand in self.candidate_menu]
            if self.progress in quoted_candidates:
                self.progress = self.progress.removeprefix('"')
                self.progress = self.progress.removesuffix('"')
                if self.progress == NO_MATCH_FUNCTION_NAME:
                    raise GrammarError(
                        f"no available function matches prompt {self.prompt!r}"
                    )
                self.chosen_function_name = self.progress
                self.phase = Phase.EXPECT_COMMA_OR_CLOSE
                self.progress = ""
            return
        elif self.phase == Phase.EMIT_KNOWN_LITERAL:
            self.progress = self.progress + vocab.token_text(token_id)
            if self.progress == json.dumps(self.prompt):
                self.progress = ""
                self.phase = Phase.EXPECT_COMMA_OR_CLOSE
            return
        elif self.phase == Phase.EXPECT_STRING_OPEN_QUOTE:
            self.progress = ""
            self.phase = Phase.EXPECT_FREE_STRING_VALUE
            self.string_escape_pending = False
            return
        elif self.phase == Phase.EXPECT_COMMA_OR_CLOSE:
            if (
                self.top_level_key == "parameters"
                and self.current_parameter_name is not None
            ):
                if self._all_required_parameters_collected():
                    self.current_parameter_name = None
                else:
                    next_param = None
                    required_names = (
                        self._chosen_function().required_parameter_names()
                    )
                    for name in required_names:
                        if name not in self.collected_parameters:
                            next_param = name
                            break
                    if next_param is None:
                        raise GrammarError(
                            "could not select the next parameter"
                        )
                    self.candidate_menu = [next_param]
                    self.phase = Phase.EXPECT_KEY
                    self.progress = ""
            elif self.top_level_key == "prompt":
                self.candidate_menu = ["name"]
                self.phase = Phase.EXPECT_KEY
                self.progress = ""
            elif self.top_level_key == "name":
                self.candidate_menu = ["parameters"]
                self.phase = Phase.EXPECT_KEY
                self.progress = ""
            elif self.top_level_key == "parameters":
                self.phase = Phase.DONE
            return
        elif self.phase == Phase.EXPECT_FREE_STRING_VALUE:
            token_text = vocab.token_text(token_id)
            if not self.string_escape_pending and token_text == '"':
                try:
                    decoded = json.loads(f'"{self.progress}"')
                except json.JSONDecodeError as error:
                    raise GrammarError(
                        "could not decode generated string"
                    ) from error
                self.collected_parameters[self._current_parameter()] = decoded
                self.progress = ""
                self.string_escape_pending = False
                self.phase = Phase.EXPECT_COMMA_OR_CLOSE
            else:
                self.progress += token_text
                self.string_escape_pending = (
                    not self.string_escape_pending and token_text == "\\"
                )
            return
        elif self.phase == Phase.EXPECT_FREE_NUMBER_VALUE:
            token_text = vocab.token_text(token_id)
            if token_text in ",}":
                if self._parameter_type() == "number":
                    number_value: int | float = float(self.progress)
                else:
                    number_value = int(self.progress)
                self.collected_parameters[self._current_parameter()] = (
                    number_value
                )
                self.progress = ""
                if token_text == "}":
                    self.current_parameter_name = None
                    self.phase = Phase.EXPECT_COMMA_OR_CLOSE
                else:
                    required_names = (
                        self._chosen_function().required_parameter_names()
                    )
                    next_param = next(
                        (
                            name
                            for name in required_names
                            if name not in self.collected_parameters
                        ),
                        None,
                    )
                    if next_param is None:
                        raise GrammarError(
                            "number ended with a comma after the "
                            "last parameter"
                        )
                    self.candidate_menu = [next_param]
                    self.phase = Phase.EXPECT_KEY
            else:
                self.progress += token_text
            return

        elif self.phase == Phase.EXPECT_FREE_BOOLEAN_VALUE:
            self.progress = self.progress + vocab.token_text(token_id)
            if self.progress in ["true", "false"]:
                self.collected_parameters[self._current_parameter()] = (
                    self.progress == "true"
                )
                self.progress = ""
                self.phase = Phase.EXPECT_COMMA_OR_CLOSE
            return
        elif self.phase == Phase.DONE:
            raise GrammarError(
                "advance called after generation is already DONE"
            )
        else:
            raise GrammarError(f"unsupported generation phase: {self.phase}")

    def is_done(self) -> bool:
        """Return whether the complete function-call object was generated."""
        return self.phase == Phase.DONE

    def result(self) -> dict[str, Any]:
        """Return the collected call after generation reaches DONE."""
        if self.is_done():
            return {
                "prompt": self.prompt,
                "name": self.chosen_function_name,
                "parameters": self.collected_parameters,
            }
        raise GrammarError("result requested before generation completed")

    @classmethod
    def start(
        cls, prompt: str, registry: FunctionRegistry
    ) -> "GenerationState":
        """Create an initial state for a new natural-language prompt."""
        return cls(
            prompt=prompt,
            registry=registry,
            phase=Phase.EXPECT_OPEN_BRACE,
            candidate_menu=[],
            progress="",
            chosen_function_name=None,
            collected_parameters={},
            top_level_key=None,
            current_parameter_name=None,
            string_escape_pending=False,
        )
