"""
This is the core of the "constrained decoding" requirement.

TODO:
    - Design a state object that holds (at minimum, per our design discussion):
        * a fixed reference to the FunctionRegistry (doesn't change during
          generation)
        * "scratchpad" data: whatever partial literal is currently being
          matched (e.g. a JSON key being spelled out, a function name being
          spelled out, a number being accumulated)
        * "decided" data: values that have been finalized earlier in this
          same generation and must be remembered for later (e.g. once
          "name" is fully decided, which function was chosen -- because
          that determines what "parameters" must contain)
        * an explicit notion of "phase" (expecting '{', expecting a key,
          expecting ':', expecting a value, expecting ',' or '}', done...)
    - Implement the core method: given the current state, return the set
      of valid token ids (by consulting the vocab lookup + the registry).
    - Implement a transition method: given the token id that was actually
      chosen, update the state (advance the phase, extend the scratchpad,
      or commit a decided value and reset the scratchpad).
    - Implement a "is this state complete" check (top-level object closed)
      to know when to stop generating for a given prompt.
    - Handle the "number" sub-grammar (digits, optional '-', optional
      single '.', terminated by ',' or '}') separately from the
      "literal/enum matching" sub-grammar (keys, function names, string
      values) used for "prompt", "name", string-typed parameters, etc.
"""

from enum import Enum
from pydantic import BaseModel, Field
from schema import FunctionRegistry
from vocab import TokenVocab


class Phase(Enum):
    EXPECT_OPEN_BRACE = 1
    EXPECT_KEY = 2
    EXPECT_COLON = 3
    EMIT_KNOWN_LITERAL = 4
    EXPECT_ENUM_VALUE = 5
    EXPECT_FREE_STRING_VALUE = 6
    EXPECT_FREE_NUMBER_VALUE = 7
    EXPECT_FREE_BOOLEAN_VALUE = 8
    EXPECT_COMMA_OR_CLOSE = 9
    DONE = 10


class GrammarError(Exception):
    pass


class GenerationState(BaseModel):
    prompt: str
    registry: FunctionRegistry
    phase: Phase
    candidate_menu: list[str] = Field(default_factory=list)
    progress: str
    chosen_function_name: str | None
    collected_parameters: dict[str, int | str | bool] = Field(
        default_factory=dict
    )
    top_level_key: str | None
    current_parameter_name: str | None

    def _prefix_match_token_ids(
        self, vocab: TokenVocab, candidates: list[str]
    ) -> set[int]:
        quoted_candidates = [f'"{cand}"' for cand in candidates]
        survived_candidate = [
            cand
            for cand in quoted_candidates
            if cand.startswith(self.progress)
        ]
        remaining_suffix = [
            cand.removeprefix(self.progress) for cand in survived_candidate
        ]
        result = {
            id
            for id, _ in vocab
            for suffix in remaining_suffix
            if suffix.startswith(vocab.token_text(id))
            and vocab.token_text(id) != ""
        }
        return result

    def _all_required_parameters_collected(self) -> bool:
        required = self.registry.get(
            self.chosen_function_name
        ).required_parameter_names()
        return set(self.collected_parameters.keys()) == set(required)

    def _is_valid_number_prefix(self, candidate: str) -> bool:
        if candidate == "":
            return False
        if candidate.__contains__("-") or candidate.__contains__("."):
            count_dot = 0
            count = 0
            for char in candidate:
                if char == "-":
                    count += 1
                elif char == ".":
                    count_dot += 1
            if (
                (candidate.__contains__("-") and candidate[0] != "-")
                or count > 1
                or count_dot > 1
            ):
                return False
        for char in candidate:
            if char not in "0123456789.-":
                return False
        return True

    def valid_token_ids(self, vocab: TokenVocab) -> set[int]:
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
            return self._prefix_match_token_ids(vocab, [self.prompt])
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
            forbidden_characters = ['"', "\\", "\n"]
            valid_token_ids = set()
            for token_id, _ in vocab:
                token_text = vocab.token_text(token_id)
                if (
                    not any(
                        char in token_text for char in forbidden_characters
                    )
                    or token_text == '"'
                ):
                    valid_token_ids.add(token_id)
            if not valid_token_ids:
                raise GrammarError("no valid characters found!")
            return valid_token_ids
        elif self.phase == Phase.EXPECT_FREE_NUMBER_VALUE:
            valid_token_ids = set()
            for token_id, _ in vocab:
                token_text = vocab.token_text(token_id)
                if self._is_valid_number_prefix(
                    self.progress + vocab.token_text(token_id)
                ):
                    valid_token_ids.add(token_id)
                elif self.progress != "":
                    if (
                        vocab.token_text(token_id) == "}"
                        and self._all_required_parameters_collected()
                    ):
                        valid_token_ids.add(token_id)
                    elif (
                        vocab.token_text(token_id) == ","
                        and not self._all_required_parameters_collected()
                    ):
                        valid_token_ids.add(token_id)
            if not valid_token_ids:
                raise GrammarError("no valid characters found!")
            return valid_token_ids

    def advance(self, token_id: int, vocab: TokenVocab) -> None:
        if self.phase == Phase.EXPECT_OPEN_BRACE:
            if self.top_level_key == "parameters":
                self.phase = Phase.EXPECT_KEY
                function_def = self.registry.get(self.chosen_function_name)
                param_names = function_def.required_parameter_names()
                self.candidate_menu = param_names
                self.progress = ""
                return
            self.phase = Phase.EXPECT_KEY
            self.candidate_menu = ["prompt", "name", "parameters"]
            self.progress = ""
            return
        elif self.phase == Phase.EXPECT_COLON:
            if self.top_level_key == "prompt":
                self.phase = Phase.EMIT_KNOWN_LITERAL
            elif self.top_level_key == "name":
                self.phase = Phase.EXPECT_ENUM_VALUE
                self.candidate_menu = self.registry.function_names()
                self.progress = ""
            elif self.top_level_key == "parameters":
                if self.current_parameter_name is None:
                    self.phase = Phase.EXPECT_OPEN_BRACE
                else:
                    param_type = self.registry.get(
                        self.chosen_function_name
                    ).parameter_type(self.current_parameter_name)
                    if param_type == "string":
                        self.phase = Phase.EXPECT_FREE_STRING_VALUE
                    elif param_type == "number":
                        self.phase = Phase.EXPECT_FREE_NUMBER_VALUE
                    else:
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
                self.chosen_function_name = self.progress
                self.phase = Phase.EXPECT_COMMA_OR_CLOSE
                self.progress = ""
            return
        elif self.phase == Phase.EMIT_KNOWN_LITERAL:
            self.progress = self.progress + vocab.token_text(token_id)
            if self.progress == '"' + self.prompt + '"':
                self.progress = ""
                self.phase = Phase.EXPECT_COMMA_OR_CLOSE
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
                    for name in self.registry.get(
                        self.chosen_function_name
                    ).required_parameter_names():
                        if name not in self.collected_parameters:
                            next_param = name
                            break
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
            if vocab.token_text(token_id) == '"':
                self.collected_parameters[self.current_parameter_name] = (
                    self.progress
                )
                self.progress = ""
                self.phase = Phase.EXPECT_COMMA_OR_CLOSE
            else:
                self.progress = self.progress + vocab.token_text(token_id)
            return
        elif self.phase == Phase.EXPECT_FREE_NUMBER_VALUE:
            if vocab.token_text(token_id) in ",}":
                if "." in self.progress:
                    self.collected_parameters[self.current_parameter_name] = (
                        float(self.progress)
                    )
                    self.progress = ""
                    self.phase = Phase.EXPECT_COMMA_OR_CLOSE
                else:
                    self.collected_parameters[self.current_parameter_name] = (
                        int(self.progress)
                    )
                    self.progress = ""
                    self.phase = Phase.EXPECT_COMMA_OR_CLOSE
            else:
                self.progress = self.progress + vocab.token_text(token_id)
            return

    def is_done(self) -> bool:
        return self.phase == Phase.DONE

    def result(self) -> dict:
        if self.is_done():
            return {
                "prompt": self.prompt,
                "name": self.chosen_function_name,
                "parameters": self.collected_parameters,
            }
        else:
            raise GrammarError

    @classmethod
    def start(
        cls, prompt: str, registry: FunctionRegistry
    ) -> "GenerationState":
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
        )
