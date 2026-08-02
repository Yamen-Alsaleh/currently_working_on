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
from pydantic import BaseModel
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
    EXPECT_COMMA_OR_CLOSE = 8
    DONE = 9


class GrammarError(Exception):
    pass


class GenerationState(BaseModel):
    prompt: str
    registry: FunctionRegistry
    phase: Phase
    candidate_menu: list[str]
    progress: str
    chosen_function_name: str | None
    collected_parameters: dict[str, int | str]
    key: str  # to knoe if we are filling "name" or "parameters" which it will have as a value

    def valid_token_ids(self, vobac: TokenVocab) -> set[int]:
        pass  # I don't understand what to do I know we should make a if-elif chain to check on the phase then go in but didn't figure what to do for each one

    def advance(self, token_id: int, vocab: TokenVocab) -> None:
        pass  # couldn't figure out what to do

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
