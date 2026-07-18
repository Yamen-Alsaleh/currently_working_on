# ABOUTME: State machine that tracks position within the JSON structure being
# ABOUTME: generated, and computes the set of valid next token ids at each step.

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
