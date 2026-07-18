# ABOUTME: Runs the token-by-token constrained generation loop for a single prompt.
# ABOUTME: Combines the LLM SDK, vocab lookup, and grammar state machine.

"""
TODO:
    - Given a single prompt string, the loaded model, the vocab lookup, and
      the function registry:
        1. Build the initial input ids (encode the prompt + any framing
           text you decide to use).
        2. Initialize a fresh grammar state.
        3. Loop:
             a. get_logits_from_input_ids(current_input_ids)
             b. ask the grammar state for the set of valid token ids
             c. mask all other logits to -inf
             d. pick the highest-scoring surviving token (argmax)
             e. append it to current_input_ids
             f. update the grammar state based on the chosen token
             g. stop when the grammar state reports the JSON object
                is complete
        4. Return the structured result -- likely built directly from the
           grammar state's "decided" values rather than by re-parsing
           generated text, per our design discussion.
    - Consider what "framing text" (if any) precedes the prompt -- and
      whether it needs to be included in the constrained portion or just
      as unconstrained context before decoding starts.
    - Handle errors per-prompt so one bad prompt doesn't crash the whole
      batch run.
"""
