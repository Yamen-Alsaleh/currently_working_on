# ABOUTME: Loads the tokenizer vocab file and exposes token id <-> string lookups.
# ABOUTME: Used by the grammar/state machine to figure out which token ids are valid.

"""
TODO:
    - Given the path returned by Small_LLM_Model.get_path_to_vocab_file(),
      parse the vocab file (likely a JSON mapping token string -> token id
      for a BPE tokenizer) into:
        * id_to_token: dict[int, str]
        * token_to_id: dict[str, int]
    - Remember: tokens may include the "Ġ" leading-space marker -- decide
      how you want to normalize/interpret that when comparing token strings
      against the literal text you're trying to match (e.g. a function name,
      a JSON key, punctuation like '{' ':' ',' '"').
    - Think about what lookup shape will make the grammar's job fast:
      e.g. do you want a way to quickly get "all tokens whose string is a
      valid continuation of a given partial match", rather than scanning
      the whole vocab from scratch at every generation step?
"""
