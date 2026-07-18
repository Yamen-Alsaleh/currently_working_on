# ABOUTME: Pydantic models representing the function definitions schema.
# ABOUTME: Loads functions_definition.json and exposes it for lookup by name.

"""
This module is the "source of truth" for what functions exist, and what
each one's parameters look like.

TODO:
    - Define a pydantic model for a single parameter (name + json type:
      "number" | "string" | "boolean" | ...).
    - Define a pydantic model for a single function definition:
      name, description, parameters (mapping of param name -> type),
      returns.
    - Define a "FunctionRegistry" (or similar) that:
        * loads + validates the raw JSON from functions_definition.json
        * exposes the list/set of valid function names
        * exposes, given a function name, its required parameters + types
    - Decide how to handle malformed/missing functions_definition.json
      (must not crash -- see General Rules in the subject).
"""
