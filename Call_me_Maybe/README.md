*This activity has been created as part of the 42 curriculum by yalsaleh.*

# call me maybe

## Description

call me maybe translates natural-language requests into structured function
calls. It uses Qwen3-0.6B to select a function and extract its arguments, while
a schema-aware constrained decoder guarantees the structure of every result.
The program writes one JSON object per input prompt with exactly three keys:
prompt, name, and parameters.

The project demonstrates that a small language model can produce dependable,
machine-readable output when token generation is restricted by a grammar
derived from the supplied function definitions.

## Requirements

- Python 3.10 or later
- uv
- Enough disk space to download Qwen/Qwen3-0.6B on the first run
- Internet access on the first run, unless the model is already cached

An NVIDIA GPU is optional. The SDK automatically uses CUDA when available and
otherwise runs on the CPU.

## Instructions

Install the exact locked dependencies:

    uv sync

Run with the default paths:

    uv run python -m src

The default files are:

- Function definitions: data/input/functions_definition.json
- Prompts: data/input/function_calling_tests.json
- Results: data/output/function_calling_results.json

Use custom paths when needed:

    uv run python -m src \
      --functions_definition path/to/functions.json \
      --input path/to/prompts.json \
      --output path/to/results.json

The output parent directory is created automatically. Invalid paths, malformed
JSON, schema violations, and generation failures produce a clear error message
and a non-zero exit code instead of an unhandled traceback.

The Makefile provides the mandatory commands:

    make install
    make run
    make debug
    make clean
    make lint
    make lint-strict

It also provides:

    make test
    make integration

The integration target loads the real model and evaluates all supplied prompts,
so it is intentionally slower than the unit-test target.

## Input and output formats

The prompt file must contain a JSON array of objects:

    [
      {"prompt": "What is the sum of 2 and 3?"},
      {"prompt": "Greet Ada"}
    ]

The function-definition file must contain a JSON array. Every function requires
a name, description, parameters object, and return schema. Supported parameter
types are number, integer, string, and boolean. A parameter may also declare an
`enum` array; generation is then restricted to exactly those predefined values.
Blank or whitespace-only prompts are rejected with a clear error instead of
being forced into an unrelated function.

Example result:

    [
      {
        "prompt": "What is the sum of 2 and 3?",
        "name": "fn_add_numbers",
        "parameters": {
          "a": 2,
          "b": 3
        }
      }
    ]

## Algorithm explanation

### 1. Validate the inputs

Function definitions are parsed into Pydantic models. The registry rejects
missing or extra fields, unsupported primitive types, invalid enum values,
duplicate function names, an empty function list, and malformed JSON. The
prompt file is separately validated as an array containing only objects with a
non-blank string prompt field.

### 2. Build the model context

The generator gives the model concise instructions, the currently available
function signatures and descriptions, and a few extraction examples. The
current request is encoded with JSON escaping so nested quotes remain
unambiguous.

### 3. Maintain a generation state

GenerationState is a Pydantic model representing a finite-state machine. Its
phase records what may legally appear next, including:

- the opening or closing object brace;
- one exact top-level key;
- a colon or comma;
- one registered function name;
- one required parameter name;
- a string opening quote and string contents;
- a JSON number, integer, boolean, or schema-defined enum value.

At each step, valid_token_ids examines the tokenizer vocabulary and returns only
token IDs that keep the partial output valid. Exact names and literals use
prefix matching, so a token is allowed only when its decoded text remains a
prefix of at least one valid candidate.

### 4. Select from constrained logits

When a semantic choice exists, the SDK returns logits for the next token. The
generator takes the highest-logit token from the grammar-approved set. The
function name is therefore selected by the LLM, not by keyword rules or a
hardcoded lookup.

String values permit only tokens that keep a valid JSON string possible.
Escaped quotes, escaped backslashes, Windows paths, and template braces are
supported. Number prefixes are checked character by character, integer fields
cannot contain decimals, boolean generation is limited to true or false, and
enum fields can only produce declared values. Parameter names and types come
from the function selected by the model.

For extractive string arguments, the grammar discovers general boundaries in
the request: quoted spans, the suffix after a colon, and Unix or Windows file
paths. The LLM must begin the value. Once its generated prefix identifies one
unique prompt span, the decoder emits the remaining exact span with no further
semantic choice and then forces the closing quote. Leading whitespace and
tokens that cross a detected boundary are masked unless that whitespace was
actually part of the prompt span. This both prevents value overrun and avoids
unnecessary full-model calls for already-known text.

Numeric arguments use the same principle. The grammar collects numeric
literals from the request without forbidding other schema-valid numbers. After
an LLM-generated prefix identifies one unique request number, the remaining
digits are emitted directly. Long decimals therefore require the semantic
choice and boundary decision rather than one full-model call per token.

Function selection also contains a constrained no-match sentinel. The LLM may
select it when none of the supplied definitions can satisfy a non-blank
request; the CLI then reports a clear generation error and writes no invented
function call. Before generation, meaningful words from the request are
compared with words from registered function names, descriptions, and parameter
names. When there is a direct relationship, the synthetic sentinel is omitted
and the LLM chooses only among real registered functions. When there is no
relationship, the sentinel remains available with a small confidence margin.
This relevance check never chooses the function itself; function selection is
still made from constrained LLM logits.

Completed extracted strings receive a closing-quote preference. This prevents
the small model from extending a correct value with repetitive prose while
still allowing continuation when its logit is substantially stronger. The
preference changes only when a string stops; it does not choose its content or
the function.

Regex character classes have one additional grammar boundary. After a bare
class such as `[aeiou]`, only a closing quote or a single-character quantifier
is allowed. After a quantified class such as `[0-9]+`, only the closing quote
is legal. Tokens that would cross that boundary and append an alternative are
masked out, preventing invalid continuations such as `[0-9]+|([0-9]+`.

For regex parameters, a final behavior-based safeguard handles a narrow small-
model mistake: emitting an unquoted alphabetic character list instead of a
character class. The value is wrapped as a class only when the request asks for
all or every occurrence, the literal pattern does not match the source, its
character-class form does match, and the prompt did not request a quoted word,
phrase, or substring. This rule is independent of any specific character list.

### 5. Collect and serialize

The state machine collects typed Python values and reaches DONE only after every
required parameter is present. Before output, a Pydantic model and the selected
function schema verify the exact top-level keys, required parameter set, value
types, and enum membership. The CLI serializes the results with the standard
JSON library, which guarantees valid escaping and prevents trailing commas,
non-finite numbers, or extra prose.

## Design decisions

- Pydantic validates every project-defined data model and provides precise
  schema errors.
- Top-level keys and required parameter keys are emitted in a deterministic
  order. JSON does not require that order, but a fixed order makes the grammar
  smaller and easier to verify.
- The decoder queries the LLM only when a semantic choice remains. Forced
  punctuation, known keys, and the copied prompt are emitted directly from the
  grammar. This produces the same token text while avoiding unnecessary full
  model passes, which is particularly important on CPU-only workstations.
- Output is collected in memory before writing, so a generation failure cannot
  leave a partially successful result file.
- Plain character-list regexes are normalized only after behavioral checks;
  literal, quoted, word, phrase, and already-valid matching patterns are kept.
- The source uses package-relative imports so the required python -m src command
  works without setting PYTHONPATH.
- Empty prompt arrays produce an empty JSON array without loading the model.
- Empty individual prompts are invalid and never reach function selection.
- Functions with no parameters are supported and produce an empty parameters
  object.

## Performance analysis

The supplied 11-prompt integration batch achieved 11/11 semantically correct
results in 14.42 seconds on an RTX 4060 laptop before the deterministic-token
optimization was added. All generated results were valid JSON and matched the
required function schemas.

CPU inference is substantially slower because the provided public SDK performs
a full model forward pass for each requested logit. The generator therefore
skips model inference for all grammar-forced tokens. Only function selection and
free number, string, or boolean values require logits. This reduces work without
using private SDK attributes or changing semantic selection.

Memory usage is dominated by Qwen3-0.6B and its tokenizer. The grammar keeps a
small amount of state per prompt. Vocabulary indexes and the safe-string token
set are built once and reused for every prompt.

## Challenges faced

The first string implementation entered free-string generation before emitting
an opening quote. The model consequently continued values as prose. Adding a
dedicated opening-quote phase made string boundaries structurally explicit.

Regex values then began correctly but continued with irrelevant alternatives.
Clearer extraction examples and completion-aware stopping fixed the overrun
without hardcoding the generated regex.

On CPU, the model sometimes understood a character category but emitted the
plain list without regex brackets. Prompt wording alone did not reliably change
that close-logit decision. A general applicability check now distinguishes an
unmatched character list from a literal pattern and constructs the usable
character-class form without changing function selection.

Another issue was test quality: the original integration script considered any
non-crashing generation successful. It now checks function names and parameter
values. Regex cases are validated through their substitution behavior, allowing
equivalent patterns such as lowercase-only and case-inclusive vowel classes.

Finally, the first test suite depended on one developer's absolute Hugging Face
cache path. Portable unit tests now use synthetic vocabularies; only the
explicit integration target loads the real model vocabulary.

## Testing strategy

Run fast, portable unit tests:

    make test

They cover:

- valid and invalid function-definition files;
- malformed, missing, and incorrectly shaped prompt files;
- vocabulary loading, duplicate IDs, Qwen whitespace markers, and lookup;
- every grammar value type and state transition;
- integer, float, enum, escaped quote, backslash, and template string handling;
- blank-prompt and no-matching-function rejection;
- closing-quote selection behavior;
- plain character-list normalization and literal-pattern preservation;
- deterministic structural selection without bypassing LLM function choice;
- output-directory creation and JSON serialization;
- graceful CLI errors without tracebacks.

Run static checks:

    make lint

Run the real Qwen acceptance batch:

    make integration

Finally, exercise the exact reviewer command and validate its file:

    uv run python -m src
    python -m json.tool data/output/function_calling_results.json

For the supplied moulinette version, the grading command exposed by its CLI is
`grade_student_answers`:

    cd moulinette
    uv sync
    uv run python -m moulinette prepare_exercises --set private
    cd ..
    uv run python -m src \
      --functions_definition moulinette/data/input/functions_definition.json \
      --input moulinette/data/input/function_calling_tests.json \
      --output data/output/function_calling_results.json
    cd moulinette
    uv run python -m moulinette grade_student_answers \
      --set private \
      --student_answer_path ../data/output/function_calling_results.json

## Resources

- Python json documentation:
  https://docs.python.org/3/library/json.html
- Python argparse documentation:
  https://docs.python.org/3/library/argparse.html
- Python typing documentation:
  https://docs.python.org/3/library/typing.html
- Pydantic documentation:
  https://docs.pydantic.dev/
- Qwen3 model card:
  https://huggingface.co/Qwen/Qwen3-0.6B
- Hugging Face tokenizers documentation:
  https://huggingface.co/docs/tokenizers/
- JSON specification, RFC 8259:
  https://www.rfc-editor.org/rfc/rfc8259
- mypy documentation:
  https://mypy.readthedocs.io/
- flake8 documentation:
  https://flake8.pycqa.org/

AI tools were used as a reviewed development aid for explaining tokenizer and
constrained-decoding concepts, identifying the missing string-opening state,
suggesting error cases, reviewing CLI completeness against the subject, and
drafting parts of the tests and documentation. Every suggestion was inspected,
adapted to the project constraints, and validated with unit tests and real-model
outputs. The function-selection mechanism and final behavior were checked to
ensure they still depend on LLM logits rather than heuristic routing.
