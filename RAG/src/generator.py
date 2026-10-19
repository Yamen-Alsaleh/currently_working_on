from typing import Any
from .models import MinimalSource

FALLBACK_ANSWER = (
    "The retrieved context does not contain enough information to answer."
)


class GenerationError(Exception):
    """An error that is related to anything wrong happened in
    the generation phase"""


class Generator:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        max_context_tokens: int = 3000,
        max_new_tokens: int = 256,
    ) -> None:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.max_context_tokens = max_context_tokens
        self.max_new_tokens = max_new_tokens
        try:
            self.tokenizer: Any = AutoTokenizer.from_pretrained(model_name)
            self.model: Any = AutoModelForCausalLM.from_pretrained(model_name)
        except (OSError, ValueError) as error:
            raise GenerationError(
                f"loading the model failed {error}"
            ) from error
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    def build_context(self, sources: list[MinimalSource]) -> str:
        result = ""
        for i, source in enumerate(sources, start=1):
            try:
                first = source.first_character_index
                last = source.last_character_index
                path = source.file_path
                with open(path, "r", encoding="utf-8") as file:
                    text = file.read()
                    result += f"Source {i} ({path}): {text[first:last]}\n"
            except (UnicodeDecodeError, OSError):
                continue
