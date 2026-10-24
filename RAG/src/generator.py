from typing import Any
from .models import MinimalSource, MinimalSearchResults, MinimalAnswer
from tqdm import tqdm
import sys

FALLBACK_ANSWER = (
    "The retrieved context does not contain" + " enough information to answer."
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
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        self.max_context_tokens = max_context_tokens
        self.max_new_tokens = max_new_tokens
        try:
            self.tokenizer: Any = AutoTokenizer.from_pretrained(model_name)
            self.model: Any = AutoModelForCausalLM.from_pretrained(model_name)
        except (OSError, ValueError) as error:
            raise GenerationError(
                f"loading the model failed: {error}"
            ) from error
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    def build_context(self, sources: list[MinimalSource]) -> str:
        used_tokens = 0
        result = ""
        for i, source in enumerate(sources, start=1):
            try:
                first = source.first_character_index
                last = source.last_character_index
                path = source.file_path
                with open(path, "r", encoding="utf-8") as file:
                    text = file.read()
                    snippet = f"Source {i} ({path}): {text[first:last]}\n"
                    snippet_tokens = len(self.tokenizer.encode(snippet))
                    if used_tokens + snippet_tokens > self.max_context_tokens:
                        break
                    result += snippet
                    used_tokens += snippet_tokens
            except (UnicodeDecodeError, OSError):
                continue
        return result

    def build_messages(
        self, question: str, context: str
    ) -> list[dict[str, str]]:
        system = (
            "You answer questions about the vLLM codebase. "
            "Use only the context provided. "
            "If the context does not contain the answer, reply exactly: "
            f'"{FALLBACK_ANSWER}" '
            "Do not invent function names, file names or options that "
            "do not appear in the context. "
            "Answer in a few sentences."
        )
        user = f"Context: {context}\n\nQuestion: {question}"
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def generate(self, question: str, sources: list[MinimalSource]) -> str:
        import torch

        context = self.build_context(sources)

        if not context:
            return FALLBACK_ANSWER

        messages = self.build_messages(question, context)
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = inputs.to(self.device)

        try:
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )
        except RuntimeError as error:
            print(f"generation failed: {error}", file=sys.stderr)
            return FALLBACK_ANSWER
        prompt_length = inputs["input_ids"].shape[1]
        new_tokens = output_ids[0][prompt_length:]
        text = self.tokenizer.decode(
            new_tokens, skip_special_tokens=True
        ).strip()
        if "</think>" in text:
            text = text.split("</think>")[-1].strip()
        if not text:
            return FALLBACK_ANSWER
        return text

    def answer_batch(
        self, results: list[MinimalSearchResults]
    ) -> list[MinimalAnswer]:
        ls = []
        for result in tqdm(results, desc="Answering"):
            answer = self.generate(result.question, result.retrieved_sources)
            ls.append(
                MinimalAnswer(
                    question_id=result.question_id,
                    question=result.question,
                    retrieved_sources=result.retrieved_sources,
                    answer=answer,
                )
            )
        return ls
