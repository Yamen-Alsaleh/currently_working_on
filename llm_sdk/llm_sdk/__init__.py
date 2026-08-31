# ABOUTME: LLM SDK for local inference using Hugging Face transformers.
# ABOUTME: Provides Small_LLM_Model for loading and running causal models.

import torch
from huggingface_hub import hf_hub_download
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    logging,
)


logging.set_verbosity_error()


class Small_LLM_Model:
    """Wrap a lightweight Hugging Face causal language model.

    Parameters
    ----------
    model_name: str, default="Qwen/Qwen3-0.6B"
        Identifier of the model on the Hugging Face Hub.
    device: str | None, default=None
        Computation device. If None, automatically select MPS when available
        on macOS, CUDA when available, or otherwise use the CPU.
    dtype: torch.dtype | None, default=None
        Numerical precision. GPU and MPS default to float16 to reduce memory
        use. CPU defaults to float32 for compatibility.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        *,
        device: str | None = None,
        dtype: torch.dtype | None = None,
        trust_remote_code: bool = True,
    ) -> None:
        self._model_name = model_name

        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self._device = device

        if dtype is None:
            dtype = (
                torch.float16
                if self._device in ["cuda", "mps"]
                else torch.float32
            )
        self._dtype = dtype

        self._tokenizer: PreTrainedTokenizer = (
            AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code,
            )
        )
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        self._model: PreTrainedModel = (
            AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=self._dtype,
                device_map="auto" if self._device == "cuda" else None,
                trust_remote_code=trust_remote_code,
            )
        )
        self._model.to(self._device)
        self._model.eval()

        for parameter in self._model.parameters():
            parameter.requires_grad = False

    def encode(self, text: str) -> torch.Tensor:
        """Tokenize text into a two-dimensional tensor on the target device."""
        ids = self._tokenizer.encode(text, add_special_tokens=False)
        return torch.tensor([ids], device=self._device, dtype=torch.long)

    def decode(self, ids: torch.Tensor | list[int]) -> str:
        """Decode token IDs into text while removing special tokens."""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return self._tokenizer.decode(ids, skip_special_tokens=True)

    def get_logits_from_input_ids(
        self,
        input_ids: list[int],
    ) -> list[float]:
        """Return raw next-token logits for a list of input token IDs."""
        input_tensor = torch.tensor(
            [input_ids],
            device=self._device,
            dtype=torch.long,
        )
        with torch.no_grad():
            output = self._model(input_ids=input_tensor)
        logits = output.logits[0, -1].tolist()
        return [float(value) for value in logits]

    def get_path_to_vocab_file(self) -> str:
        """Return the local path to the model vocabulary file."""
        filename = self._tokenizer.vocab_files_names.get(
            "vocab_file",
            "vocab.json",
        )
        return hf_hub_download(
            repo_id=self._model_name,
            filename=filename,
        )

    def get_path_to_merges_file(self) -> str:
        """Return the local path to the tokenizer merges file."""
        filename = self._tokenizer.vocab_files_names.get(
            "merges_file",
            "merges.txt",
        )
        return hf_hub_download(
            repo_id=self._model_name,
            filename=filename,
        )

    def get_path_to_tokenizer_file(self) -> str:
        """Return the local path to the tokenizer JSON file."""
        filename = self._tokenizer.vocab_files_names.get(
            "tokenizer_file",
            "tokenizer.json",
        )
        return hf_hub_download(
            repo_id=self._model_name,
            filename=filename,
        )
