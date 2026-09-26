from __future__ import annotations

class GPTTokenizer:
    """Small adapter exposing the tokenizer interface expected by the compressor.

    It uses tiktoken for GPT-side token accounting, removing the old dependency on
    a local Qwen tokenizer path.
    """

    def __init__(self, model: str = "gpt-5"):
        import tiktoken
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except Exception:
            self.encoding = tiktoken.get_encoding("o200k_base")

    def encode(self, text: str, add_special_tokens: bool = False):
        del add_special_tokens
        return self.encoding.encode(str(text), disallowed_special=())

    def count(self, text: str) -> int:
        return len(self.encode(text))
