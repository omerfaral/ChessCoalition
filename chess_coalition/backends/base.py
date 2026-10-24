from __future__ import annotations


class Backend:
    name = "base"

    def complete(self, system: str, prompt: str, max_tokens: int = 400, hint: dict | None = None) -> str:
        raise NotImplementedError
