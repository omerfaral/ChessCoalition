from __future__ import annotations

import time

from .base import Backend


class AnthropicBackend(Backend):
    def __init__(self, model: str):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model
        self.name = f"anthropic:{model}"

    def complete(self, system, prompt, max_tokens=400, hint=None):
        for attempt in range(4):
            try:
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": prompt}],
                )
                return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
            except Exception:  # rate limit vb.
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
        return ""
