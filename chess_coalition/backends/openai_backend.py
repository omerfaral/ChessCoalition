from __future__ import annotations

import time

from .base import Backend


class OpenAICompatBackend(Backend):
    """OpenAI uyumlu herhangi bir uç nokta (OpenAI, Ollama, vLLM...). OPENAI_BASE_URL / OPENAI_API_KEY ortam değişkenleri."""

    def __init__(self, model: str):
        from openai import OpenAI

        self.client = OpenAI()
        self.model = model
        self.name = f"openai:{model}"

    def complete(self, system, prompt, max_tokens=400, hint=None):
        for attempt in range(4):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                )
                return resp.choices[0].message.content or ""
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
        return ""
