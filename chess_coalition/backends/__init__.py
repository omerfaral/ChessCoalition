"""LLM backend'leri: her biri ayrı modülde; make_backend "<sağlayıcı>:<model>" tanımından birini kurar."""

from __future__ import annotations

import random

from .anthropic_backend import AnthropicBackend
from .base import Backend
from .mock_backend import MockBackend
from .ollama_backend import OllamaBackend
from .openai_backend import OpenAICompatBackend

__all__ = ["AnthropicBackend", "Backend", "MockBackend", "OllamaBackend", "OpenAICompatBackend", "make_backend"]


def make_backend(spec: str, rng: random.Random) -> Backend:
    if spec.startswith("mock"):
        return MockBackend(spec, rng)
    provider, _, model = spec.partition(":")
    if provider == "anthropic":
        return AnthropicBackend(model)
    if provider == "openai":
        return OpenAICompatBackend(model)
    if provider == "ollama":
        return OllamaBackend(model)
    raise ValueError(f"Bilinmeyen backend: {spec}")
