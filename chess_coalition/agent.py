from __future__ import annotations

from dataclasses import dataclass

from .backends.base import Backend


@dataclass
class Agent:
    agent_id: str  # örn. "Queen-d1" (başlangıç karesiyle sabit kimlik)
    piece_type: int
    tier: str  # "weak" | "strong"
    backend: Backend
    alive: bool = True
    captured_at_ply: int | None = None
    notes: str = ""  # ajanın bir önceki context'inden kendi seçip aktardığı not
    mark: int = 0  # context'i transcript'in bu satırından başlar
