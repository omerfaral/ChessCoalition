from __future__ import annotations

import json
import random

from .base import Backend


class MockBackend(Backend):
    """API olmadan akışı test etmek için: rastgele mesaj, oy ve hamle üretir."""

    def __init__(self, label: str = "mock", rng: random.Random | None = None):
        self.name = label
        self.rng = rng or random.Random()

    def complete(self, system, prompt, max_tokens=400, hint=None):
        hint = hint or {}
        task = hint.get("task")
        if task == "discuss":
            return f"Mock message {self.rng.randint(0, 999)}: what is the best reply to 1.e4?"
        if task == "vote":
            return json.dumps({"vote": self.rng.choice(hint["candidates"]), "reason": "mock"})
        if task == "memory":
            return json.dumps({"notes": f"Mock notes {self.rng.randint(0, 999)}"})
        if task == "turn":  # bazen öneri yok, ara sıra illegal öneri
            r = self.rng.random()
            move = "" if r < 0.3 else "Zz9" if r < 0.35 else self.rng.choice(hint["legal_san"])
            return json.dumps({"message": f"Mock talk {self.rng.randint(0, 999)}", "move": move})
        if task in ("propose", "decide"):
            moves = hint["legal_san"]
            # ara sıra illegal hamle üreterek doğrulama/yeniden deneme yolunu da test et
            move = "Zz9" if self.rng.random() < 0.05 else self.rng.choice(moves)
            return json.dumps({"move": move, "reason": "mock"})
        return ""
