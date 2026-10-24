from __future__ import annotations

import json
import time

from .base import Backend


class OllamaBackend(Backend):
    """Yerel Ollama (/api/chat). Düşünme kapalı: açıkken küçük token limitinin hepsi düşünmeye gidip cevap boş dönüyor.
    Çıktı JSON şemasıyla kısıtlı: biçim her zaman geçerli, hamlenin legal olup olmadığı yine modele kalıyor."""

    SCHEMAS = {
        # "plan" önce gelir: düşünme kapalıyken model planını oraya yazar, "message" yalnızca söylenen söz kalır
        "discuss": {"plan": {"type": "string"}, "message": {"type": "string"}},
        "vote": {"vote": {"type": "string"}, "reason": {"type": "string"}},
        "propose": {"move": {"type": "string"}, "reason": {"type": "string"}},
        "turn": {"plan": {"type": "string"}, "message": {"type": "string"}, "move": {"type": "string"}},
        "memory": {"notes": {"type": "string"}},
    }
    NUM_CTX = 8192  # açıkça ayarlı: Ollama varsayılanı küçük ve sığmayan prompt'u sessizce baştan keser

    def __init__(self, model: str):
        import os

        self.url = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/") + "/api/chat"
        self.model = model
        self.name = f"ollama:{model}"

    def complete(self, system, prompt, max_tokens=400, hint=None):
        import urllib.request

        task = (hint or {}).get("task")
        props = self.SCHEMAS["propose" if task == "decide" else task]
        body = {
            "model": self.model,
            "stream": False,
            "think": False,
            "format": {"type": "object", "properties": props, "required": list(props)},
            "options": {"num_predict": max_tokens, "num_ctx": self.NUM_CTX},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        }
        req = urllib.request.Request(self.url, json.dumps(body).encode(), {"Content-Type": "application/json"})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=300) as r:
                    text = json.load(r)["message"]["content"]
                break
            except OSError:  # bağlantı / zaman aşımı
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
        return text
