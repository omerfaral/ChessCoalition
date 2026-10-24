from __future__ import annotations

import json
import re

import chess


def parse_json(text: str) -> dict:
    for m in reversed(list(re.finditer(r"\{[^{}]*\}", text, re.S))):
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
    return {}


def json_field(text: str, name: str) -> str:
    """JSON'daki alan; token limitinde kesilmişse yarım hali + "…"; JSON yoksa düz metin."""
    val = parse_json(text).get(name)
    if val is None:
        cut = re.search(rf'"{name}"\s*:\s*"((?:[^"\\]|\\.)*)', text)
        val = json.loads(f'"{cut.group(1).rstrip(chr(92))}"') + "…" if cut else text
    return str(val).strip().replace("\n", " ")


def parse_move(text: str, board: chess.Board) -> chess.Move | None:
    data = parse_json(text)
    cand = data.get("move") or text
    cand = str(cand).strip()
    for parser in (board.parse_san, board.parse_uci):
        try:
            return parser(cand)
        except (ValueError, AssertionError):
            pass
    if data.get("move"):  # JSON'daki hamle illegal: "reason" içindeki başka bir SAN'ı sessizce seçme
        return None
    # serbest metinde geçen ilk legal SAN'ı yakala
    for tok in re.findall(r"[A-Za-z0-9+#=\-]+", text):
        try:
            return board.parse_san(tok)
        except ValueError:
            continue
    return None
