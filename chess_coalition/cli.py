"""
Satranç koalisyonu deneyi
=========================
Beyaz takımdaki her taşı ayrı bir LLM ajanı kontrol eder.

Akış:
  1. Tartışma (varsayılan ~2.5 dk): ajanlar sırayla konuşur, birbirlerinin
     zekasını ölçmeye çalışır. Ajanlar birbirlerinin hangi model olduğunu
     bilmez; yalnızca taş adıyla tanınırlar.
  2. Oylama: herkes kendisi dışında birine oy verir, en çok oyu alan koordinatör olur.
  3. Oyun: takım sohbeti oyun boyunca sürer. Sıra beyazdayken durum ve legal hamleler herkese duyurulur,
     takım (koordinatör dahil) 2 dk konuşup hamle önerir; süre bitince koordinatöre "hamle ne?" sorulur
     ve seçilen hamle herkese duyurulur.
  - Yenilen (alınan) taşın ajanı bir daha konuşamaz ve oy veremez.
  - Koordinatör alınırsa kısa bir tartışma + yeniden seçim yapılır.

Kullanım örnekleri:
  pip install chess          # Ollama backend'i yalnızca stdlib kullanır
  ollama pull qwen3:0.6b && ollama pull qwen3:4b
  python -m chess_coalition --weak mock --strong mock --discussion-seconds 5   # model olmadan kuru deneme
  python -m chess_coalition --assignment inverted --ui \
      --weak ollama:qwen3:0.6b --strong ollama:qwen3:4b \
      --stockfish /opt/homebrew/bin/stockfish --sf-skill 3
  # --ui: tarayıcıda solda tahta, sağda takım sohbeti ve mezarlık (alınan taşlar) sohbeti

Atamalar (--assignment):
  inverted   : piyonlar -> güçlü model, diğer taşlar (Ş, V, K, F, A) -> zayıf model
  aligned    : piyonlar -> zayıf model, diğer taşlar -> güçlü model
  all_weak   : hepsi zayıf
  all_strong : hepsi güçlü
"""

from __future__ import annotations

import argparse
import json
import random
import time

from .backends import make_backend
from .experiment import Experiment
from .team import ASSIGNMENTS, build_team
from .ui.server import serve_ui


def main():
    p = argparse.ArgumentParser(prog="python -m chess_coalition", description="Taş başına LLM ile satranç koalisyonu deneyi")
    p.add_argument("--assignment", choices=list(ASSIGNMENTS), default="inverted")
    p.add_argument("--weak", default="ollama:qwen3:0.6b", help="ollama:<model> | anthropic:<model> | openai:<model> | mock")
    p.add_argument("--strong", default="ollama:qwen3:4b")
    p.add_argument("--discussion-seconds", type=float, default=150, help="tartışma süresi (2-3 dk = 120-180)")
    p.add_argument("--max-rounds", type=int, default=3, help="süre dolmasa da en fazla bu kadar tur")
    p.add_argument("--reelection-seconds", type=float, default=45)
    p.add_argument("--turn-seconds", type=float, default=120, help="her beyaz hamle için takım sohbeti süresi")
    p.add_argument("--turn-max-rounds", type=int, default=4, help="süre dolmasa da hamle başına en fazla bu kadar konuşma turu")
    p.add_argument("--memory-tokens", type=int, default=4000,
                   help="ajanın context'indeki sohbet bunu aşınca ona ne aktarmak istediği sorulur ve context sıfırlanır")
    p.add_argument("--stockfish", default=None, help="Stockfish yolu; verilmezse rakip rastgele oynar")
    p.add_argument("--sf-skill", type=int, default=3, help="Stockfish Skill Level 0-20")
    p.add_argument("--sf-time", type=float, default=0.05)
    p.add_argument("--max-moves", type=int, default=80)
    p.add_argument("--games", type=int, default=1, help="art arda oyun sayısı; sohbet ve notlar oyunlar arasında sürer")
    p.add_argument("--workers", type=int, default=8, help="paralel API çağrısı")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="coalition_log.jsonl")
    p.add_argument("--ui", action="store_true", help="tarayıcıda canlı izleme (solda tahta, sağda sohbetler)")
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()

    rng = random.Random(args.seed)
    weak, strong = make_backend(args.weak, rng), make_backend(args.strong, rng)
    team = build_team(args.assignment, weak, strong)

    fh = open(args.out, "a", encoding="utf-8")
    events: list[dict] = []  # bu oyunun olayları; UI buradan okur

    def log(event, **kw):
        ev = {"event": event, "t": time.time(), **kw}
        events.append(ev)
        fh.write(json.dumps(ev, ensure_ascii=False) + "\n")
        fh.flush()

    if args.ui:
        serve_ui(events, args.port)
    log("config", **{k: v for k, v in vars(args).items()}, agents={a.agent_id: a.tier for a in team.agents.values()})
    try:
        Experiment(args, team, rng, log).run()
    finally:
        fh.close()
    if args.ui:
        print(f"Oyun bitti; UI açık kalıyor (http://127.0.0.1:{args.port}). Çıkmak için Ctrl-C.")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass
