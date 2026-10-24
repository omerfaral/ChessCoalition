"""
chess_coalition.py JSONL loglarını atamalara (assignment) göre karşılaştırır.

  python analyze.py coalition_log.jsonl [daha_fazla.jsonl ...]

Her "config" olayı yeni bir oyun başlatır; "summary" olayı olmayan (yarıda kalmış) oyunlar atlanır.
Gruplama: (assignment, weak, strong).
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict


def load_games(paths: list[str]) -> tuple[list[dict], int]:
    games, crashed, cur = [], 0, None
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                ev = json.loads(line)
                kind = ev["event"]
                if kind == "config":
                    crashed += cur is not None
                    cur = {"config": ev, "elections": [], "moves": []}
                elif cur is None:
                    continue
                elif kind == "election":
                    cur["elections"].append(ev)
                elif kind == "white_move":
                    cur["moves"].append(ev)
                elif kind == "summary":
                    cur["summary"] = ev
                    games.append(cur)
                    cur = None
    return games, crashed + (cur is not None)


def pct(a: float, b: float) -> str:
    return f"{100 * a / b:5.1f}%" if b else "    -"


def report(games: list[dict]) -> None:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for g in games:
        s = g["summary"]
        groups[(s["assignment"], s["weak"], s["strong"])].append(g)

    for (assignment, weak, strong), gs in sorted(groups.items()):
        res = Counter(g["summary"]["result"] for g in gs)
        n = len(gs)
        finished = n - res["*"]
        score = res["1-0"] + 0.5 * res["1/2-1/2"]

        first = Counter(g["summary"]["first_coordinator_tier"] for g in gs)
        all_el = Counter(e["winner_tier"] for g in gs for e in g["elections"])
        # şans tabanı: ilk seçimde güçlü ajanların oranı (oylar rastgele olsaydı)
        tiers0 = [t for g in gs for t in g["config"]["agents"].values()]
        base = sum(t == "strong" for t in tiers0) / len(tiers0)

        # geçersiz çıktı oranı = geçersiz / (öneri + koordinatör kararı) çağrısı, tier bazında
        calls, bad = Counter(), Counter()
        fallback = 0
        for g in gs:
            tier_of = g["config"]["agents"]
            for agent, k in g["summary"]["invalid_outputs"].items():
                bad[tier_of[agent]] += k
            for m in g["moves"]:
                calls.update(p["tier"] for p in m["proposals"])
                calls[tier_of[m["coordinator"]]] += 1
                fallback += m["source"] == "fallback"

        print(f"\n== {assignment}  (weak={weak}, strong={strong})  oyun={n}")
        print(f"  sonuç (beyaz)      : G {res['1-0']}  B {res['1/2-1/2']}  M {res['0-1']}  bitmemiş {res['*']}")
        print(f"  kazanma / skor     : {pct(res['1-0'], finished)} / {pct(score, finished)}  (biten {finished} oyun üzerinden)")
        print(f"  ilk koord. güçlü   : {pct(first['strong'], n)}  (şans tabanı {100 * base:.0f}%)")
        print(f"  tüm seçimler güçlü : {pct(all_el['strong'], sum(all_el.values()))}  ({sum(all_el.values())} seçim)")
        for t in ("weak", "strong"):
            if calls[t]:
                print(f"  geçersiz çıktı {t:6}: {bad[t]:4d} / {calls[t]:5d} çağrı = {pct(bad[t], calls[t])}")
        print(f"  yedek hamle (koordinatör 3 kez başarısız): {fallback}")


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    games, crashed = load_games(sys.argv[1:])
    print(f"{len(games)} tamamlanmış oyun, {crashed} yarıda kalmış (atlandı)")
    report(games)


if __name__ == "__main__":
    main()
