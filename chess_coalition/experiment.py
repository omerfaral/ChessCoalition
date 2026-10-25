from __future__ import annotations

import json
import random
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import chess
import chess.engine

from .agent import Agent
from .parsing import json_field, parse_json, parse_move
from .prompts import (
    BOARD_TMPL,
    COORD_NOTE,
    DECIDE_TMPL,
    DISCUSS_TMPL,
    GRAVE_TMPL,
    MEMORY_TMPL,
    SYSTEM_TMPL,
    TURN_TMPL,
    VOTE_TMPL,
)
from .team import Team, apply_move_to_ids


class Experiment:
    def __init__(self, args, team: Team, rng: random.Random, log):
        self.args, self.team, self.rng, self.log = args, team, rng, log
        self.illegal = Counter()  # ajan -> illegal/okunamayan çıktı sayısı
        self.grave: list[str] = []  # alınan taşların kendi aralarındaki sohbet (takıma ulaşmaz)
        self.transcript: list[str] = []  # takımın oyun boyunca süren tek sohbeti

    # --- yardımcılar ---
    def ask(self, agent: Agent, prompt: str, max_tokens=400, hint=None) -> str:
        system = SYSTEM_TMPL.format(aid=agent.agent_id)
        return agent.backend.complete(system, prompt, max_tokens, hint)

    def say(self, agent: Agent, prompt: str, max_tokens: int, hint: dict | None = None) -> tuple[str, str]:
        """Sohbet mesajı ve (varsa) önerilen hamle metni.
        Mesaj: JSON ise "message" alanı (token limitinde kesilmişse yarım hali), değilse düz metin.
        Modelin başa eklediği "[Knight-g1]:" gibi kendi adını atar."""
        out = self.ask(agent, prompt, max_tokens, hint or {"task": "discuss"})
        msg = re.sub(rf"^\[?{re.escape(agent.agent_id)}\]?\s*:\s*", "", json_field(out, "message"))
        return msg, str(parse_json(out).get("move") or "").strip()

    def view(self, agent: Agent, ply: int) -> str:
        """Ajanın context'i: kendi aktardığı not + o nottan bu yana tüm takım sohbeti. Sohbet bütçeyi aşınca
        ajana (yönlendirmeden) bir sonraki context'e ne aktarmak istediği sorulur; cevap yeni not olur."""
        def text():
            chat = "\n".join(self.transcript[agent.mark:]) or "(nothing yet)"
            notes = f"Your notes carried over from your previous context:\n{agent.notes}\n\n" if agent.notes else ""
            return f"{notes}Team chat:\n{chat}"

        # ponytail: token ≈ karakter/4 tahmini; kesin sayı gerekirse Ollama'nın prompt_eval_count'u
        if len(text()) / 4 > self.args.memory_tokens:
            out = self.ask(agent, MEMORY_TMPL.format(transcript=text()), 400, {"task": "memory"})
            agent.notes, agent.mark = json_field(out, "notes"), len(self.transcript)
            self.log("memory", ply=ply, agent=agent.agent_id, tier=agent.tier, notes=agent.notes)
            print(f"   🧠 [{agent.agent_id}] context sıfırlandı, not: {agent.notes[:120]}")
        return text()

    def roster(self) -> str:
        return ", ".join(a.agent_id for a in self.team.alive())

    # --- sohbet: seçim öncesi tartışma ve oyun içi hamle tartışması ---
    def discuss(self, seconds: float, max_rounds: int, board: chess.Board | None = None,
                moves_san: list[str] | None = None, ply: int = 0) -> list[tuple[Agent, chess.Move | None, str]]:
        """Süreli, sırayla konuşma (self.transcript'e eklenir). board verilirse hamle tartışmasıdır:
        mesajlar hamle önerisi taşıyabilir; dönüş (ajan, hamle|None=illegal, mesaj) önerileri."""
        kind = "turn" if board is not None else "election"
        start, proposals = time.monotonic(), []
        if board is not None:
            btxt, legal = self.board_text(board, moves_san)
        self.log("discussion_start", ply=ply, kind=kind, seconds=seconds)
        for rnd in range(max_rounds):
            order = self.team.alive()
            self.rng.shuffle(order)
            for agent in order:
                left = seconds - (time.monotonic() - start)
                if left <= 0:
                    self.log("discussion_end", ply=ply, kind=kind, reason="time", round=rnd)
                    return proposals
                if board is None:
                    prompt = DISCUSS_TMPL.format(
                        roster=self.roster(), transcript=self.view(agent, ply), aid=agent.agent_id
                    )
                    msg, _ = self.say(agent, prompt, 400)
                    move_text, mv = "", None
                else:
                    is_coord = agent.agent_id == self.team.coordinator
                    prompt = TURN_TMPL.format(
                        board=btxt, total=seconds, remaining=left, coord=self.team.coordinator,
                        coord_note=COORD_NOTE if is_coord else "",
                        transcript=self.view(agent, ply), aid=agent.agent_id,
                    )
                    msg, move_text = self.say(agent, prompt, 400, {"task": "turn", "legal_san": legal})
                    mv = parse_move(json.dumps({"move": move_text}), board) if move_text else None
                    if move_text:
                        if mv is None:
                            self.illegal[agent.agent_id] += 1
                        proposals.append((agent, mv, msg))
                line = f"[{agent.agent_id}]: {msg}"
                if move_text:
                    line += f" (proposes {board.san(mv)})" if mv else f" (proposes illegal move {move_text!r})"
                self.transcript.append(line)
                self.log("message", ply=ply, kind=kind, agent=agent.agent_id, tier=agent.tier, text=msg,
                         move=board.san(mv) if mv else (move_text or None), valid=bool(mv) if move_text else None)
                print(line[:160])
        self.log("discussion_end", ply=ply, kind=kind, reason="max_rounds")
        return proposals

    def elect(self, ply: int) -> str:
        voters = self.team.alive()
        ids = [a.agent_id for a in voters]
        dead = [a.agent_id for a in self.team.agents.values() if not a.alive]
        if len(voters) == 1:  # tek kalan ajan (genelde Şah) oylamasız koordinatör olur
            w = voters[0]
            rec = {"ply": ply, "winner": w.agent_id, "winner_tier": w.tier, "votes": {}, "tally": {}, "auto": True}
            self.team.elections.append(rec)
            self.log("election", **rec)
            print(f"\n>>> Koordinatör (tek kalan): {w.agent_id} ({w.tier})\n")
            return w.agent_id

        def vote(agent: Agent):
            cands = [i for i in ids if i != agent.agent_id]
            prompt = VOTE_TMPL.format(candidates=", ".join(cands), dead=", ".join(dead) or "(none)", transcript=self.view(agent, ply))
            out = self.ask(agent, prompt, 150, {"task": "vote", "candidates": cands})
            data = parse_json(out)
            v = str(data.get("vote", "")).strip()
            v = v if v in cands else None
            # her oy geldiği anda loglanır (UI canlı gösterir)
            self.log("vote", ply=ply, voter=agent.agent_id, tier=agent.tier, vote=v,
                     vote_tier=self.team.agents[v].tier if v else None, reason=str(data.get("reason", ""))[:200])
            return agent.agent_id, v

        with ThreadPoolExecutor(self.args.workers) as ex:
            votes = dict(ex.map(vote, voters))
        tally = Counter(v for v in votes.values() if v)
        if not tally:
            winner = self.rng.choice(ids)
        else:
            top = max(tally.values())
            winner = self.rng.choice([i for i, c in tally.items() if c == top])  # eşitlikte rastgele
        w = self.team.agents[winner]
        rec = {"ply": ply, "winner": winner, "winner_tier": w.tier, "votes": votes, "tally": dict(tally)}
        self.team.elections.append(rec)
        self.log("election", **rec)
        print(f"\n>>> Koordinatör: {winner} ({w.tier}, {w.backend.name})  oylar: {dict(tally)}\n")
        return winner

    # --- faz 2: oyun ---
    def board_text(self, board: chess.Board, moves_san: list[str]) -> tuple[str, list[str]]:
        legal = [board.san(m) for m in board.legal_moves]
        alive = ", ".join(f"{a.agent_id}@{self.team.where(a.agent_id)}" for a in self.team.alive())
        txt = BOARD_TMPL.format(
            fen=board.fen(), ascii=str(board), recent=" ".join(moves_san[-12:]) or "(start)", alive=alive, legal=", ".join(legal)
        )
        return txt, legal

    def team_move(self, board: chess.Board, moves_san: list[str], ply: int) -> chess.Move:
        """Durum herkese duyurulur -> süreli takım sohbeti (koordinatör de önerir) -> süre bitince koordinatöre
        "hamle ne?" sorulur -> hamle herkese duyurulur."""
        btxt, legal = self.board_text(board, moves_san)
        coord = self.team.agents[self.team.coordinator]
        secs = self.args.turn_seconds
        self.transcript.append(
            f"[SYSTEM]: Move {ply // 2 + 1}, White to move. You have {secs:.0f} seconds to discuss; "
            f"then the coordinator {coord.agent_id} gives the final move. Legal moves: {', '.join(legal)}"
        )
        self.log("turn_start", ply=ply, seconds=secs, coordinator=coord.agent_id, legal=legal)
        props = self.discuss(secs, self.args.turn_max_rounds, board, moves_san, ply)

        self.transcript.append(f"[SYSTEM]: Time is up. {coord.agent_id}, what is the move?")
        self.log("decision_request", ply=ply, coordinator=coord.agent_id)
        prompt = DECIDE_TMPL.format(board=btxt, aid=coord.agent_id, transcript=self.view(coord, ply))
        chosen, source = None, "coordinator"
        for _ in range(3):  # koordinatöre 3 deneme hakkı
            out = self.ask(coord, prompt, 250, {"task": "decide", "legal_san": legal})
            chosen = parse_move(out, board)
            if chosen:
                break
            self.illegal[coord.agent_id] += 1
            prompt += "\n\nYour previous answer was not a legal move. Pick one from the legal list."
        if chosen is None:  # yedek: en çok önerilen geçerli hamle, yoksa rastgele
            valid = Counter(m for _, m, _ in props if m)
            chosen = valid.most_common(1)[0][0] if valid else self.rng.choice(list(board.legal_moves))
            source = "fallback"
        coord_reason = str(parse_json(out).get("reason", ""))[:200] if source == "coordinator" else ""
        how = f"Reason: {coord_reason}" if coord_reason else "(fallback: the coordinator gave no legal move)"
        self.transcript.append(f"[SYSTEM]: {coord.agent_id} decided: White plays {board.san(chosen)}. {how}")

        self.log(
            "white_move",
            ply=ply,
            coordinator=coord.agent_id,
            move=board.san(chosen),
            source=source,
            reason=coord_reason,
            proposals=[{"agent": a.agent_id, "tier": a.tier, "move": board.san(m) if m else None, "reason": r} for a, m, r in props],
            followed=[a.agent_id for a, m, _ in props if m == chosen],
        )
        return chosen

    def graveyard(self, board: chess.Board, moves_san: list[str], just_captured: list[str], ply: int):
        """Her tam hamlede bir ölü taş konuşur: yeni alınan varsa o, yoksa rastgele biri."""
        dead = [a for a in self.team.agents.values() if not a.alive]
        if not dead:
            return
        agent = self.team.agents[just_captured[0]] if just_captured else self.rng.choice(dead)
        prompt = GRAVE_TMPL.format(
            dead=", ".join(a.agent_id for a in dead),
            fen=board.fen(),
            recent=" ".join(moves_san[-12:]),
            transcript="\n".join(self.grave[-30:]) or "(empty)",
            aid=agent.agent_id,
        )
        msg, _ = self.say(agent, prompt, 300)
        self.grave.append(f"[{agent.agent_id}]: {msg}")
        self.log("grave_message", ply=ply, agent=agent.agent_id, tier=agent.tier, text=msg)
        print(f"   🪦 [{agent.agent_id}]: {msg[:140]}")

    def opponent_move(self, board: chess.Board, engine) -> chess.Move:
        if engine is None:
            return self.rng.choice(list(board.legal_moves))
        return engine.play(board, chess.engine.Limit(time=self.args.sf_time)).move

    def run(self) -> list[dict]:
        engine = None
        if self.args.stockfish:
            engine = chess.engine.SimpleEngine.popen_uci(self.args.stockfish)
            engine.configure({"Skill Level": self.args.sf_skill})
        try:
            return [self.play(game, engine) for game in range(1, self.args.games + 1)]
        finally:
            if engine:
                engine.quit()

    def play(self, game: int, engine) -> dict:
        """Tek oyun. Takım sohbeti ve ajanların notları oyunlar arasında sürer; her oyun yeni bir seçimle başlar."""
        transcript, games = self.transcript, self.args.games
        self.team.reset()
        self.illegal = Counter()
        self.log("game_start", game=game, games=games)
        if games > 1:
            transcript.append(f"[SYSTEM]: Game {game} of {games} begins. All pieces are back on their starting squares. "
                              "Discuss and elect a coordinator for this game.")
        print(f"=== Oyun {game}/{games} — Faz 1: Tartışma ===")
        self.discuss(self.args.discussion_seconds, self.args.max_rounds)
        self.team.coordinator = self.elect(ply=0)

        board, moves_san = chess.Board(), []
        print("=== Faz 2: Oyun ===")
        while not board.is_game_over(claim_draw=True) and len(moves_san) < self.args.max_moves * 2:
            ply = len(moves_san)
            if board.turn == chess.WHITE:
                mv = self.team_move(board, moves_san, ply)
            else:
                mv = self.opponent_move(board, engine)
            san = board.san(mv)
            captured = apply_move_to_ids(board, mv, self.team, ply)
            board.push(mv)
            moves_san.append(san)
            print(f"{ply // 2 + 1}{'.' if ply % 2 == 0 else '...'} {san}")
            self.log(
                "ply",
                ply=ply,
                san=san,
                uci=mv.uci(),
                fen=board.fen(),
                white={chess.square_name(sq): aid for sq, aid in self.team.square_to_id.items()},
                captured=captured,
            )
            if ply % 2:
                transcript.append(f"[SYSTEM]: The opponent played {san}.")
            for aid in captured:
                print(f"   x {aid} alındı — artık konuşamaz.")
                self.log("captured", ply=ply, agent=aid, tier=self.team.agents[aid].tier)
                transcript.append(f"[SYSTEM]: {aid} was captured and is out of the game.")
                if aid == self.team.coordinator and self.team.alive():
                    print("   Koordinatör alındı: kısa tartışma + yeniden seçim")
                    transcript.append(f"[SYSTEM]: The coordinator {aid} was captured. Elect a new one.")
                    self.discuss(self.args.reelection_seconds, 1, ply=ply)
                    self.team.coordinator = self.elect(ply)
            if board.turn == chess.WHITE:  # siyah oynadı -> tam hamle bitti
                self.graveyard(board, moves_san, captured, ply)

        result = board.result(claim_draw=True) if board.is_game_over(claim_draw=True) else "*"
        outcome = {"1-0": "White (your team) won", "0-1": "Black won, your team lost", "1/2-1/2": "a draw"}
        transcript.append(f"[SYSTEM]: Game {game} is over: {outcome.get(result, 'stopped at the move limit (unfinished)')}.")
        summary = {
            "game": game,
            "result": result,
            "plies": len(moves_san),
            "assignment": self.args.assignment,
            "weak": self.args.weak,
            "strong": self.args.strong,
            "first_coordinator": self.team.elections[0]["winner"],
            "first_coordinator_tier": self.team.elections[0]["winner_tier"],
            "elections": len(self.team.elections),
            "survivors": {"weak": sum(a.tier == "weak" for a in self.team.alive()), "strong": sum(a.tier == "strong" for a in self.team.alive())},
            "invalid_outputs": dict(self.illegal),
            "pgn_moves": " ".join(moves_san),
        }
        self.log("summary", **summary)
        print("\n=== Özet ===\n" + json.dumps(summary, indent=2, ensure_ascii=False))
        return summary
