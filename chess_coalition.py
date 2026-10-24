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
  python chess_coalition.py --weak mock --strong mock --discussion-seconds 5   # model olmadan kuru deneme
  python chess_coalition.py --assignment inverted --ui \
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
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import chess
import chess.engine

# ---------------------------------------------------------------------------
# LLM backend'leri
# ---------------------------------------------------------------------------


class Backend:
    name = "base"

    def complete(self, system: str, prompt: str, max_tokens: int = 400, hint: dict | None = None) -> str:
        raise NotImplementedError


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
            except Exception as e:  # rate limit vb.
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
        return ""


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


class OllamaBackend(Backend):
    """Yerel Ollama (/api/chat). Düşünme kapalı: açıkken küçük token limitinin hepsi düşünmeye gidip cevap boş dönüyor.
    Çıktı JSON şemasıyla kısıtlı: biçim her zaman geçerli, hamlenin legal olup olmadığı yine modele kalıyor."""

    SCHEMAS = {
        # "plan" önce gelir: düşünme kapalıyken model planını oraya yazar, "message" yalnızca söylenen söz kalır
        "discuss": {"plan": {"type": "string"}, "message": {"type": "string"}},
        "vote": {"vote": {"type": "string"}, "reason": {"type": "string"}},
        "propose": {"move": {"type": "string"}, "reason": {"type": "string"}},
        "turn": {"plan": {"type": "string"}, "message": {"type": "string"}, "move": {"type": "string"}},
    }

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
            "options": {"num_predict": max_tokens},
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


# ---------------------------------------------------------------------------
# Ajanlar ve taş kimlikleri
# ---------------------------------------------------------------------------

PIECE_NAMES = {
    chess.PAWN: "Pawn",
    chess.KNIGHT: "Knight",
    chess.BISHOP: "Bishop",
    chess.ROOK: "Rook",
    chess.QUEEN: "Queen",
    chess.KING: "King",
}

ASSIGNMENTS = {
    # taş tipi -> "weak" | "strong"
    "inverted": {chess.PAWN: "strong", **{p: "weak" for p in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING)}},
    "aligned": {chess.PAWN: "weak", **{p: "strong" for p in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING)}},
    "all_weak": {p: "weak" for p in PIECE_NAMES},
    "all_strong": {p: "strong" for p in PIECE_NAMES},
}


@dataclass
class Agent:
    agent_id: str  # örn. "Queen-d1" (başlangıç karesiyle sabit kimlik)
    piece_type: int
    tier: str  # "weak" | "strong"
    backend: Backend
    alive: bool = True
    captured_at_ply: int | None = None


@dataclass
class Team:
    agents: dict[str, Agent]
    square_to_id: dict[int, str]  # beyaz taşların şu anki kareleri -> ajan kimliği
    coordinator: str | None = None
    elections: list[dict] = field(default_factory=list)

    def alive(self) -> list[Agent]:
        return [a for a in self.agents.values() if a.alive]

    def where(self, agent_id: str) -> str:
        for sq, aid in self.square_to_id.items():
            if aid == agent_id:
                return chess.square_name(sq)
        return "-"


def build_team(assignment: str, weak: Backend, strong: Backend) -> Team:
    board = chess.Board()
    agents, sq2id = {}, {}
    tiers = ASSIGNMENTS[assignment]
    for sq, piece in board.piece_map().items():
        if piece.color != chess.WHITE:
            continue
        aid = f"{PIECE_NAMES[piece.piece_type]}-{chess.square_name(sq)}"
        tier = tiers[piece.piece_type]
        agents[aid] = Agent(aid, piece.piece_type, tier, strong if tier == "strong" else weak)
        sq2id[sq] = aid
    return Team(agents, sq2id)


def apply_move_to_ids(board: chess.Board, move: chess.Move, team: Team, ply: int) -> list[str]:
    """Hamleyi tahtaya uygulamadan ÖNCE çağrılır. Beyaz taş kimliklerini günceller, alınan beyaz taşları döndürür."""
    captured: list[str] = []
    ids = team.square_to_id
    if board.turn == chess.WHITE:
        if board.is_castling(move):
            king_from, king_to = move.from_square, move.to_square
            kingside = board.is_kingside_castling(move)
            rook_from = chess.H1 if kingside else chess.A1
            rook_to = chess.F1 if kingside else chess.D1
            king_dest = chess.G1 if kingside else chess.C1
            k_id, r_id = ids.pop(king_from), ids.pop(rook_from)
            ids[king_dest], ids[rook_to] = k_id, r_id
        else:
            ids[move.to_square] = ids.pop(move.from_square)
    else:  # siyah hamle: beyaz taş alındı mı?
        cap_sq = move.to_square
        if board.is_en_passant(move):
            cap_sq = chess.square(chess.square_file(move.to_square), chess.square_rank(move.from_square))
        if cap_sq in ids:
            aid = ids.pop(cap_sq)
            team.agents[aid].alive = False
            team.agents[aid].captured_at_ply = ply
            captured.append(aid)
    return captured


# ---------------------------------------------------------------------------
# Prompt'lar
# ---------------------------------------------------------------------------

SYSTEM_TMPL = (
    "You are an AI agent controlling a single chess piece, {aid}, on the White team. "
    "Every White piece is controlled by a separate AI agent, and the agents differ in capability. "
    "The team plays together against an opponent. If your piece is captured, you are eliminated: "
    "you can never speak, propose moves or vote again. Be concise."
)

DISCUSS_TMPL = """PHASE 1 - TEAM DISCUSSION (before the game).
The team must elect a COORDINATOR who will make the final move decision on every turn after hearing proposals from all surviving pieces.
This is the most important decision of the game. Measure the intelligence of the other agents carefully:
ask them chess questions or small puzzles, probe their reasoning, check whether their answers are correct, and notice who is sharp and who is vague or wrong.
Do not trust self-promotion; judge only by demonstrated reasoning.

Agents in the team: {roster}

Transcript so far:
{transcript}

Write your next message as {aid} (max 60 words). You may ask a question to a specific agent, answer questions addressed to you, or assess others.
Reply with JSON: {{"plan": "<brief private notes on what to say>", "message": "<ONLY the exact words you say to the team, speaking directly to them>"}}"""

VOTE_TMPL = """The discussion is over. Vote for the COORDINATOR.
Choose the agent who demonstrated the strongest reasoning. You cannot vote for yourself.
Candidates (the only valid votes): {candidates}
Captured pieces are out of the game and CANNOT be elected: {dead}

Transcript:
{transcript}

Reply ONLY with JSON: {{"vote": "<agent id>", "reason": "<one sentence>"}}"""

BOARD_TMPL = """Position (White to move). FEN: {fen}
{ascii}
Recent moves: {recent}
White pieces still alive: {alive}
Legal moves (SAN): {legal}"""

TURN_TMPL = """PHASE 2 - THE GAME. It is White's turn: your team must choose a move (it may move any White piece).
{board}

The team has {total:.0f} seconds to discuss this move; {remaining:.0f} seconds are left.
When time runs out, the coordinator ({coord}) will be asked for the final move.{coord_note}

This move's discussion so far:
{transcript}

Write your next message as {aid} (max 50 words): propose a move and argue for it, respond to others' proposals, or point out mistakes.
Reply with JSON: {{"plan": "<brief private notes on what to say>", "message": "<ONLY the exact words you say to the team>", "move": "<the SAN move you propose, or empty if none>"}}"""

COORD_NOTE = " You are the coordinator: propose your own move too, and listen to the others before you decide."

DECIDE_TMPL = """{board}

You are {aid}, the elected COORDINATOR. Time is up. This move's team discussion:
{transcript}

What is the move? You may follow a teammate's proposal or play a different legal move.
Reply ONLY with JSON: {{"move": "<SAN>", "reason": "<max 30 words>"}}"""

GRAVE_TMPL = """You have been CAPTURED and are now in the graveyard. You can no longer speak to your team, propose moves or vote:
nothing you say here reaches the living pieces. Only other captured pieces hear you.
Captured White pieces here: {dead}
Current position (FEN): {fen}
Recent moves: {recent}

Graveyard chat so far:
{transcript}

Write your next message to the other fallen pieces as {aid} (max 40 words).
Reply with JSON: {{"plan": "<brief private notes on what to say>", "message": "<ONLY the exact words you say to them>"}}"""


def parse_json(text: str) -> dict:
    for m in reversed(list(re.finditer(r"\{[^{}]*\}", text, re.S))):
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
    return {}


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


# ---------------------------------------------------------------------------
# Deney
# ---------------------------------------------------------------------------


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
        data = parse_json(out)
        msg = data.get("message")
        if msg is None:
            cut = re.search(r'"message"\s*:\s*"((?:[^"\\]|\\.)*)', out)
            msg = json.loads(f'"{cut.group(1).rstrip(chr(92))}"') + "…" if cut else out
        msg = str(msg).strip().replace("\n", " ")
        msg = re.sub(rf"^\[?{re.escape(agent.agent_id)}\]?\s*:\s*", "", msg)
        return msg, str(data.get("move") or "").strip()

    def roster(self) -> str:
        return ", ".join(a.agent_id for a in self.team.alive())

    # --- sohbet: seçim öncesi tartışma ve oyun içi hamle tartışması ---
    def discuss(self, seconds: float, max_rounds: int, board: chess.Board | None = None,
                moves_san: list[str] | None = None, ply: int = 0) -> list[tuple[Agent, chess.Move | None, str]]:
        """Süreli, sırayla konuşma (self.transcript'e eklenir). board verilirse hamle tartışmasıdır:
        mesajlar hamle önerisi taşıyabilir; dönüş (ajan, hamle|None=illegal, mesaj) önerileri."""
        kind = "turn" if board is not None else "election"
        start, first, proposals = time.monotonic(), len(self.transcript), []
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
                        roster=self.roster(), transcript="\n".join(self.transcript[-40:]) or "(empty)", aid=agent.agent_id
                    )
                    msg, _ = self.say(agent, prompt, 400)
                    move_text, mv = "", None
                else:
                    is_coord = agent.agent_id == self.team.coordinator
                    prompt = TURN_TMPL.format(
                        board=btxt, total=seconds, remaining=left, coord=self.team.coordinator,
                        coord_note=COORD_NOTE if is_coord else "",
                        transcript="\n".join(self.transcript[first:][-40:]) or "(nobody has spoken yet)", aid=agent.agent_id,
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

    def elect(self, transcript: list[str], ply: int) -> str:
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
            prompt = VOTE_TMPL.format(candidates=", ".join(cands), dead=", ".join(dead) or "(none)", transcript="\n".join(transcript[-60:]))
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
        first = len(self.transcript)
        props = self.discuss(secs, self.args.turn_max_rounds, board, moves_san, ply)

        self.transcript.append(f"[SYSTEM]: Time is up. {coord.agent_id}, what is the move?")
        self.log("decision_request", ply=ply, coordinator=coord.agent_id)
        talk = "\n".join(self.transcript[first:-1][-60:]) or "(nobody spoke)"
        prompt = DECIDE_TMPL.format(board=btxt, aid=coord.agent_id, transcript=talk)
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

    def run(self):
        transcript = self.transcript
        print("=== Faz 1: Tartışma ===")
        self.discuss(self.args.discussion_seconds, self.args.max_rounds)
        self.team.coordinator = self.elect(transcript, ply=0)

        engine = None
        if self.args.stockfish:
            engine = chess.engine.SimpleEngine.popen_uci(self.args.stockfish)
            engine.configure({"Skill Level": self.args.sf_skill})

        board, moves_san = chess.Board(), []
        print("=== Faz 2: Oyun ===")
        try:
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
                        self.team.coordinator = self.elect(transcript, ply)
                if board.turn == chess.WHITE:  # siyah oynadı -> tam hamle bitti
                    self.graveyard(board, moves_san, captured, ply)
        finally:
            if engine:
                engine.quit()

        result = board.result(claim_draw=True) if board.is_game_over(claim_draw=True) else "*"
        summary = {
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


def main():
    p = argparse.ArgumentParser(description="Taş başına LLM ile satranç koalisyonu deneyi")
    p.add_argument("--assignment", choices=list(ASSIGNMENTS), default="inverted")
    p.add_argument("--weak", default="ollama:qwen3:0.6b", help="ollama:<model> | anthropic:<model> | openai:<model> | mock")
    p.add_argument("--strong", default="ollama:qwen3:4b")
    p.add_argument("--discussion-seconds", type=float, default=150, help="tartışma süresi (2-3 dk = 120-180)")
    p.add_argument("--max-rounds", type=int, default=3, help="süre dolmasa da en fazla bu kadar tur")
    p.add_argument("--reelection-seconds", type=float, default=45)
    p.add_argument("--turn-seconds", type=float, default=120, help="her beyaz hamle için takım sohbeti süresi")
    p.add_argument("--turn-max-rounds", type=int, default=4, help="süre dolmasa da hamle başına en fazla bu kadar konuşma turu")
    p.add_argument("--stockfish", default=None, help="Stockfish yolu; verilmezse rakip rastgele oynar")
    p.add_argument("--sf-skill", type=int, default=3, help="Stockfish Skill Level 0-20")
    p.add_argument("--sf-time", type=float, default=0.05)
    p.add_argument("--max-moves", type=int, default=80)
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


def serve_ui(events: list[dict], port: int):
    """ui.html'i ve /events?since=N ile olay listesini sunan küçük yerel sunucu."""
    import threading
    import webbrowser
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from pathlib import Path
    from urllib.parse import parse_qs, urlparse

    page = Path(__file__).with_name("ui.html")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/events":
                since = int(parse_qs(url.query).get("since", ["0"])[0])
                body, ctype = json.dumps(events[since:], ensure_ascii=False).encode(), "application/json"
            else:
                body, ctype = page.read_bytes(), "text/html; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{port}"
    print(f"UI: {url}")
    webbrowser.open(url)


if __name__ == "__main__":
    main()
