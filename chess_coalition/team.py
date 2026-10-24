from __future__ import annotations

from dataclasses import dataclass, field

import chess

from .agent import Agent
from .backends.base import Backend

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
