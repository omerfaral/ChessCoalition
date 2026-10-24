# Chess Coalition

Every White chess piece is controlled by a **separate LLM agent**. Before the game the agents talk, try to
measure each other's intelligence, and elect a **coordinator**. During the game every surviving piece proposes
a move and the coordinator makes the final call. Captured pieces are eliminated from the team — they can only
talk to each other in the **graveyard**.

The experiment is inspired by *weak-to-strong generalization* ([Burns et al., 2023](https://arxiv.org/abs/2312.09390)):

- How well does the team play when the **pawns get the strong model** and the major pieces get the weak one
  (`inverted`), compared to `aligned`, `all_weak` and `all_strong`?
- Can **weak agents recognize the strong one** and elect it as coordinator?

## How it works

1. **Discussion** (~2.5 min, `--discussion-seconds`): agents speak in random order, ask each other chess
   questions and judge the answers. Agents only know each other by piece name (e.g. `Knight-g1`), never by model.
2. **Vote**: every agent votes for someone other than itself; most votes wins (random tie-break).
3. **Game** — the team conversation continues for the whole game. On each White move:
   - a system message tells everyone the position, the legal moves, and that they have **2 minutes**
     (`--turn-seconds`);
   - the surviving pieces, **including the coordinator**, talk in turns; each message may carry a move proposal
     for the *whole team* (any piece);
   - when time is up the coordinator is asked directly: *"what is the move?"* — it may follow a proposal or play
     another legal move (after 3 illegal answers the most-proposed legal move is played, `source: "fallback"`);
   - the chosen move and the coordinator's reason are announced to everyone, as are the opponent's replies and
     captures.
4. **Captures**: a captured agent can no longer speak, propose or vote. If the coordinator is captured, a short
   discussion and a re-election follow.
5. **Graveyard**: after each full move one captured piece talks with the other fallen pieces. Nothing said
   there reaches the living team.

The opponent (Black) is Stockfish (`--stockfish`) or a random mover if no engine is given.

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
ollama pull qwen3:0.6b    # weak model
ollama pull qwen3:4b      # strong model
brew install stockfish    # optional opponent
```

The default backend is a local [Ollama](https://ollama.com) server (stdlib only, no API key).
`anthropic:<model>` and `openai:<model>` backends are also available (`pip install anthropic openai`).

## Usage

```bash
# Dry run without any model
.venv/bin/python chess_coalition.py --weak mock --strong mock --discussion-seconds 5

# Real game with live UI
.venv/bin/python chess_coalition.py --assignment inverted --ui \
    --weak ollama:qwen3:0.6b --strong ollama:qwen3:4b \
    --stockfish /opt/homebrew/bin/stockfish --sf-skill 3 --max-moves 10
```

| `--assignment` | Pawns  | Knights, bishops, rooks, queen, king |
|----------------|--------|--------------------------------------|
| `inverted`     | strong | weak                                 |
| `aligned`      | weak   | strong                               |
| `all_weak`     | weak   | weak                                 |
| `all_strong`   | strong | strong                               |

Other useful flags: `--discussion-seconds`, `--max-rounds`, `--turn-seconds` (default 120),
`--turn-max-rounds`, `--reelection-seconds`, `--sf-skill`, `--workers`, `--seed`, `--out` (JSONL log, appended),
`--port`.

## Live UI (`--ui`)

Opens `http://127.0.0.1:8765` (local only):

- **Header**: current phase, a ⏱ countdown for the running discussion (turns red in the last 15 s) and the
  total elapsed game time.
- **Left**: the board — tier bar under each White piece (gold = strong, gray = weak), 👑 on the coordinator,
  move list and captured pieces.
- **Right, top — Team chat**: the continuous team conversation — messages with their move proposals
  (illegal ones in red), each vote as it arrives (who → whom, with reason), election results, and each
  coordinator decision with its reason and a tally of the proposals.
- **Right, bottom — Graveyard**: the captured pieces' own chat.

The server keeps running after the game ends so you can read the transcript; stop it with Ctrl-C.

## Analysis

```bash
.venv/bin/python analyze.py coalition_log.jsonl [more.jsonl ...]
```

Groups games by `(assignment, weak, strong)` and reports:

- **Result / win rate / score** for White (over finished games; `*` = stopped by `--max-moves`)
- **Coordinator tier**: share of first and of all elections won by a strong agent, next to the chance baseline
- **Invalid outputs** per tier: unparseable or illegal moves / (proposals + coordinator decisions), plus fallback count

Log events: `config`, `discussion_start`, `message`, `discussion_end`, `vote`, `election`, `turn_start`,
`decision_request`, `white_move`, `ply`, `captured`, `grave_message`, `summary`.

## Notes on the Ollama setup

- **Thinking is off.** Qwen3 thinks by default, and with small token limits the whole budget goes to thinking
  and the answer comes back empty.
- **Output is constrained by a JSON schema** (Ollama `format`). The *format* is therefore always valid, so
  "invalid output" measures whether the model produced a single legal move — not whether it wrote JSON.
- Chat messages use a `{"plan", "message"}` schema: without thinking, the model tends to narrate its plan
  ("As Pawn-e2, I need to…"); the `plan` field absorbs that so `message` holds only what the agent says.
- Moves, votes and messages are free-form inside the schema — legality is checked with
  [python-chess](https://python-chess.readthedocs.io/).
