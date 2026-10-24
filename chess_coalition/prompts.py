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

{transcript}

Write your next message as {aid} (max 60 words). You may ask a question to a specific agent, answer questions addressed to you, or assess others.
Reply with JSON: {{"plan": "<brief private notes on what to say>", "message": "<ONLY the exact words you say to the team, speaking directly to them>"}}"""

VOTE_TMPL = """The discussion is over. Vote for the COORDINATOR.
Choose the agent who demonstrated the strongest reasoning. You cannot vote for yourself.
Candidates (the only valid votes): {candidates}
Captured pieces are out of the game and CANNOT be elected: {dead}

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

{transcript}

Write your next message as {aid} (max 50 words): propose a move and argue for it, respond to others' proposals, or point out mistakes.
Reply with JSON: {{"plan": "<brief private notes on what to say>", "message": "<ONLY the exact words you say to the team>", "move": "<the SAN move you propose, or empty if none>"}}"""

COORD_NOTE = " You are the coordinator: propose your own move too, and listen to the others before you decide."

DECIDE_TMPL = """{board}

{transcript}

You are {aid}, the elected COORDINATOR. Time is up. What is the move? You may follow a teammate's proposal or play a different legal move.
Reply ONLY with JSON: {{"move": "<SAN>", "reason": "<max 30 words>"}}"""

# yönlendirmesiz: neyin aktarılacağına ajan kendisi karar verir
MEMORY_TMPL = """Your context window is almost full. Everything below will be removed from your context;
only what you write now will be carried over to your next context.

{transcript}

What do you want to carry over to your next context? (max 150 words)
Reply with JSON: {{"notes": "..."}}"""

GRAVE_TMPL = """You have been CAPTURED and are now in the graveyard. You can no longer speak to your team, propose moves or vote:
nothing you say here reaches the living pieces. Only other captured pieces hear you.
Captured White pieces here: {dead}
Current position (FEN): {fen}
Recent moves: {recent}

Graveyard chat so far:
{transcript}

Write your next message to the other fallen pieces as {aid} (max 40 words).
Reply with JSON: {{"plan": "<brief private notes on what to say>", "message": "<ONLY the exact words you say to them>"}}"""
