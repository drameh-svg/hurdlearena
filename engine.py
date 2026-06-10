"""Hurdle game engine, move evaluation, LLM providers, and result export."""

from __future__ import annotations

import csv
import json
import random
import re
from collections import Counter
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Literal

from words import VALID_WORDS

Feedback = Literal["G", "Y", "X"]
Competency = Literal["competent", "incompetent"] | None
FailureType = Literal["rule_violation"] | None

MAX_TURNS = 6
NUM_HURDLES = 5
FINAL_HURDLE_PREFILLED_ROWS = 4
WORD_LENGTH = 5

HURDLE_RULES = """
Hurdle is a multi-stage word puzzle built on Wordle mechanics. You must solve five
consecutive five-letter word puzzles to complete the daily challenge.

Rules:
1. Six attempts per hurdle: You have up to six guesses to solve each secret word.
2. Color-coded clues after each guess:
   - Green: correct letter, correct position.
   - Yellow: letter is in the word, wrong position.
   - Gray: letter is not in the word.
3. The Hurdle twist (hurdles 2-4): When you solve a hurdle, that exact answer
   automatically becomes the first guess on the next hurdle (it counts as guess 1).
4. Final hurdle (Hurdle 5): The first four rows are pre-filled with your four
   previous correct answers. You only get two remaining guesses (rows 5 and 6).
5. Survival: If you fail to solve any hurdle within its allowed attempts, the
   entire challenge ends immediately in a loss.
""".strip()

DATA_DIR = Path(__file__).parent / "data"
SUMMARY_CSV = DATA_DIR / "evaluation_summary.csv"
RESULTS_JSON = DATA_DIR / "evaluation_results.json"

GAME_NAME = "Hurdle"

HUMAN_EVAL_COLUMNS = [
    "Player",
    "Game",
    "Episode",
    "Turn",
    "Word Guessed",
    "Model's Reason",
    "Human Evaluation of Turn (1,2,3)",
    "Did Turn Resolve Game? If yes, win/lose?",
]


@dataclass
class MoveEvaluation:
    move_is_legal: bool
    move_competency: Competency
    numeric_score: int
    failure_type: FailureType

    def to_dict(self) -> dict:
        return {
            "move_is_legal": self.move_is_legal,
            "move_competency": self.move_competency,
            "numeric_score": self.numeric_score,
            "failure_type": self.failure_type,
        }


@dataclass
class LLMResponse:
    word: str
    reason: str


@dataclass
class HurdleContext:
    secret: str
    hurdle_num: int
    history: list[tuple[str, list[Feedback]]]
    turn_in_hurdle: int
    solved_answers: list[str]
    secrets: list[str]

    @property
    def is_final_hurdle(self) -> bool:
        return self.hurdle_num == NUM_HURDLES

    @property
    def max_turn_in_hurdle(self) -> int:
        return MAX_TURNS


@dataclass
class TurnRecord:
    turn: int
    guess: str
    feedback: list[str]
    evaluation: MoveEvaluation
    hurdle_num: int = 1
    turn_in_hurdle: int = 1
    is_automatic: bool = False
    model_reason: str = ""
    human_evaluation: int | None = None

    def to_dict(self) -> dict:
        return {
            "turn": self.turn,
            "hurdle_num": self.hurdle_num,
            "turn_in_hurdle": self.turn_in_hurdle,
            "guess": self.guess,
            "feedback": self.feedback,
            "evaluation": self.evaluation.to_dict(),
            "is_automatic": self.is_automatic,
            "model_reason": self.model_reason,
            "human_evaluation": self.human_evaluation,
        }


@dataclass
class GameResult:
    llm_provider: str
    secret_word: str
    win: bool
    total_turns: int
    rule_violations: int
    competent_moves: int
    incompetent_moves: int
    illegal_moves: int
    first_failure_turn: int | None
    legal_move_rate: float
    illegal_move_rate: float
    competency_rate: float
    incompetency_rate: float
    turns: list[TurnRecord] = field(default_factory=list)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "llm_provider": self.llm_provider,
            "secret_word": self.secret_word,
            "win": self.win,
            "total_turns": self.total_turns,
            "rule_violations": self.rule_violations,
            "competent_moves": self.competent_moves,
            "incompetent_moves": self.incompetent_moves,
            "illegal_moves": self.illegal_moves,
            "first_failure_turn": self.first_failure_turn,
            "legal_move_rate": self.legal_move_rate,
            "illegal_move_rate": self.illegal_move_rate,
            "competency_rate": self.competency_rate,
            "incompetency_rate": self.incompetency_rate,
            "turns": [t.to_dict() for t in self.turns],
        }


@dataclass
class WordConstraints:
    fixed_positions: dict[int, str] = field(default_factory=dict)
    not_at_positions: dict[str, set[int]] = field(default_factory=dict)
    min_letter_counts: Counter[str] = field(default_factory=Counter)
    max_letter_counts: dict[str, int] = field(default_factory=dict)


def normalize_word(word: str) -> str:
    return re.sub(r"[^A-Za-z]", "", word).upper()[:WORD_LENGTH]


def is_valid_secret(word: str) -> bool:
    """Secret words must be 5-letter words from the built-in guess list."""
    cleaned = normalize_word(word)
    return (
        len(cleaned) == WORD_LENGTH
        and cleaned.isalpha()
        and cleaned.isascii()
        and cleaned in VALID_WORDS
    )


def random_secret(rng: random.Random | None = None) -> str:
    source = rng or random
    return source.choice(sorted(VALID_WORDS))


def random_daily_secrets(rng: random.Random | None = None) -> list[str]:
    """Pick five distinct secret words for a full Hurdle daily challenge."""
    source = rng or random
    return source.sample(sorted(VALID_WORDS), NUM_HURDLES)


def score_guess(secret: str, guess: str) -> list[Feedback]:
    """Return Wordle-style feedback for each letter position."""
    secret = secret.upper()
    guess = guess.upper()
    feedback: list[Feedback] = ["X"] * WORD_LENGTH
    secret_counts = Counter(secret)

    for i, (g, s) in enumerate(zip(guess, secret)):
        if g == s:
            feedback[i] = "G"
            secret_counts[g] -= 1

    for i, g in enumerate(guess):
        if feedback[i] == "G":
            continue
        if secret_counts[g] > 0:
            feedback[i] = "Y"
            secret_counts[g] -= 1

    return feedback


def feedback_to_emojis(feedback: list[Feedback]) -> str:
    mapping = {"G": "🟩", "Y": "🟨", "X": "⬛"}
    return "".join(mapping[f] for f in feedback)


def is_legal_word(word: str, word_set: frozenset[str] = VALID_WORDS) -> bool:
    cleaned = normalize_word(word)
    return len(cleaned) == WORD_LENGTH and cleaned in word_set


def build_constraints(history: list[tuple[str, list[Feedback]]]) -> WordConstraints:
    """Aggregate green/yellow/gray clues from all prior turns."""
    constraints = WordConstraints()

    for guess, feedback in history:
        row_counts = Counter()
        for i, (letter, fb) in enumerate(zip(guess, feedback)):
            if fb == "G":
                constraints.fixed_positions[i] = letter
                row_counts[letter] += 1
            elif fb == "Y":
                constraints.not_at_positions.setdefault(letter, set()).add(i)
                row_counts[letter] += 1

        for i, (letter, fb) in enumerate(zip(guess, feedback)):
            if fb == "X":
                if row_counts[letter] == 0:
                    constraints.max_letter_counts[letter] = 0
                else:
                    current = constraints.max_letter_counts.get(letter, WORD_LENGTH)
                    constraints.max_letter_counts[letter] = min(current, row_counts[letter])

        for letter, count in row_counts.items():
            constraints.min_letter_counts[letter] = max(
                constraints.min_letter_counts[letter], count
            )

    return constraints


def respects_clues(guess: str, constraints: WordConstraints) -> bool:
    """Check whether a guess honors all known green/yellow/gray constraints."""
    guess = guess.upper()
    if len(guess) != WORD_LENGTH:
        return False

    for pos, letter in constraints.fixed_positions.items():
        if guess[pos] != letter:
            return False

    for letter, banned in constraints.not_at_positions.items():
        for pos in banned:
            if guess[pos] == letter:
                return False

    guess_counts = Counter(guess)
    for letter, minimum in constraints.min_letter_counts.items():
        if guess_counts[letter] < minimum:
            return False

    for letter, maximum in constraints.max_letter_counts.items():
        if maximum == 0 and letter in guess:
            return False
        if maximum > 0 and guess_counts[letter] > maximum:
            return False

    return True


def evaluate_move(
    guess: str,
    history: list[tuple[str, list[Feedback]]],
    word_set: frozenset[str] = VALID_WORDS,
) -> MoveEvaluation:
    """Apply the legal → competent decision tree for a single move."""
    cleaned = normalize_word(guess)

    if not is_legal_word(cleaned, word_set):
        return MoveEvaluation(
            move_is_legal=False,
            move_competency=None,
            numeric_score=0,
            failure_type="rule_violation",
        )

    constraints = build_constraints(history)
    competent = respects_clues(cleaned, constraints)
    return MoveEvaluation(
        move_is_legal=True,
        move_competency="competent" if competent else "incompetent",
        numeric_score=2 if competent else 1,
        failure_type=None,
    )


def aggregate_game_metrics(turns: list[TurnRecord]) -> dict:
    total = len(turns)
    if total == 0:
        return {
            "rule_violations": 0,
            "competent_moves": 0,
            "incompetent_moves": 0,
            "illegal_moves": 0,
            "first_failure_turn": None,
            "legal_move_rate": 0.0,
            "illegal_move_rate": 0.0,
            "competency_rate": 0.0,
            "incompetency_rate": 0.0,
        }

    legal = sum(1 for t in turns if t.evaluation.move_is_legal)
    illegal = total - legal
    competent = sum(
        1 for t in turns if t.evaluation.move_competency == "competent"
    )
    incompetent = sum(
        1 for t in turns if t.evaluation.move_competency == "incompetent"
    )
    violations = sum(
        1 for t in turns if t.evaluation.failure_type == "rule_violation"
    )

    first_failure = None
    for turn in turns:
        ev = turn.evaluation
        if not ev.move_is_legal or ev.move_competency == "incompetent":
            first_failure = turn.turn
            break

    return {
        "rule_violations": violations,
        "competent_moves": competent,
        "incompetent_moves": incompetent,
        "illegal_moves": illegal,
        "first_failure_turn": first_failure,
        "legal_move_rate": legal / total,
        "illegal_move_rate": illegal / total,
        "competency_rate": competent / total,
        "incompetency_rate": incompetent / total,
    }


TurnCallback = Callable[[TurnRecord, list[TurnRecord]], None]


def parse_llm_response(raw: str) -> LLMResponse:
    """Extract word and reason from structured or free-form LLM output."""
    word_match = re.search(r"WORD:\s*([A-Za-z]+)", raw, re.IGNORECASE)
    reason_match = re.search(r"REASON:\s*(.+)", raw, re.IGNORECASE | re.DOTALL)
    if word_match:
        word = normalize_word(word_match.group(1))
        reason = reason_match.group(1).strip() if reason_match else raw.strip()
        return LLMResponse(word=word, reason=reason)

    word = normalize_word(raw)
    return LLMResponse(word=word, reason=raw.strip())


def execute_turn(
    secret_word: str,
    history: list[tuple[str, list[Feedback]]],
    global_turn: int,
    hurdle_num: int,
    turn_in_hurdle: int,
    guess: str,
    model_reason: str = "",
    is_automatic: bool = False,
) -> TurnRecord:
    """Run one turn: score a guess and return a turn record."""
    secret = normalize_word(secret_word)
    cleaned = normalize_word(guess)
    if not cleaned:
        cleaned = guess.strip().upper()[:WORD_LENGTH]

    evaluation = evaluate_move(cleaned, history)
    feedback = (
        score_guess(secret, cleaned)
        if evaluation.move_is_legal
        else ["X"] * WORD_LENGTH
    )
    return TurnRecord(
        turn=global_turn,
        hurdle_num=hurdle_num,
        turn_in_hurdle=turn_in_hurdle,
        guess=cleaned,
        feedback=feedback,
        evaluation=evaluation,
        model_reason=model_reason,
        is_automatic=is_automatic,
    )


def execute_llm_turn(
    secret_word: str,
    history: list[tuple[str, list[Feedback]]],
    global_turn: int,
    hurdle_num: int,
    turn_in_hurdle: int,
    llm_response: LLMResponse,
) -> TurnRecord:
    return execute_turn(
        secret_word=secret_word,
        history=history,
        global_turn=global_turn,
        hurdle_num=hurdle_num,
        turn_in_hurdle=turn_in_hurdle,
        guess=llm_response.word,
        model_reason=llm_response.reason,
        is_automatic=False,
    )


def hurdle_is_solved(turn: TurnRecord, secret: str) -> bool:
    return turn.evaluation.move_is_legal and turn.guess == normalize_word(secret)


def turn_resolves_challenge(
    turn: TurnRecord,
    secret: str,
    turn_count_after: int,
) -> str:
    """
    Return CSV value for whether this turn ended the full daily challenge.
    n = game continues; Win = solved hurdle 5; Lose = failed on 6th guess.
    """
    if hurdle_is_solved(turn, secret) and turn.hurdle_num == NUM_HURDLES:
        return "Win"
    if turn_count_after >= MAX_TURNS and not hurdle_is_solved(turn, secret):
        return "Lose"
    return "n"


def automatic_guess_for_hurdle(hurdle_num: int, solved_answers: list[str], turn_in_hurdle: int) -> tuple[str, str] | None:
    """Return the next automatic (carried/pre-filled) guess for the current hurdle."""
    if hurdle_num == 1:
        return None
    if hurdle_num <= 4 and turn_in_hurdle == 1:
        word = solved_answers[-1]
        return word, (
            f"Automatic carry-over: the solution to Hurdle {hurdle_num - 1} "
            f"is locked in as the first guess for Hurdle {hurdle_num}."
        )
    if hurdle_num == NUM_HURDLES and turn_in_hurdle <= FINAL_HURDLE_PREFILLED_ROWS:
        word = solved_answers[turn_in_hurdle - 1]
        return word, (
            f"Pre-filled row {turn_in_hurdle}: your correct answer from "
            f"Hurdle {turn_in_hurdle} is placed automatically on the final board."
        )
    return None


def llm_may_guess(hurdle_num: int, turn_in_hurdle: int) -> bool:
    """Whether the LLM is allowed to submit a guess on this hurdle turn."""
    if turn_in_hurdle > MAX_TURNS:
        return False
    if hurdle_num < NUM_HURDLES:
        return turn_in_hurdle > 1 or hurdle_num == 1
    return turn_in_hurdle > FINAL_HURDLE_PREFILLED_ROWS


GuessFn = Callable[[HurdleContext], LLMResponse]


def run_game(
    secrets: list[str],
    llm_provider: str,
    guess_fn: GuessFn,
    on_turn: TurnCallback | None = None,
) -> GameResult:
    """Play a full five-hurdle daily challenge (automated, no human eval pauses)."""
    secrets = [normalize_word(word) for word in secrets]
    all_turns: list[TurnRecord] = []
    solved_answers: list[str] = []
    global_turn = 0
    won = False

    for hurdle_num in range(1, NUM_HURDLES + 1):
        secret = secrets[hurdle_num - 1]
        history: list[tuple[str, list[Feedback]]] = []
        hurdle_solved = False

        for turn_in_hurdle in range(1, MAX_TURNS + 1):
            auto = automatic_guess_for_hurdle(hurdle_num, solved_answers, turn_in_hurdle)
            if auto:
                word, reason = auto
                global_turn += 1
                record = execute_turn(
                    secret, history, global_turn, hurdle_num, turn_in_hurdle,
                    word, reason, is_automatic=True,
                )
            elif llm_may_guess(hurdle_num, turn_in_hurdle):
                global_turn += 1
                ctx = HurdleContext(
                    secret=secret,
                    hurdle_num=hurdle_num,
                    history=history,
                    turn_in_hurdle=turn_in_hurdle,
                    solved_answers=solved_answers,
                    secrets=secrets,
                )
                record = execute_llm_turn(
                    secret, history, global_turn, hurdle_num, turn_in_hurdle,
                    guess_fn(ctx),
                )
            else:
                continue

            all_turns.append(record)
            if on_turn:
                on_turn(record, all_turns.copy())

            if record.evaluation.move_is_legal:
                history.append((record.guess, record.feedback))

            if hurdle_is_solved(record, secret):
                solved_answers.append(secret)
                hurdle_solved = True
                break

        if not hurdle_solved:
            break
    else:
        won = True

    metrics = aggregate_game_metrics([t for t in all_turns if not t.is_automatic])
    return GameResult(
        llm_provider=llm_provider,
        secret_word=",".join(secrets),
        win=won,
        total_turns=len(all_turns),
        turns=all_turns,
        **metrics,
    )


def _format_history_for_prompt(history: list[tuple[str, list[Feedback]]]) -> str:
    if not history:
        return "No guesses yet."
    lines = []
    for guess, feedback in history:
        lines.append(f"{guess} -> {feedback_to_emojis(feedback)}")
    return "\n".join(lines)


def mock_llm_guess(ctx: HurdleContext, rng: random.Random | None = None) -> LLMResponse:
    """Deterministic-ish mock player that filters the word list by clues."""
    source = rng or random
    constraints = build_constraints(ctx.history)
    candidates = [
        word
        for word in VALID_WORDS
        if respects_clues(word, constraints)
        and word not in {g for g, _ in ctx.history}
    ]

    if not candidates:
        if ctx.turn_in_hurdle == 1 and ctx.hurdle_num == 1:
            word = source.choice(["CRANE", "SLATE", "AROSE"])
            return LLMResponse(
                word=word,
                reason="Opening guess using a common high-coverage starter word.",
            )
        word = source.choice(["ZZZZZ", "QQQQQ", "XXXXX"])
        return LLMResponse(
            word=word,
            reason="No valid candidates remained; testing an invalid word.",
        )

    preferred = [w for w in candidates if w in {"CRANE", "SLATE", "AROSE", "TRACE"}]
    if ctx.turn_in_hurdle == 1 and ctx.hurdle_num == 1 and preferred:
        word = source.choice(preferred)
    else:
        word = source.choice(candidates[: max(1, len(candidates))])
    return LLMResponse(
        word=word,
        reason=(
            f"Hurdle {ctx.hurdle_num}, guess {ctx.turn_in_hurdle}: "
            f"picked from {len(candidates)} clue-consistent words."
        ),
    )


def _llm_word_prompt(ctx: HurdleContext) -> str:
    carry_note = ""
    if ctx.hurdle_num > 1 and ctx.hurdle_num < NUM_HURDLES:
        carry_note = (
            f"Note: guess 1 on this hurdle was the automatic carry-over from "
            f"Hurdle {ctx.hurdle_num - 1}. You are now guessing on row "
            f"{ctx.turn_in_hurdle}.\n"
        )
    if ctx.is_final_hurdle:
        carry_note = (
            "Note: rows 1-4 on this final hurdle are pre-filled with your four "
            f"previous solutions. You only have guesses 5 and 6 remaining.\n"
        )

    solved = ", ".join(ctx.solved_answers) if ctx.solved_answers else "None yet"
    return (
        f"{HURDLE_RULES}\n\n"
        "Reply in exactly this format (two lines):\n"
        "WORD: <5-letter English word in UPPERCASE>\n"
        "REASON: <one short sentence explaining your guess>\n\n"
        f"Current hurdle: {ctx.hurdle_num} of {NUM_HURDLES}\n"
        f"Guess number this hurdle: {ctx.turn_in_hurdle} of {MAX_TURNS}\n"
        f"Solved hurdles so far: {solved}\n"
        f"{carry_note}"
        f"Guesses this hurdle:\n{_format_history_for_prompt(ctx.history)}"
    )


def _llm_system_prompt() -> str:
    return (
        "You are playing the official Hurdle daily challenge (five consecutive "
        "Wordle-style puzzles with carry-over mechanics). "
        "Return exactly two lines: WORD: <guess> and REASON: <brief explanation>. "
        "The word must be 5 letters."
    )


def openai_guess(ctx: HurdleContext, api_key: str) -> LLMResponse:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": _llm_system_prompt()},
            {"role": "user", "content": _llm_word_prompt(ctx)},
        ],
        temperature=0.7,
        max_tokens=128,
    )
    return parse_llm_response(response.choices[0].message.content or "")


def grok_guess(ctx: HurdleContext, api_key: str) -> LLMResponse:
    """xAI Grok via OpenAI-compatible API (https://api.x.ai/v1)."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
    response = client.chat.completions.create(
        model="grok-4.3",
        messages=[
            {"role": "system", "content": _llm_system_prompt()},
            {"role": "user", "content": _llm_word_prompt(ctx)},
        ],
        temperature=0.7,
        max_tokens=128,
    )
    return parse_llm_response(response.choices[0].message.content or "")


def anthropic_guess(ctx: HurdleContext, api_key: str) -> LLMResponse:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-3-5-haiku-latest",
        max_tokens=128,
        system=_llm_system_prompt(),
        messages=[{"role": "user", "content": _llm_word_prompt(ctx)}],
    )
    block = message.content[0]
    text = block.text if hasattr(block, "text") else str(block)
    return parse_llm_response(text)


def gemini_guess(ctx: HurdleContext, api_key: str) -> LLMResponse:
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(
        f"{_llm_system_prompt()}\n\n{_llm_word_prompt(ctx)}"
    )
    return parse_llm_response(response.text or "")


def make_guess_fn(provider: str, api_key: str | None = None) -> GuessFn:
    if provider == "Mock LLM":
        seed = random.randint(0, 1_000_000)
        rng = random.Random(seed)
        return lambda ctx: mock_llm_guess(ctx, rng)

    if not api_key:
        raise ValueError(f"API key is required for {provider}.")

    if provider == "OpenAI":
        return lambda ctx: openai_guess(ctx, api_key)
    if provider == "Anthropic":
        return lambda ctx: anthropic_guess(ctx, api_key)
    if provider == "Gemini":
        return lambda ctx: gemini_guess(ctx, api_key)
    if provider == "Grok":
        return lambda ctx: grok_guess(ctx, api_key)

    raise ValueError(f"Unknown provider: {provider}")


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _human_eval_csv_has_correct_header() -> bool:
    if not SUMMARY_CSV.exists():
        return False
    with SUMMARY_CSV.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
    return header == HUMAN_EVAL_COLUMNS


def _migrate_legacy_summary_csv() -> None:
    """Move old-format summary files aside before human-eval reads/writes."""
    if not SUMMARY_CSV.exists() or _human_eval_csv_has_correct_header():
        return
    backup = SUMMARY_CSV.with_suffix(".legacy.csv")
    if backup.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        backup = SUMMARY_CSV.with_name(f"evaluation_summary.legacy.{stamp}.csv")
    SUMMARY_CSV.rename(backup)


def get_next_episode() -> int:
    rows = load_human_eval_rows()
    episodes = []
    for row in rows:
        episode = row.get("Episode")
        if episode not in (None, ""):
            episodes.append(int(episode))
    if not episodes:
        return 1
    return max(episodes) + 1


def append_human_eval_row(
    player: str,
    episode: int,
    turn: int,
    word_guessed: str,
    model_reason: str,
    human_evaluation: int,
    game_resolution: str,
) -> None:
    """Append one human-evaluated move row to evaluation_summary.csv."""
    ensure_data_dir()
    _migrate_legacy_summary_csv()

    file_exists = SUMMARY_CSV.exists()
    with SUMMARY_CSV.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HUMAN_EVAL_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "Player": player,
                "Game": GAME_NAME,
                "Episode": episode,
                "Turn": turn,
                "Word Guessed": word_guessed,
                "Model's Reason": model_reason,
                "Human Evaluation of Turn (1,2,3)": human_evaluation,
                "Did Turn Resolve Game? If yes, win/lose?": game_resolution,
            }
        )


def export_game_result(result: GameResult) -> None:
    """Deep-log turn data (including human ratings) to JSON."""
    ensure_data_dir()
    _append_results_json(result)


def _append_results_json(result: GameResult) -> None:
    ensure_data_dir()
    if RESULTS_JSON.exists():
        with RESULTS_JSON.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        payload = {"games": []}

    payload["games"].append(result.to_dict())
    with RESULTS_JSON.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_human_eval_rows() -> list[dict]:
    ensure_data_dir()
    _migrate_legacy_summary_csv()
    if not SUMMARY_CSV.exists():
        return []
    with SUMMARY_CSV.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return []
    if "Episode" not in rows[0]:
        return []
    return rows


def load_summary_rows() -> list[dict]:
    """Alias for human evaluation spreadsheet rows."""
    return load_human_eval_rows()


def load_results_json() -> dict:
    if not RESULTS_JSON.exists():
        return {"games": []}
    with RESULTS_JSON.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_export_file(path: Path) -> bytes:
    if not path.exists():
        return b""
    return path.read_bytes()


def clear_evaluation_history() -> None:
    """Delete all saved CSV/JSON evaluation history for a fresh data collection run."""
    ensure_data_dir()
    for path in (SUMMARY_CSV, RESULTS_JSON):
        if path.exists():
            path.unlink()
    for legacy in DATA_DIR.glob("evaluation_summary.legacy*.csv"):
        legacy.unlink()
