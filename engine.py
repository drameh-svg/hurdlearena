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
WORD_LENGTH = 5

DATA_DIR = Path(__file__).parent / "data"
SUMMARY_CSV = DATA_DIR / "evaluation_summary.csv"
RESULTS_JSON = DATA_DIR / "evaluation_results.json"

SUMMARY_COLUMNS = [
    "timestamp",
    "llm_provider",
    "secret_word",
    "win",
    "total_turns",
    "rule_violations",
    "competent_moves",
    "incompetent_moves",
    "illegal_moves",
    "first_failure_turn",
    "legal_move_rate",
    "illegal_move_rate",
    "competency_rate",
    "incompetency_rate",
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
class TurnRecord:
    turn: int
    guess: str
    feedback: list[str]
    evaluation: MoveEvaluation

    def to_dict(self) -> dict:
        return {
            "turn": self.turn,
            "guess": self.guess,
            "feedback": self.feedback,
            "evaluation": self.evaluation.to_dict(),
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
    """Secret words may be any 5-letter alphabetic string."""
    cleaned = normalize_word(word)
    return len(cleaned) == WORD_LENGTH and cleaned.isalpha() and cleaned.isascii()


def random_secret(rng: random.Random | None = None) -> str:
    source = rng or random
    return source.choice(sorted(VALID_WORDS))


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


def run_game(
    secret_word: str,
    llm_provider: str,
    guess_fn: Callable[[str, list[tuple[str, list[Feedback]]], int], str],
    on_turn: TurnCallback | None = None,
) -> GameResult:
    """Play a full Hurdle match and return scored results."""
    secret = normalize_word(secret_word)
    history: list[tuple[str, list[Feedback]]] = []
    turns: list[TurnRecord] = []
    won = False

    for turn_num in range(1, MAX_TURNS + 1):
        raw_guess = guess_fn(secret, history, turn_num)
        guess = normalize_word(raw_guess)
        evaluation = evaluate_move(guess, history)
        feedback = (
            score_guess(secret, guess)
            if evaluation.move_is_legal
            else ["X"] * WORD_LENGTH
        )

        record = TurnRecord(
            turn=turn_num,
            guess=guess if guess else raw_guess.strip().upper()[:WORD_LENGTH],
            feedback=feedback,
            evaluation=evaluation,
        )
        turns.append(record)

        if on_turn:
            on_turn(record, turns.copy())

        if evaluation.move_is_legal:
            history.append((guess, feedback))
            if guess == secret:
                won = True
                break

    metrics = aggregate_game_metrics(turns)
    return GameResult(
        llm_provider=llm_provider,
        secret_word=secret,
        win=won,
        total_turns=len(turns),
        turns=turns,
        **metrics,
    )


def _format_history_for_prompt(history: list[tuple[str, list[Feedback]]]) -> str:
    if not history:
        return "No guesses yet."
    lines = []
    for guess, feedback in history:
        lines.append(f"{guess} -> {feedback_to_emojis(feedback)}")
    return "\n".join(lines)


def mock_llm_guess(
    secret: str,
    history: list[tuple[str, list[Feedback]]],
    turn: int,
    rng: random.Random | None = None,
) -> str:
    """Deterministic-ish mock player that filters the word list by clues."""
    _ = secret
    source = rng or random
    constraints = build_constraints(history)
    candidates = [
        word
        for word in VALID_WORDS
        if respects_clues(word, constraints)
        and word not in {g for g, _ in history}
    ]

    if not candidates:
        if turn == 1:
            return source.choice(["CRANE", "SLATE", "AROSE"])
        return source.choice(["ZZZZZ", "QQQQQ", "XXXXX"])

    preferred = [w for w in candidates if w in {"CRANE", "SLATE", "AROSE", "TRACE"}]
    if turn == 1 and preferred:
        return source.choice(preferred)
    return source.choice(candidates[: max(1, len(candidates))])


def _llm_word_prompt(turn: int, history: list[tuple[str, list[Feedback]]]) -> str:
    return (
        "You are playing Hurdle, a 5-letter Wordle variant. "
        "Reply with exactly one 5-letter English word in UPPERCASE, nothing else.\n\n"
        f"Turn: {turn}/{MAX_TURNS}\n"
        f"History:\n{_format_history_for_prompt(history)}"
    )


def openai_guess(secret: str, history: list[tuple[str, list[Feedback]]], turn: int, api_key: str) -> str:
    _ = secret
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Return only a single 5-letter word."},
            {"role": "user", "content": _llm_word_prompt(turn, history)},
        ],
        temperature=0.7,
        max_tokens=8,
    )
    return response.choices[0].message.content or ""


def grok_guess(secret: str, history: list[tuple[str, list[Feedback]]], turn: int, api_key: str) -> str:
    """xAI Grok via OpenAI-compatible API (https://api.x.ai/v1)."""
    _ = secret
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
    response = client.chat.completions.create(
        model="grok-4.3",
        messages=[
            {"role": "system", "content": "Return only a single 5-letter word."},
            {"role": "user", "content": _llm_word_prompt(turn, history)},
        ],
        temperature=0.7,
        max_tokens=8,
    )
    return response.choices[0].message.content or ""


def anthropic_guess(secret: str, history: list[tuple[str, list[Feedback]]], turn: int, api_key: str) -> str:
    _ = secret
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    prompt = (
        "You are playing Hurdle (5-letter Wordle). "
        "Respond with exactly one 5-letter English word in UPPERCASE only.\n\n"
        f"Turn: {turn}/{MAX_TURNS}\n"
        f"History:\n{_format_history_for_prompt(history)}"
    )
    message = client.messages.create(
        model="claude-3-5-haiku-latest",
        max_tokens=16,
        messages=[{"role": "user", "content": prompt}],
    )
    block = message.content[0]
    text = block.text if hasattr(block, "text") else str(block)
    return text


def gemini_guess(secret: str, history: list[tuple[str, list[Feedback]]], turn: int, api_key: str) -> str:
    _ = secret
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")
    prompt = (
        "You are playing Hurdle (5-letter Wordle). "
        "Reply with exactly one 5-letter English word in UPPERCASE, no punctuation.\n\n"
        f"Turn: {turn}/{MAX_TURNS}\n"
        f"History:\n{_format_history_for_prompt(history)}"
    )
    response = model.generate_content(prompt)
    return response.text or ""


def make_guess_fn(provider: str, api_key: str | None = None) -> Callable:
    if provider == "Mock LLM":
        seed = random.randint(0, 1_000_000)
        rng = random.Random(seed)
        return lambda secret, history, turn: mock_llm_guess(secret, history, turn, rng)

    if not api_key:
        raise ValueError(f"API key is required for {provider}.")

    if provider == "OpenAI":
        return lambda secret, history, turn: openai_guess(secret, history, turn, api_key)
    if provider == "Anthropic":
        return lambda secret, history, turn: anthropic_guess(secret, history, turn, api_key)
    if provider == "Gemini":
        return lambda secret, history, turn: gemini_guess(secret, history, turn, api_key)
    if provider == "Grok":
        return lambda secret, history, turn: grok_guess(secret, history, turn, api_key)

    raise ValueError(f"Unknown provider: {provider}")


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def export_game_result(result: GameResult) -> None:
    """Append summary row to CSV and deep-log turn data to JSON."""
    ensure_data_dir()
    _append_summary_csv(result)
    _append_results_json(result)


def _append_summary_csv(result: GameResult) -> None:
    file_exists = SUMMARY_CSV.exists()
    with SUMMARY_CSV.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": result.timestamp,
                "llm_provider": result.llm_provider,
                "secret_word": result.secret_word,
                "win": result.win,
                "total_turns": result.total_turns,
                "rule_violations": result.rule_violations,
                "competent_moves": result.competent_moves,
                "incompetent_moves": result.incompetent_moves,
                "illegal_moves": result.illegal_moves,
                "first_failure_turn": result.first_failure_turn,
                "legal_move_rate": f"{result.legal_move_rate:.4f}",
                "illegal_move_rate": f"{result.illegal_move_rate:.4f}",
                "competency_rate": f"{result.competency_rate:.4f}",
                "incompetency_rate": f"{result.incompetency_rate:.4f}",
            }
        )


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


def load_summary_rows() -> list[dict]:
    if not SUMMARY_CSV.exists():
        return []
    with SUMMARY_CSV.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_results_json() -> dict:
    if not RESULTS_JSON.exists():
        return {"games": []}
    with RESULTS_JSON.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_export_file(path: Path) -> bytes:
    if not path.exists():
        return b""
    return path.read_bytes()
