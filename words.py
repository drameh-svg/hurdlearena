"""Hurdle / Wordle-scale word lists for secrets and guess validation."""

from __future__ import annotations

import json
import re
from pathlib import Path

WORD_LENGTH = 5
WORDLIST_DIR = Path(__file__).parent / "wordlists"


def _parse_wordlist_file(path: Path) -> frozenset[str]:
    """Load a comma-separated quoted word list (NYT Wordle export format)."""
    if not path.exists():
        raise FileNotFoundError(f"Missing word list: {path}")

    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return frozenset()

    try:
        words = json.loads(f"[{raw}]")
    except json.JSONDecodeError:
        words = re.findall(r"[A-Za-z]{5}", raw)

    return frozenset(word.upper() for word in words if len(word) == WORD_LENGTH)


def _load_word_sets() -> tuple[frozenset[str], frozenset[str]]:
    solutions = _parse_wordlist_file(WORDLIST_DIR / "answers.txt")
    nonsolutions = _parse_wordlist_file(WORDLIST_DIR / "nonsolutions.txt")
    guesses = solutions | nonsolutions
    return solutions, guesses


SOLUTION_WORDS, GUESS_WORDS = _load_word_sets()

# Backward-compatible alias: all playable guesses (~13k, Arkadium-scale).
VALID_WORDS = GUESS_WORDS
