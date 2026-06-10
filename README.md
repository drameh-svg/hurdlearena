# Hurdle Arena

Evaluate and compare LLMs playing **Hurdle** — the five-stage daily word challenge (Wordle mechanics with carry-over guesses) — via a Streamlit dashboard with human move ratings and CSV/JSON export.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open the URL shown in the terminal (default `http://localhost:8501`).

## Features

- **Sidebar controls:** LLM provider, API key, five hurdle secret words, randomize daily challenge, clear history, Run Evaluation
- **Live match grid:** Wordle-style 🟩🟨⬛ tiles updating as the LLM plays
- **Metrics:** Win/Loss, Total Turns, Competency Rate, Rule Violations
- **Human evaluation:** After each LLM move, rate it **1** (rule violation), **2** (incompetent), or **3** (competent) before the game continues
- **Exports:** `data/evaluation_summary.csv` (Player, Game, Episode, Turn, Word Guessed, Model's Reason, Human Evaluation of Turn, Did Turn Resolve Game?) and `data/evaluation_results.json`

## Mock LLM

Select **Mock LLM** to test the full UI, scoring logic, and file downloads without an API key.

## Project layout

```
app.py       # Streamlit dashboard
engine.py    # Game logic, scoring, LLM clients, export
words.py     # Built-in 5-letter word list
data/        # Generated evaluation exports
```
