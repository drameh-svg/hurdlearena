# Hurdle Arena

Evaluate and compare LLMs playing **Hurdle** — a 5-letter Wordle variant — via a Streamlit dashboard with live scoring, move evaluation, and CSV/JSON export.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open the URL shown in the terminal (default `http://localhost:8501`).

## Features

- **Sidebar controls:** LLM provider (OpenAI, Anthropic, Gemini, Mock LLM), API key, secret word, randomize, Run Evaluation
- **Live match grid:** Wordle-style 🟩🟨⬛ tiles updating as the LLM plays
- **Metrics:** Win/Loss, Total Turns, Competency Rate, Rule Violations
- **Exports:** `data/evaluation_summary.csv` and `data/evaluation_results.json` with browser download buttons

## Mock LLM

Select **Mock LLM** to test the full UI, scoring logic, and file downloads without an API key.

## Project layout

```
app.py       # Streamlit dashboard
engine.py    # Game logic, scoring, LLM clients, export
words.py     # Built-in 5-letter word list
data/        # Generated evaluation exports
```
