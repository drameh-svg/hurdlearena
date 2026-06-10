# Hurdle Arena

Streamlit web app that evaluates LLMs playing Hurdle (5-letter Wordle variant).

## Cursor Cloud specific instructions

### Services

| Service | Command | Notes |
|---------|---------|-------|
| Streamlit dashboard | `streamlit run app.py --server.port 8501 --server.address 0.0.0.0` | Main (and only) application |

No database, Docker, or external services are required. Use **Mock LLM** in the sidebar for full end-to-end testing without API keys.

### Lint / test

There is no dedicated test suite yet. Validate with:

```bash
python -c "from engine import run_game, make_guess_fn; from words import SOLUTION_WORDS; r = run_game(list(SOLUTION_WORDS)[:5], 'Mock LLM', make_guess_fn('Mock LLM')); print(r.win, r.total_turns)"
```

Word lists live in `wordlists/` (~2,309 solutions, ~12,947 accepted guesses). Invalid or duplicate guesses are logged off-board; humans rate rule violations (1) — the engine does not auto-flag them.

### Data exports

Evaluation results append to `data/evaluation_summary.csv` and `data/evaluation_results.json` after each run.

### API keys

OpenAI, Anthropic, and Gemini require keys in the sidebar. API errors surface as `st.error` alerts in the UI.
