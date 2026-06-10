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
python -c "from engine import run_game, make_guess_fn, export_game_result; r = run_game('CRANE', 'Mock LLM', make_guess_fn('Mock LLM')); print(r.win, r.total_turns)"
```

### Data exports

Evaluation results append to `data/evaluation_summary.csv` and `data/evaluation_results.json` after each run.

### API keys

OpenAI, Anthropic, and Gemini require keys in the sidebar. API errors surface as `st.error` alerts in the UI.
