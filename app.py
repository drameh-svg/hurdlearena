"""Streamlit dashboard for Hurdle LLM evaluation."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from engine import (
    MAX_TURNS,
    RESULTS_JSON,
    SUMMARY_CSV,
    GameResult,
    TurnRecord,
    aggregate_game_metrics,
    append_human_eval_row,
    execute_turn,
    export_game_result,
    feedback_to_emojis,
    get_next_episode,
    is_valid_secret,
    load_human_eval_rows,
    make_guess_fn,
    random_secret,
    read_export_file,
)

st.set_page_config(
    page_title="Hurdle Arena — LLM Evaluation",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .tile-row {
        font-size: 2rem;
        letter-spacing: 0.35rem;
        font-family: "Courier New", monospace;
        margin-bottom: 0.35rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.25rem 1rem;
        text-align: center;
        box-shadow: 0 4px 14px rgba(0,0,0,0.25);
    }
    .metric-label {
        color: #94a3b8;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .metric-value {
        color: #f8fafc;
        font-size: 2rem;
        font-weight: 700;
        margin-top: 0.35rem;
    }
    .hero {
        background: linear-gradient(90deg, #6366f1, #8b5cf6, #d946ef);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.4rem;
        margin-bottom: 0.25rem;
    }
    .eval-legend {
        background: #f1f5f9;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.75rem;
        color: #0f172a;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

PROVIDERS = ["Mock LLM", "OpenAI", "Anthropic", "Gemini", "Grok"]

HUMAN_EVAL_LABELS = {
    1: "1 — Rule violation",
    2: "2 — Incompetent",
    3: "3 — Competent",
}


def init_session_state() -> None:
    defaults = {
        "secret_word": random_secret(),
        "last_result": None,
        "grid_rows": [],
        "game_phase": "idle",
        "active_game": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_metric_card(label: str, value: str) -> str:
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
    </div>
    """


def render_grid(rows: list[dict]) -> None:
    if not rows:
        st.info("Press **Run Evaluation** to start. You will rate each move before the next turn.")
        return

    for row in rows:
        letters = " ".join(row["guess"].ljust(5)[:5])
        emojis = row["emoji_line"]
        st.markdown(
            f'<div class="tile-row">{letters}<br>{emojis}</div>',
            unsafe_allow_html=True,
        )

    remaining = MAX_TURNS - len(rows)
    for _ in range(remaining):
        st.markdown(
            '<div class="tile-row">- - - - -<br>⬜⬜⬜⬜⬜</div>',
            unsafe_allow_html=True,
        )


def build_grid_row(turn: TurnRecord) -> dict:
    return {
        "guess": turn.guess,
        "emoji_line": feedback_to_emojis(turn.feedback),
    }


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def start_new_game(provider: str, api_key: str | None, secret_word: str) -> None:
    st.session_state.active_game = {
        "provider": provider,
        "api_key": api_key,
        "secret": secret_word.upper(),
        "episode": get_next_episode(),
        "history": [],
        "turns": [],
        "grid_rows": [],
        "pending_turn": None,
    }
    st.session_state.grid_rows = []
    st.session_state.game_phase = "playing"
    request_llm_turn()


def request_llm_turn() -> None:
    game = st.session_state.active_game
    turn_num = len(game["turns"]) + 1
    guess_fn = make_guess_fn(game["provider"], game["api_key"])
    llm_response = guess_fn(game["secret"], game["history"], turn_num)
    turn = execute_turn(game["secret"], game["history"], turn_num, llm_response)
    game["pending_turn"] = turn
    st.session_state.game_phase = "await_human"


def submit_human_evaluation(human_score: int) -> None:
    game = st.session_state.active_game
    turn: TurnRecord = game["pending_turn"]
    turn.human_evaluation = human_score

    append_human_eval_row(
        player=game["provider"],
        episode=game["episode"],
        turn=turn.turn,
        word_guessed=turn.guess,
        model_reason=turn.model_reason,
        human_evaluation=human_score,
    )

    game["turns"].append(turn)
    game["grid_rows"].append(build_grid_row(turn))
    st.session_state.grid_rows = game["grid_rows"]
    game["pending_turn"] = None

    if turn.evaluation.move_is_legal:
        game["history"].append((turn.guess, turn.feedback))

    won = turn.evaluation.move_is_legal and turn.guess == game["secret"]
    if won or len(game["turns"]) >= MAX_TURNS:
        finish_game(won)
    else:
        st.session_state.game_phase = "playing"
        request_llm_turn()


def finish_game(won: bool) -> None:
    game = st.session_state.active_game
    metrics = aggregate_game_metrics(game["turns"])
    result = GameResult(
        llm_provider=game["provider"],
        secret_word=game["secret"],
        win=won,
        total_turns=len(game["turns"]),
        turns=game["turns"],
        **metrics,
    )
    export_game_result(result)
    st.session_state.last_result = result
    st.session_state.game_phase = "finished"
    st.session_state.active_game = None


def render_metrics(turns: list[TurnRecord], won: bool | None = None) -> None:
    if not turns:
        return
    metrics = aggregate_game_metrics(turns)
    if won is None:
        last = turns[-1]
        secret = st.session_state.active_game["secret"]
        if last.evaluation.move_is_legal and last.guess == secret:
            win_label = "WIN ✅"
        elif len(turns) >= MAX_TURNS:
            win_label = "LOSS ❌"
        else:
            win_label = "IN PROGRESS"
    else:
        win_label = "WIN ✅" if won else "LOSS ❌"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(render_metric_card("Win / Loss", win_label), unsafe_allow_html=True)
    with c2:
        st.markdown(render_metric_card("Total Turns", str(len(turns))), unsafe_allow_html=True)
    with c3:
        st.markdown(
            render_metric_card("Competency Rate", pct(metrics["competency_rate"])),
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            render_metric_card("Rule Violations", str(metrics["rule_violations"])),
            unsafe_allow_html=True,
        )


def render_human_eval_prompt() -> None:
    game = st.session_state.active_game
    turn: TurnRecord = game["pending_turn"]

    st.markdown(
        '<div class="eval-legend">'
        "<strong>Rate this move:</strong> "
        "<strong>1</strong> = Rule violation &nbsp;|&nbsp; "
        "<strong>2</strong> = Incompetent &nbsp;|&nbsp; "
        "<strong>3</strong> = Competent"
        "</div>",
        unsafe_allow_html=True,
    )

    st.subheader(f"Turn {turn.turn} — Your Evaluation")
    st.write(f"**Word guessed:** `{turn.guess}` {feedback_to_emojis(turn.feedback)}")
    st.write(f"**Model's reason:** {turn.model_reason}")
    st.caption(
        f"Auto-scoring hint (not saved to CSV): "
        f"{turn.evaluation.move_competency or turn.evaluation.failure_type} "
        f"(engine score {turn.evaluation.numeric_score})"
    )

    human_score = st.radio(
        "Human Evaluation (1, 2, 3)",
        options=[1, 2, 3],
        format_func=lambda x: HUMAN_EVAL_LABELS[x],
        horizontal=True,
        key=f"human_eval_{game['episode']}_{turn.turn}",
    )

    if st.button("Submit Evaluation & Continue", type="primary", use_container_width=True):
        submit_human_evaluation(human_score)
        st.rerun()


def main() -> None:
    init_session_state()

    st.markdown('<p class="hero">🎯 Hurdle Arena</p>', unsafe_allow_html=True)
    st.caption(
        "Evaluate LLMs playing Hurdle. After each move, you rate it 1–3 before the game continues."
    )

    with st.sidebar:
        st.header("⚙️ Controls")
        provider = st.selectbox("Target LLM", PROVIDERS, index=0)
        api_key = st.text_input(
            "API Key",
            type="password",
            help="Required for OpenAI, Anthropic, Gemini, and Grok. Mock LLM needs no key.",
            disabled=provider == "Mock LLM",
        )

        secret_input = st.text_input(
            "Secret 5-letter word",
            value=st.session_state.secret_word,
            max_chars=5,
        ).strip()

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🎲 Randomize", use_container_width=True):
                st.session_state.secret_word = random_secret()
                st.rerun()
        with col_b:
            if secret_input:
                cleaned = secret_input.upper()
                if is_valid_secret(cleaned):
                    st.session_state.secret_word = cleaned
                else:
                    st.warning("Enter a valid 5-letter alphabetic word.")

        game_busy = st.session_state.game_phase in {"playing", "await_human"}
        run_clicked = st.button(
            "▶️ Run Evaluation",
            type="primary",
            use_container_width=True,
            disabled=game_busy,
        )

    secret_word = st.session_state.secret_word
    st.subheader("Live Match")
    render_grid(st.session_state.grid_rows)

    if run_clicked:
        if not is_valid_secret(secret_word):
            st.error("Secret word must be a valid 5-letter alphabetic word.")
        elif provider != "Mock LLM" and not api_key:
            st.error(f"Please provide an API key for {provider}.")
        else:
            try:
                start_new_game(provider, api_key or None, secret_word)
                st.rerun()
            except Exception as exc:
                st.session_state.game_phase = "idle"
                st.session_state.active_game = None
                st.error(f"API / runtime error: {exc}")

    phase = st.session_state.game_phase

    if phase == "await_human":
        render_metrics(st.session_state.active_game["turns"])
        render_human_eval_prompt()
    elif phase == "finished" and st.session_state.last_result:
        result: GameResult = st.session_state.last_result
        render_metrics(result.turns, won=result.win)
        st.success(
            f"Episode complete — {'won' if result.win else 'lost'} in "
            f"{result.total_turns} turn(s). Spreadsheet updated."
        )
    elif st.session_state.last_result and phase == "idle":
        render_metrics(st.session_state.last_result.turns, won=st.session_state.last_result.win)

    st.divider()
    st.subheader("📊 Historical Analysis")

    rows = load_human_eval_rows()
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No evaluation runs yet. Complete a match to populate the spreadsheet.")

    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            label="⬇️ Download evaluation_summary.csv",
            data=read_export_file(SUMMARY_CSV),
            file_name="evaluation_summary.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with d2:
        st.download_button(
            label="⬇️ Download evaluation_results.json",
            data=read_export_file(RESULTS_JSON),
            file_name="evaluation_results.json",
            mime="application/json",
            use_container_width=True,
        )

    if st.session_state.last_result:
        with st.expander("Last run — turn-by-turn detail"):
            for turn in st.session_state.last_result.turns:
                ev = turn.evaluation
                human = turn.human_evaluation
                st.write(
                    f"**Turn {turn.turn}:** `{turn.guess}` "
                    f"{feedback_to_emojis(turn.feedback)} — "
                    f"human eval **{human}**, auto {ev.numeric_score} "
                    f"({ev.move_competency or ev.failure_type})"
                )
                st.caption(f"Reason: {turn.model_reason}")


if __name__ == "__main__":
    main()
