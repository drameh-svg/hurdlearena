"""Streamlit dashboard for Hurdle LLM evaluation."""

from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from engine import (
    MAX_TURNS,
    RESULTS_JSON,
    SUMMARY_CSV,
    GameResult,
    TurnRecord,
    export_game_result,
    feedback_to_emojis,
    is_valid_secret,
    load_summary_rows,
    make_guess_fn,
    random_secret,
    read_export_file,
    run_game,
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
    </style>
    """,
    unsafe_allow_html=True,
)

PROVIDERS = ["Mock LLM", "OpenAI", "Anthropic", "Gemini"]


def init_session_state() -> None:
    defaults = {
        "secret_word": random_secret(),
        "last_result": None,
        "is_running": False,
        "grid_rows": [],
        "live_metrics": None,
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
        st.info("Press **Run Evaluation** to watch the LLM play live.")
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


def main() -> None:
    init_session_state()

    st.markdown('<p class="hero">🎯 Hurdle Arena</p>', unsafe_allow_html=True)
    st.caption(
        "Evaluate and compare LLMs playing Hurdle — a 5-letter Wordle variant with live scoring."
    )

    with st.sidebar:
        st.header("⚙️ Controls")
        provider = st.selectbox("Target LLM", PROVIDERS, index=0)
        api_key = st.text_input(
            "API Key",
            type="password",
            help="Required for OpenAI, Anthropic, and Gemini. Mock LLM needs no key.",
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

        run_clicked = st.button("▶️ Run Evaluation", type="primary", use_container_width=True)

    secret_word = st.session_state.secret_word
    st.subheader("Live Match")
    grid_placeholder = st.empty()
    metrics_placeholder = st.empty()

    if run_clicked:
        if not is_valid_secret(secret_word):
            st.error("Secret word must be a valid 5-letter word from the built-in list.")
        elif provider != "Mock LLM" and not api_key:
            st.error(f"Please provide an API key for {provider}.")
        else:
            st.session_state.is_running = True
            st.session_state.grid_rows = []
            st.session_state.live_metrics = None

            grid_slot = grid_placeholder.container()
            metrics_slot = metrics_placeholder.container()

            try:
                guess_fn = make_guess_fn(provider, api_key or None)

                def on_turn(turn: TurnRecord, all_turns: list[TurnRecord]) -> None:
                    st.session_state.grid_rows.append(build_grid_row(turn))
                    legal = sum(1 for t in all_turns if t.evaluation.move_is_legal)
                    competent = sum(
                        1 for t in all_turns if t.evaluation.move_competency == "competent"
                    )
                    violations = sum(
                        1 for t in all_turns if t.evaluation.failure_type == "rule_violation"
                    )
                    st.session_state.live_metrics = {
                        "win": turn.guess == secret_word and turn.evaluation.move_is_legal,
                        "turns": len(all_turns),
                        "competency_rate": competent / len(all_turns),
                        "violations": violations,
                        "finished": turn.guess == secret_word and turn.evaluation.move_is_legal
                        or len(all_turns) >= MAX_TURNS,
                    }

                    with grid_slot:
                        render_grid(st.session_state.grid_rows)
                    with metrics_slot:
                        m = st.session_state.live_metrics
                        c1, c2, c3, c4 = st.columns(4)
                        with c1:
                            st.markdown(
                                render_metric_card("Win / Loss", "WIN ✅" if m["win"] else "IN PROGRESS"),
                                unsafe_allow_html=True,
                            )
                        with c2:
                            st.markdown(
                                render_metric_card("Total Turns", str(m["turns"])),
                                unsafe_allow_html=True,
                            )
                        with c3:
                            st.markdown(
                                render_metric_card("Competency Rate", pct(m["competency_rate"])),
                                unsafe_allow_html=True,
                            )
                        with c4:
                            st.markdown(
                                render_metric_card("Rule Violations", str(m["violations"])),
                                unsafe_allow_html=True,
                            )
                    time.sleep(0.45)

                result = run_game(
                    secret_word=secret_word,
                    llm_provider=provider,
                    guess_fn=guess_fn,
                    on_turn=on_turn,
                )
                export_game_result(result)
                st.session_state.last_result = result
                st.session_state.is_running = False

                with metrics_slot:
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.markdown(
                            render_metric_card(
                                "Win / Loss",
                                "WIN ✅" if result.win else "LOSS ❌",
                            ),
                            unsafe_allow_html=True,
                        )
                    with c2:
                        st.markdown(
                            render_metric_card("Total Turns", str(result.total_turns)),
                            unsafe_allow_html=True,
                        )
                    with c3:
                        st.markdown(
                            render_metric_card(
                                "Competency Rate",
                                pct(result.competency_rate),
                            ),
                            unsafe_allow_html=True,
                        )
                    with c4:
                        st.markdown(
                            render_metric_card(
                                "Rule Violations",
                                str(result.rule_violations),
                            ),
                            unsafe_allow_html=True,
                        )

                st.success(
                    f"Evaluation complete — {'won' if result.win else 'lost'} in "
                    f"{result.total_turns} turn(s). Results exported."
                )

            except Exception as exc:
                st.session_state.is_running = False
                st.error(f"API / runtime error: {exc}")

    else:
        with grid_placeholder.container():
            render_grid(st.session_state.grid_rows)
        if st.session_state.last_result:
            result: GameResult = st.session_state.last_result
            with metrics_placeholder.container():
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.markdown(
                        render_metric_card(
                            "Win / Loss",
                            "WIN ✅" if result.win else "LOSS ❌",
                        ),
                        unsafe_allow_html=True,
                    )
                with c2:
                    st.markdown(
                        render_metric_card("Total Turns", str(result.total_turns)),
                        unsafe_allow_html=True,
                    )
                with c3:
                    st.markdown(
                        render_metric_card(
                            "Competency Rate",
                            pct(result.competency_rate),
                        ),
                        unsafe_allow_html=True,
                    )
                with c4:
                    st.markdown(
                        render_metric_card(
                            "Rule Violations",
                            str(result.rule_violations),
                        ),
                        unsafe_allow_html=True,
                    )

    st.divider()
    st.subheader("📊 Historical Analysis")

    rows = load_summary_rows()
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No evaluation runs yet. Complete a match to populate history.")

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
                st.write(
                    f"**Turn {turn.turn}:** `{turn.guess}` "
                    f"{feedback_to_emojis(turn.feedback)} — "
                    f"score {ev.numeric_score} "
                    f"({ev.move_competency or ev.failure_type})"
                )


if __name__ == "__main__":
    main()
