"""Streamlit dashboard for Hurdle LLM evaluation."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from engine import (
    HURDLE_RULES,
    MAX_TURNS,
    NUM_HURDLES,
    RESULTS_JSON,
    SUMMARY_CSV,
    GameResult,
    HurdleContext,
    TurnRecord,
    aggregate_game_metrics,
    append_human_eval_row,
    automatic_guess_for_hurdle,
    clear_evaluation_history,
    execute_llm_turn,
    execute_turn,
    export_game_result,
    feedback_to_emojis,
    get_next_episode,
    hurdle_is_solved,
    is_valid_secret,
    load_human_eval_rows,
    llm_may_guess,
    make_guess_fn,
    random_daily_secrets,
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
        "daily_secrets": random_daily_secrets(),
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
        st.info(
            "Press **Run Evaluation** to start the five-hurdle daily challenge. "
            "You will rate each **LLM** move before play continues."
        )
        return

    for row in rows:
        prefix = "↪ " if row.get("automatic") else ""
        letters = " ".join(row["guess"].ljust(5)[:5])
        emojis = row["emoji_line"]
        st.markdown(
            f'<div class="tile-row">{prefix}{letters}<br>{emojis}</div>',
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
        "automatic": turn.is_automatic,
    }


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def current_secret(game: dict) -> str:
    return game["secrets"][game["hurdle_num"] - 1]


def finalize_turn(game: dict, turn: TurnRecord) -> str:
    game["hurdle_turns"].append(turn)
    game["all_turns"].append(turn)
    game["grid_rows"].append(build_grid_row(turn))
    st.session_state.grid_rows = game["grid_rows"]

    if turn.evaluation.move_is_legal:
        game["hurdle_history"].append((turn.guess, turn.feedback))

    secret = current_secret(game)
    if hurdle_is_solved(turn, secret):
        game["solved_answers"].append(secret)
        if game["hurdle_num"] >= NUM_HURDLES:
            finish_game(True)
            return "won"
        game["hurdle_num"] += 1
        game["hurdle_turns"] = []
        game["hurdle_history"] = []
        game["grid_rows"] = []
        st.session_state.grid_rows = []
        return "next_hurdle"

    if len(game["hurdle_turns"]) >= MAX_TURNS:
        finish_game(False)
        return "lost"
    return "continue"


def finish_game(won: bool) -> None:
    game = st.session_state.active_game
    llm_turns = [t for t in game["all_turns"] if not t.is_automatic]
    metrics = aggregate_game_metrics(llm_turns)
    result = GameResult(
        llm_provider=game["provider"],
        secret_word=",".join(game["secrets"]),
        win=won,
        total_turns=len(game["all_turns"]),
        turns=game["all_turns"],
        **metrics,
    )
    export_game_result(result)
    st.session_state.last_result = result
    st.session_state.game_phase = "finished"
    st.session_state.active_game = None


def request_llm_turn() -> None:
    game = st.session_state.active_game
    hurdle_num = game["hurdle_num"]
    turn_in_hurdle = len(game["hurdle_turns"]) + 1
    secret = current_secret(game)

    game["global_turn"] += 1
    ctx = HurdleContext(
        secret=secret,
        hurdle_num=hurdle_num,
        history=game["hurdle_history"],
        turn_in_hurdle=turn_in_hurdle,
        solved_answers=game["solved_answers"],
        secrets=game["secrets"],
    )
    guess_fn = make_guess_fn(game["provider"], game["api_key"])
    llm_response = guess_fn(ctx)
    turn = execute_llm_turn(
        secret,
        game["hurdle_history"],
        game["global_turn"],
        hurdle_num,
        turn_in_hurdle,
        llm_response,
    )
    game["pending_turn"] = turn
    st.session_state.game_phase = "await_human"


def continue_game() -> None:
    game = st.session_state.active_game
    while st.session_state.game_phase not in {"finished", "await_human"}:
        hurdle_num = game["hurdle_num"]
        turn_in_hurdle = len(game["hurdle_turns"]) + 1
        secret = current_secret(game)

        if turn_in_hurdle > MAX_TURNS:
            finish_game(False)
            return

        auto = automatic_guess_for_hurdle(hurdle_num, game["solved_answers"], turn_in_hurdle)
        if auto:
            word, reason = auto
            game["global_turn"] += 1
            turn = execute_turn(
                secret,
                game["hurdle_history"],
                game["global_turn"],
                hurdle_num,
                turn_in_hurdle,
                word,
                reason,
                is_automatic=True,
            )
            status = finalize_turn(game, turn)
            if status in {"won", "lost"}:
                return
            if status == "next_hurdle":
                continue
            continue

        if llm_may_guess(hurdle_num, turn_in_hurdle):
            request_llm_turn()
            return

        finish_game(False)
        return


def start_new_game(provider: str, api_key: str | None, secrets: list[str]) -> None:
    st.session_state.active_game = {
        "provider": provider,
        "api_key": api_key,
        "secrets": [word.upper() for word in secrets],
        "episode": get_next_episode(),
        "hurdle_num": 1,
        "solved_answers": [],
        "hurdle_history": [],
        "hurdle_turns": [],
        "all_turns": [],
        "grid_rows": [],
        "global_turn": 0,
        "pending_turn": None,
    }
    st.session_state.grid_rows = []
    st.session_state.game_phase = "playing"
    continue_game()


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

    game["pending_turn"] = None
    status = finalize_turn(game, turn)
    if status in {"continue", "next_hurdle"}:
        st.session_state.game_phase = "playing"
        continue_game()


def render_metrics(turns: list[TurnRecord], won: bool | None = None) -> None:
    llm_turns = [t for t in turns if not t.is_automatic]
    if not llm_turns:
        return
    metrics = aggregate_game_metrics(llm_turns)
    if won is None:
        win_label = "IN PROGRESS"
    else:
        win_label = "WIN ✅" if won else "LOSS ❌"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(render_metric_card("Challenge", win_label), unsafe_allow_html=True)
    with c2:
        st.markdown(render_metric_card("LLM Moves", str(len(llm_turns))), unsafe_allow_html=True)
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

    st.subheader(
        f"Hurdle {turn.hurdle_num} — Turn {turn.turn_in_hurdle} — Your Evaluation"
    )
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
        "Five-hurdle daily challenge — carry-over guesses, final-board pre-fill, "
        "and human ratings after each LLM move."
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

        if st.button("🎲 Randomize daily challenge (5 words)", use_container_width=True):
            st.session_state.daily_secrets = random_daily_secrets()
            st.rerun()

        with st.expander("Secret words (5 hurdles)", expanded=False):
            for idx in range(NUM_HURDLES):
                key = f"secret_{idx}"
                default = st.session_state.daily_secrets[idx]
                value = st.text_input(
                    f"Hurdle {idx + 1}",
                    value=default,
                    max_chars=5,
                    key=key,
                ).strip()
                if value and is_valid_secret(value):
                    st.session_state.daily_secrets[idx] = value.upper()

        game_busy = st.session_state.game_phase in {"playing", "await_human"}
        run_clicked = st.button(
            "▶️ Run Evaluation",
            type="primary",
            use_container_width=True,
            disabled=game_busy,
        )

        if st.button("🗑️ Clear evaluation history", use_container_width=True):
            clear_evaluation_history()
            st.session_state.last_result = None
            st.success("CSV and JSON history cleared.")
            st.rerun()

        with st.expander("Hurdle rules (sent to LLM)"):
            st.markdown(HURDLE_RULES)

    secrets = st.session_state.daily_secrets
    invalid = [idx + 1 for idx, word in enumerate(secrets) if not is_valid_secret(word)]
    if invalid:
        st.warning(f"Invalid secret word(s) for hurdle(s): {', '.join(map(str, invalid))}")

    active = st.session_state.active_game
    hurdle_label = (
        f"Hurdle {active['hurdle_num']} of {NUM_HURDLES}"
        if active
        else "Live Match"
    )
    st.subheader(hurdle_label)
    render_grid(st.session_state.grid_rows)

    if run_clicked:
        if invalid:
            st.error("All five hurdle secret words must be valid 5-letter alphabetic words.")
        elif provider != "Mock LLM" and not api_key:
            st.error(f"Please provide an API key for {provider}.")
        else:
            try:
                start_new_game(provider, api_key or None, secrets)
                st.rerun()
            except Exception as exc:
                st.session_state.game_phase = "idle"
                st.session_state.active_game = None
                st.error(f"API / runtime error: {exc}")

    phase = st.session_state.game_phase

    if phase == "await_human" and active:
        render_metrics(active["all_turns"])
        render_human_eval_prompt()
    elif phase == "finished" and st.session_state.last_result:
        result: GameResult = st.session_state.last_result
        render_metrics(result.turns, won=result.win)
        st.success(
            f"Daily challenge complete — {'won' if result.win else 'lost'} "
            f"({result.total_turns} total rows including carry-overs). Spreadsheet updated."
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
                auto = " (auto carry)" if turn.is_automatic else ""
                human_txt = f"human **{human}**" if human else "no human rating"
                st.write(
                    f"**Hurdle {turn.hurdle_num} turn {turn.turn_in_hurdle}:** "
                    f"`{turn.guess}` {feedback_to_emojis(turn.feedback)}{auto} — "
                    f"{human_txt}, auto {ev.numeric_score} "
                    f"({ev.move_competency or ev.failure_type})"
                )
                if turn.model_reason:
                    st.caption(f"Reason: {turn.model_reason}")


if __name__ == "__main__":
    main()
