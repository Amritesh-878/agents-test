"""Streamlit teacher-evaluation demo for the Adira Academy Learning Assistant.

Teachers/leads pick a student, ask questions, and inspect the retrieved source
chunks behind every answer — the point being to confirm answers are *grounded*
in that student's real class transcripts, not invented.

Thin UI glue only: all testable logic lives in ``scripts.demo_backend`` and the
committed pipeline modules. Run with:

    streamlit run app.py
"""

from __future__ import annotations

import os

import streamlit as st
from groq import RateLimitError

from scripts.chat import GROQ_EGRESS_NOTICE, ChatTurnRecord, GroqChatBackend, load_groq_api_key
from scripts.demo_backend import (
    answer_for_student,
    section_of_class,
    session_display_label,
    student_summary,
    students_by_section,
)
from scripts.retrieval import QueryEmbedder
from scripts.utils.db_url import resolve_db_url
from scripts.utils.pg_store import PgVectorStore, connect_pg_store


@st.cache_resource
def get_store() -> PgVectorStore:
    return connect_pg_store(resolve_db_url(None))


@st.cache_resource
def get_embedder() -> QueryEmbedder:
    return QueryEmbedder()


@st.cache_resource
def get_chat_backend() -> GroqChatBackend:
    return GroqChatBackend(load_groq_api_key())


def render_sources(turn: ChatTurnRecord) -> None:
    result = turn.retrieval_result
    label = f"Grounding — {result.result_count} class record(s) this answer is based on"
    with st.expander(label, expanded=False):
        if result.result_count == 0:
            st.info("No matching class records — the assistant declined rather than guess.")
            return
        for chunk in result.retrieved_chunks:
            score = "n/a" if chunk.score is None else f"{chunk.score:.3f}"
            st.markdown(
                f"**[{chunk.rank}] {chunk.chunk_type}** · score {score} · "
                f"{chunk.start:.1f}–{chunk.end:.1f}s · speaker: {chunk.source_speaker or 'n/a'}"
            )
            st.text(chunk.text)
            st.divider()


def require_access_code() -> None:
    """Optional shared-secret gate for exposing the demo over a public tunnel.

    Disabled by default so local ``streamlit run app.py`` is unchanged; set the
    ``DEMO_ACCESS_CODE`` env var before tunnelling to require the code before any
    student data renders. A leaked/preview-fetched tunnel URL alone is then useless.
    """
    expected = os.environ.get("DEMO_ACCESS_CODE", "").strip()
    if not expected or st.session_state.get("access_ok"):
        return
    st.title("Adira Academy Learning Assistant")
    entered = st.text_input("Access code", type="password")
    if entered and entered == expected:
        st.session_state["access_ok"] = True
        st.rerun()
    st.stop()


def main() -> None:
    st.set_page_config(page_title="Adira Learning Assistant", page_icon="🎓")
    require_access_code()
    st.title("Adira Academy Learning Assistant")
    st.caption(
        "Teacher review build. Each student's assistant can only see that "
        "student's own class records."
    )

    store = get_store()
    pairs = store.list_student_class_pairs()
    if not pairs:
        st.warning("No students found in pgvector yet. Run the pipeline to embed a class first.")
        return
    sections = students_by_section(pairs)

    all_classes = "All classes"
    selected_section = st.selectbox("Class", [all_classes, *sections.keys()])
    if selected_section == all_classes:
        seen: dict[str, str] = {}
        for sid, name, _ in pairs:
            seen.setdefault(sid, name)
        students = sorted(seen.items(), key=lambda item: (item[1].lower(), item[0]))
    else:
        students = sections[selected_section]

    labels = {f"{name} ({sid})": (sid, name) for sid, name in students}
    selected_label = st.selectbox("Student", list(labels.keys()))
    student_id, student_name = labels[selected_label]

    # Per-session scope. "All sessions" = no class filter (current behavior). The dropdown
    # SHOWS a cleaned label but FILTERS on the real class_name value.
    all_sessions = "All sessions"
    class_options: dict[str, str | None] = {all_sessions: None}
    for class_name in store.list_student_classes(student_id):
        if selected_section != all_classes and section_of_class(class_name) != selected_section:
            continue
        class_options[session_display_label(class_name)] = class_name
    selected_session_label = st.selectbox(
        "Session",
        list(class_options.keys()),
        help=(
            "Pick one session for 'what did we cover today?' questions. "
            "'All sessions' searches everything this student has, including "
            "their other subjects."
        ),
    )
    selected_class_name = class_options[selected_session_label]

    summary = student_summary(store, student_id, student_name)
    history: dict[str, list[ChatTurnRecord]] = st.session_state.setdefault("history", {})
    turns = history.setdefault(student_id, [])

    col_a, col_b = st.columns(2)
    col_a.metric("Sessions on record", str(len(summary.class_names)))
    col_b.metric("Class records", str(summary.chunk_count))
    if summary.class_names:
        st.caption("Sessions: " + "; ".join(session_display_label(c) for c in summary.class_names))
    st.caption(GROQ_EGRESS_NOTICE)

    for turn in turns:
        with st.chat_message("user"):
            st.write(turn.question)
        with st.chat_message("assistant"):
            st.write(turn.answer)
            render_sources(turn)

    question = st.chat_input(f"Ask {student_name}'s assistant a question…")
    if question:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Checking the class records…"):
                try:
                    turn = answer_for_student(
                        student_id=student_id,
                        student_name=student_name,
                        question=question,
                        store=store,
                        embedder=get_embedder(),
                        chat_backend=get_chat_backend(),
                        db_url=resolve_db_url(None),
                        class_name=selected_class_name,
                        history_turns=turns,
                    )
                except RateLimitError:
                    st.warning(
                        "The answer service is briefly busy. Please wait about "
                        "20 seconds and ask the same question again."
                    )
                    st.stop()
            st.write(turn.answer)
            render_sources(turn)
        turns.append(turn)


if __name__ == "__main__":
    main()
