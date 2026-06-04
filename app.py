"""Streamlit teacher-evaluation demo for the Adira Academy Learning Assistant.

Teachers/leads pick a student, ask questions, and inspect the retrieved source
chunks behind every answer — the point being to confirm answers are *grounded*
in that student's real class transcripts, not invented.

Thin UI glue only: all testable logic lives in ``scripts.demo_backend`` and the
committed pipeline modules. Run with:

    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from scripts.chat import GROQ_EGRESS_NOTICE, ChatTurnRecord, GroqChatBackend, load_groq_api_key
from scripts.demo_backend import (
    answer_for_student,
    session_display_label,
    student_summary,
    top_score,
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
    label = f"Grounding — {result.result_count} source chunk(s) retrieved"
    with st.expander(label, expanded=False):
        if result.result_count == 0:
            st.info("No source chunks matched — the assistant declined to invent an answer.")
            return
        for chunk in result.retrieved_chunks:
            score = "n/a" if chunk.score is None else f"{chunk.score:.3f}"
            st.markdown(
                f"**[{chunk.rank}] {chunk.chunk_type}** · score {score} · "
                f"{chunk.start:.1f}–{chunk.end:.1f}s · speaker: {chunk.source_speaker or 'n/a'}"
            )
            st.text(chunk.text)
            st.divider()


def main() -> None:
    st.set_page_config(page_title="Adira Learning Assistant — Teacher Demo", page_icon="🎓")
    st.title("Adira Academy Learning Assistant — Teacher Demo")
    st.caption(
        "Each student's chatbot sees ONLY that student's own class data "
        "(retrieval is scoped to their student_id — full isolation)."
    )

    store = get_store()
    students = store.list_students()
    if not students:
        st.warning("No students found in pgvector yet. Run the pipeline to embed a class first.")
        return

    labels = {f"{name} ({sid})": (sid, name) for sid, name in students}
    selected_label = st.selectbox("Student", list(labels.keys()))
    student_id, student_name = labels[selected_label]

    # Per-session scope. "All sessions" = no class filter (current behavior). The dropdown
    # SHOWS a cleaned label but FILTERS on the real class_name value.
    all_sessions = "All sessions"
    class_options: dict[str, str | None] = {all_sessions: None}
    for class_name in store.list_student_classes(student_id):
        class_options[session_display_label(class_name)] = class_name
    selected_session_label = st.selectbox("Class / session", list(class_options.keys()))
    selected_class_name = class_options[selected_session_label]

    summary = student_summary(store, student_id, student_name)
    history: dict[str, list[ChatTurnRecord]] = st.session_state.setdefault("history", {})
    turns = history.setdefault(student_id, [])
    last_score = top_score(turns[-1].retrieval_result) if turns else None

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Class(es)", str(len(summary.class_names)))
    col_b.metric("Embedded chunks", str(summary.chunk_count))
    col_c.metric("Last top score", "—" if last_score is None else f"{last_score:.3f}")
    if summary.class_names:
        st.caption("Classes: " + "; ".join(summary.class_names))
    st.caption(f"Session scope: {selected_session_label}")
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
            with st.spinner("Retrieving grounded context and generating answer…"):
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
            st.write(turn.answer)
            render_sources(turn)
        turns.append(turn)


if __name__ == "__main__":
    main()
