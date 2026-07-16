from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Iterator, Sequence

import httpx
import pytest
from fastapi.testclient import TestClient

from scripts.api import SERVICE_TOKEN_HEADER, create_app
from scripts.chat import ChatError, PromptMessage
from scripts.embed_and_store import DEFAULT_EMBEDDING_MODEL
from scripts.models.pipeline import SearchResult
from scripts.retrieval import QueryEmbedder

TOKEN = "shared-service-secret"

TEACHER_SECTIONS = {
    "arista@islorg.com": ["English.03", "English.04"],
    "nisha@islorg.com": ["Economics.02"],
}

ECONOMICS_CLASS = "Economics.02_AY2025-26_Supply Function_16 April"
POETRY_CLASS = "English.03_AY2025-26_Poem Refrain_05 May"
PROSE_CLASS = "English.04_AY2025-26_Prose Passage_06 May"

PAIRS = [
    ("2401", "Aarav Shah", ECONOMICS_CLASS),
    ("2402", "Bhavna Rao", POETRY_CLASS),
    ("2403", "Chirag Jain", PROSE_CLASS),
    ("2405", "Esha Patel", ECONOMICS_CLASS),
    ("2405", "Esha Patel", PROSE_CLASS),
]


class _FakeArray:
    def __init__(self, data: list[float]) -> None:
        self._data = data

    def tolist(self) -> list[float]:
        return self._data


def make_embedder() -> QueryEmbedder:
    embedder = QueryEmbedder(DEFAULT_EMBEDDING_MODEL)
    embedder._model = SimpleNamespace(encode=lambda query: _FakeArray([0.1, 0.2]))
    return embedder


def make_search_result(student_id: str, student_name: str = "Bhavna Rao") -> SearchResult:
    return SearchResult(
        chunk_id=f"{student_id}-c1",
        chunk_type="spoken",
        class_name=POETRY_CLASS,
        distance=0.25,
        end_time=12.0,
        metadata={},
        speaker="student",
        start_time=4.0,
        student_id=student_id,
        student_name=student_name,
        text="I said the poem uses a refrain.",
    )


class FakeStore:
    def __init__(self, *, pairs: Sequence[tuple[str, str, str]] = ()) -> None:
        self._pairs = list(pairs) or list(PAIRS)
        self.searched_student_ids: list[str] = []

    def search(
        self,
        query_embedding: list[float],
        student_id: str,
        top_k: int = 5,
        chunk_types: Sequence[str] | None = None,
        class_name: str | None = None,
    ) -> list[SearchResult]:
        self.searched_student_ids.append(student_id)
        return [make_search_result(student_id)]

    def search_lexical(
        self,
        query_text: str,
        *,
        student_id: str,
        chunk_types: Sequence[str] | None = None,
        limit: int = 25,
        class_name: str | None = None,
    ) -> list[SearchResult]:
        return []

    def get_student_chunks(self, student_id: str) -> list[SearchResult]:
        return [make_search_result(student_id)]

    def list_student_class_pairs(self) -> list[tuple[str, str, str]]:
        return list(self._pairs)


class FakeChatBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[Sequence[PromptMessage], str]] = []

    def generate(self, *, messages: Sequence[PromptMessage], model: str) -> str:
        self.calls.append((messages, model))
        return "grounded answer"


class RateLimitBackend:
    def generate(self, *, messages: Sequence[PromptMessage], model: str) -> str:
        from groq import RateLimitError

        raise RateLimitError(
            "rate limit exceeded",
            response=httpx.Response(
                status_code=429, request=httpx.Request("POST", "https://api.groq.com/v1")
            ),
            body=None,
        )


class ExplodingBackend:
    def generate(self, *, messages: Sequence[PromptMessage], model: str) -> str:
        raise ChatError("internal db dsn postgresql://user:sup3rsecret@host/db leaked")


def build_client(
    *,
    store: Any | None = None,
    chat_backend: Any | None = None,
    teacher_sections: dict[str, list[str]] | None = None,
) -> TestClient:
    app = create_app(
        store=store if store is not None else FakeStore(),
        embedder=make_embedder(),
        chat_backend=chat_backend if chat_backend is not None else FakeChatBackend(),
        teacher_sections=dict(TEACHER_SECTIONS if teacher_sections is None else teacher_sections),
        service_token=TOKEN,
    )
    return TestClient(app)


@pytest.fixture
def client() -> Iterator[TestClient]:
    with build_client() as c:
        yield c


def auth() -> dict[str, str]:
    return {SERVICE_TOKEN_HEADER: TOKEN}


def ask_body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "email": "bhavna_2402@islorg.com",
        "lms_role": "student",
        "question": "What did I say about the poem?",
    }
    body.update(overrides)
    return body


def test_ask_without_token_is_401(client: TestClient) -> None:
    assert client.post("/ask", json=ask_body()).status_code == 401


def test_ask_with_wrong_token_is_401(client: TestClient) -> None:
    response = client.post("/ask", json=ask_body(), headers={SERVICE_TOKEN_HEADER: "nope"})
    assert response.status_code == 401


def test_students_without_token_is_401(client: TestClient) -> None:
    response = client.get("/students", params={"email": "arista@islorg.com", "lms_role": "teacher"})
    assert response.status_code == 401


def test_sessions_without_token_is_401(client: TestClient) -> None:
    response = client.get(
        "/sessions", params={"email": "bhavna_2402@islorg.com", "lms_role": "student"}
    )
    assert response.status_code == 401


def test_token_comparison_is_constant_time(monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.api

    calls: list[tuple[bytes, bytes]] = []
    real = scripts.api.hmac.compare_digest

    def spy(a: Any, b: Any) -> bool:
        calls.append((a, b))
        return bool(real(a, b))

    monkeypatch.setattr(scripts.api.hmac, "compare_digest", spy)
    with build_client() as c:
        c.post("/ask", json=ask_body(), headers={SERVICE_TOKEN_HEADER: "nope"})
    assert calls == [(b"nope", TOKEN.encode("utf-8"))]


def test_healthz_needs_no_token_and_leaks_nothing(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_empty_service_token_refuses_to_start(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CHATBOT_SERVICE_TOKEN", raising=False)
    with pytest.raises(ChatError, match="CHATBOT_SERVICE_TOKEN is empty"):
        create_app(store=FakeStore(), embedder=make_embedder(), chat_backend=FakeChatBackend())


def test_blank_service_token_refuses_to_start(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHATBOT_SERVICE_TOKEN", "   ")
    with pytest.raises(ChatError, match="CHATBOT_SERVICE_TOKEN is empty"):
        create_app(store=FakeStore(), embedder=make_embedder(), chat_backend=FakeChatBackend())


def test_env_service_token_is_used_when_not_injected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHATBOT_SERVICE_TOKEN", "from-env")
    app = create_app(
        store=FakeStore(),
        embedder=make_embedder(),
        chat_backend=FakeChatBackend(),
        teacher_sections=dict(TEACHER_SECTIONS),
    )
    with TestClient(app) as c:
        assert c.post("/ask", json=ask_body(), headers={SERVICE_TOKEN_HEADER: "from-env"}).status_code == 200
        assert c.post("/ask", json=ask_body(), headers=auth()).status_code == 401


def test_student_is_answered_as_themselves(client: TestClient) -> None:
    response = client.post("/ask", json=ask_body(), headers=auth())
    assert response.status_code == 200
    payload = response.json()
    assert payload["student_id"] == "2402"
    assert payload["student_name"] == "Bhavna Rao"
    assert payload["answer"] == "grounded answer"
    assert payload["answer_source"] == "groq"


def test_student_cannot_smuggle_another_student_id() -> None:
    store = FakeStore()
    with build_client(store=store) as c:
        response = c.post("/ask", json=ask_body(student_id="2401"), headers=auth())

    assert response.status_code == 200
    payload = response.json()
    assert payload["student_id"] == "2402"
    assert payload["student_name"] == "Bhavna Rao"
    assert store.searched_student_ids == ["2402"]
    assert "2401" not in store.searched_student_ids


def test_student_smuggling_their_teachers_id_is_still_themselves() -> None:
    store = FakeStore()
    with build_client(store=store) as c:
        response = c.post(
            "/ask",
            json=ask_body(email="esha_2405@islorg.com", student_id="2403"),
            headers=auth(),
        )
    assert response.json()["student_id"] == "2405"
    assert store.searched_student_ids == ["2405"]


def test_teacher_asks_for_their_own_sections_student(client: TestClient) -> None:
    response = client.post(
        "/ask",
        json=ask_body(email="arista@islorg.com", lms_role="teacher", student_id="2402"),
        headers=auth(),
    )
    assert response.status_code == 200
    assert response.json()["student_id"] == "2402"


def test_teacher_asking_for_another_sections_student_is_403() -> None:
    store = FakeStore()
    with build_client(store=store) as c:
        response = c.post(
            "/ask",
            json=ask_body(email="arista@islorg.com", lms_role="teacher", student_id="2401"),
            headers=auth(),
        )
    assert response.status_code == 403
    assert store.searched_student_ids == []


def test_teacher_can_reach_a_dual_subject_student(client: TestClient) -> None:
    for email in ("arista@islorg.com", "nisha@islorg.com"):
        response = client.post(
            "/ask",
            json=ask_body(email=email, lms_role="teacher", student_id="2405"),
            headers=auth(),
        )
        assert response.status_code == 200
        assert response.json()["student_id"] == "2405"


def test_teacher_without_student_id_is_400(client: TestClient) -> None:
    response = client.post(
        "/ask",
        json=ask_body(email="arista@islorg.com", lms_role="teacher"),
        headers=auth(),
    )
    assert response.status_code == 400


def test_observer_is_403(client: TestClient) -> None:
    response = client.post(
        "/ask", json=ask_body(email="someone_2402@islorg.com", lms_role="observer"), headers=auth()
    )
    assert response.status_code == 403


def test_unknown_teacher_email_is_403(client: TestClient) -> None:
    response = client.post(
        "/ask",
        json=ask_body(email="newteacher@islorg.com", lms_role="teacher", student_id="2402"),
        headers=auth(),
    )
    assert response.status_code == 403


def test_roll_less_student_email_is_403(client: TestClient) -> None:
    response = client.post(
        "/ask", json=ask_body(email="arista@islorg.com", lms_role="student"), headers=auth()
    )
    assert response.status_code == 403


def test_ask_returns_the_grounding_sources(client: TestClient) -> None:
    payload = client.post("/ask", json=ask_body(), headers=auth()).json()
    assert len(payload["sources"]) == 1
    source = payload["sources"][0]
    assert source["rank"] == 1
    assert source["chunk_type"] == "spoken"
    assert source["score"] == pytest.approx(1.0 / 1.25)
    assert source["start"] == pytest.approx(4.0)
    assert source["end"] == pytest.approx(12.0)
    assert source["text"] == "I said the poem uses a refrain."
    assert source["speaker"]


def test_ask_passes_class_name_scope(client: TestClient) -> None:
    response = client.post(
        "/ask",
        json=ask_body(class_name=POETRY_CLASS),
        headers=auth(),
    )
    assert response.status_code == 200


def test_groq_rate_limit_maps_to_503_with_retry_after() -> None:
    with build_client(chat_backend=RateLimitBackend()) as c:
        response = c.post("/ask", json=ask_body(), headers=auth())
    assert response.status_code == 503
    assert response.json()["detail"] == {"retry_after_seconds": 20}


def test_other_failures_map_to_502_without_leaking_internals() -> None:
    with build_client(chat_backend=ExplodingBackend()) as c:
        response = c.post("/ask", json=ask_body(), headers=auth())
    assert response.status_code == 502
    body = response.text
    assert "sup3rsecret" not in body
    assert "postgresql://" not in body
    assert response.json()["detail"] == "Upstream answer generation failed."


def test_students_lists_exactly_the_teachers_sections(client: TestClient) -> None:
    response = client.get(
        "/students", params={"email": "arista@islorg.com", "lms_role": "teacher"}, headers=auth()
    )
    assert response.status_code == 200
    payload = response.json()
    assert [s["student_id"] for s in payload] == ["2402", "2403", "2405"]
    assert [s["student_name"] for s in payload] == ["Bhavna Rao", "Chirag Jain", "Esha Patel"]


def test_students_excludes_other_sections(client: TestClient) -> None:
    payload = client.get(
        "/students", params={"email": "arista@islorg.com", "lms_role": "teacher"}, headers=auth()
    ).json()
    assert "2401" not in [s["student_id"] for s in payload]


def test_students_reports_sections_per_student(client: TestClient) -> None:
    payload = client.get(
        "/students", params={"email": "nisha@islorg.com", "lms_role": "teacher"}, headers=auth()
    ).json()
    by_id = {s["student_id"]: s for s in payload}
    assert set(by_id) == {"2401", "2405"}
    assert by_id["2401"]["sections"] == ["Economics.02"]
    assert by_id["2405"]["sections"] == ["Economics.02"]


def test_students_does_not_reveal_other_sections_of_a_dual_subject_student(
    client: TestClient,
) -> None:
    economics = client.get(
        "/students", params={"email": "nisha@islorg.com", "lms_role": "teacher"}, headers=auth()
    ).json()
    english = client.get(
        "/students", params={"email": "arista@islorg.com", "lms_role": "teacher"}, headers=auth()
    ).json()
    assert {s["student_id"]: s["sections"] for s in economics}["2405"] == ["Economics.02"]
    assert {s["student_id"]: s["sections"] for s in english}["2405"] == ["English.04"]


def test_students_role_is_403(client: TestClient) -> None:
    response = client.get(
        "/students",
        params={"email": "bhavna_2402@islorg.com", "lms_role": "student"},
        headers=auth(),
    )
    assert response.status_code == 403


def test_students_unknown_teacher_is_403(client: TestClient) -> None:
    response = client.get(
        "/students", params={"email": "newteacher@islorg.com", "lms_role": "teacher"}, headers=auth()
    )
    assert response.status_code == 403


def test_sessions_are_scoped_and_labeled(client: TestClient) -> None:
    response = client.get(
        "/sessions",
        params={"email": "esha_2405@islorg.com", "lms_role": "student"},
        headers=auth(),
    )
    assert response.status_code == 200
    payload = response.json()
    assert [s["class_name"] for s in payload] == [ECONOMICS_CLASS, PROSE_CLASS]
    assert payload[0]["label"] == "Supply Function — 16 April"
    assert payload[1]["label"] == "Prose Passage — 06 May"


def test_sessions_ignore_a_students_smuggled_student_id(client: TestClient) -> None:
    payload = client.get(
        "/sessions",
        params={
            "email": "bhavna_2402@islorg.com",
            "lms_role": "student",
            "student_id": "2401",
        },
        headers=auth(),
    ).json()
    assert [s["class_name"] for s in payload] == [POETRY_CLASS]


def test_sessions_for_a_teachers_student(client: TestClient) -> None:
    payload = client.get(
        "/sessions",
        params={"email": "arista@islorg.com", "lms_role": "teacher", "student_id": "2402"},
        headers=auth(),
    ).json()
    assert [s["class_name"] for s in payload] == [POETRY_CLASS]


def test_sessions_for_another_sections_student_is_403(client: TestClient) -> None:
    response = client.get(
        "/sessions",
        params={"email": "arista@islorg.com", "lms_role": "teacher", "student_id": "2401"},
        headers=auth(),
    )
    assert response.status_code == 403
