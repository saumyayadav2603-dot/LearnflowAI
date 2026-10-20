"""Grounded chat boundaries, authorization and API fail-closed tests."""
import asyncio

import pytest
from fastapi.testclient import TestClient

from backend.contracts import ChatRequest, Citation, RetrievalHit
from backend.chat_service import (
    AuthorizationError,
    ChatUnavailableError,
    GroundedChatService,
    REFUSAL,
)
from backend.main import app


class Authorizer:
    def __init__(self, fail=False):
        self.fail = fail
        self.called = False

    async def ensure_allowed(self, session_id, source_ids):
        self.called = True
        if self.fail or "private" in source_ids:
            raise AuthorizationError("Private source")


class Retriever:
    def __init__(self, fail=False):
        self.fail = fail

    async def retrieve_documents(self, question, allowed_source_ids):
        if self.fail:
            raise RuntimeError("Retriever offline")
        return [
            RetrievalHit(citation=Citation(
                source_id="doc1", source_type="pdf", chunk_id="d1", page=3
            ), text="Covered material from page three"),
            RetrievalHit(citation=Citation(
                source_id="private", source_type="pdf", chunk_id="private", page=7
            ), text="SECRET"),
            RetrievalHit(citation=Citation(
                source_id="video1", source_type="video", chunk_id="fake-video",
                start_seconds=1, end_seconds=2
            ), text="Invalid video evidence in document channel"),
        ]

    async def retrieve_video(self, question, video_source_id, playback_seconds):
        return [
            RetrievalHit(citation=Citation(
                source_id="video1", source_type="video", chunk_id="v1",
                start_seconds=5, end_seconds=10
            ), text="Earlier lecture segment"),
            RetrievalHit(citation=Citation(
                source_id="video1", source_type="video", chunk_id="v2",
                start_seconds=25, end_seconds=35
            ), text="CROSSES CURRENT CUTOFF"),
            RetrievalHit(citation=Citation(
                source_id="video1", source_type="video", chunk_id="v3",
                start_seconds=50, end_seconds=60
            ), text="Future lecture segment"),
            RetrievalHit(citation=Citation(
                source_id="other-video", source_type="video", chunk_id="v4",
                start_seconds=1, end_seconds=3
            ), text="Different video"),
        ]


class Provider:
    def __init__(self, fail=False, result="Answer based on supplied lecture evidence"):
        self.evidence = None
        self.fail = fail
        self.result = result

    async def answer(self, question, evidence):
        self.evidence = evidence
        if self.fail:
            raise RuntimeError("Provider unavailable")
        return self.result


def request(playback_seconds=30, allowed_source_ids=None, video=True):
    return ChatRequest(
        session_id="s1",
        question="What was discussed?",
        allowed_source_ids=allowed_source_ids or ["doc1", "video1"],
        **({"video_source_id": "video1", "playback_seconds": playback_seconds} if video else {}),
    )


def run(service, payload):
    return asyncio.run(service.answer(payload))


def test_excludes_unpermitted_future_and_cross_boundary_video():
    provider = Provider()
    result = run(
        GroundedChatService(Authorizer(), Retriever(), provider), request()
    )
    assert result.grounded is True
    assert [c.chunk_id for c in result.citations] == ["d1", "v1"]
    assert provider.evidence == [
        "Covered material from page three", "Earlier lecture segment"
    ]


def test_backward_seek_disallows_previously_visible_video():
    provider = Provider()
    result = run(
        GroundedChatService(Authorizer(), Retriever(), provider),
        request(playback_seconds=2, allowed_source_ids=["video1"]),
    )
    assert result.grounded is False
    assert result.answer == REFUSAL
    assert not result.citations
    assert provider.evidence is None


def test_no_video_request_never_accepts_transcript_hits():
    provider = Provider()
    result = run(
        GroundedChatService(Authorizer(), Retriever(), provider),
        request(video=False, allowed_source_ids=["doc1", "video1"]),
    )
    assert [c.chunk_id for c in result.citations] == ["d1"]
    assert provider.evidence == ["Covered material from page three"]


def test_rejects_unauthorized_request_before_retrieval():
    auth = Authorizer()
    provider = Provider()
    with pytest.raises(AuthorizationError):
        run(
            GroundedChatService(auth, Retriever(), provider),
            request(video=False, allowed_source_ids=["private"]),
        )
    assert auth.called
    assert provider.evidence is None


def test_failed_retrieval_is_not_treated_as_no_evidence():
    with pytest.raises(ChatUnavailableError, match="retrieval"):
        run(
            GroundedChatService(Authorizer(), Retriever(fail=True), Provider()),
            request(),
        )


def test_failed_provider_returns_unavailable_error():
    with pytest.raises(ChatUnavailableError, match="generation"):
        run(
            GroundedChatService(Authorizer(), Retriever(), Provider(fail=True)),
            request(),
        )


def test_provider_abstention_returns_ungrounded_response_without_citations():
    result = run(
        GroundedChatService(
            Authorizer(),
            Retriever(),
            Provider(result=None),
        ),
        request(video=False, allowed_source_ids=["doc1"]),
    )

    assert result.grounded is False
    assert result.answer == REFUSAL
    assert result.citations == []


def test_chat_api_is_fail_closed_when_service_is_not_configured(monkeypatch):
    monkeypatch.setattr(app.state, "chat_service", None, raising=False)
    client = TestClient(app, client=("127.0.0.1", 8001))
    response = client.post("/chat", json=request().model_dump(mode="json"))
    assert response.status_code == 503
    assert response.json()["detail"] == "Chat integrations are not configured."


def test_chat_api_rejects_missing_playback_timestamp():
    with TestClient(app, client=("127.0.0.1", 8001)) as client:
        response = client.post(
            "/chat", json={
                "session_id": "s1", "question": "What?", "allowed_source_ids": ["video1"],
                "video_source_id": "video1"
            }
        )
    assert response.status_code == 422
