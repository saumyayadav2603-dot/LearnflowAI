import asyncio

import pytest
from botocore.exceptions import ClientError
from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend import main as backend_main
from backend.chat_service import (
    AuthorizationError,
    ChatUnavailableError,
    GroundedChatService,
)
from backend.contracts import ChatRequest, SourceType
from backend.llm_provider import ProviderConfig, ProviderError
from backend.services.document_retriever import (
    BedrockDocumentRetriever,
    RetrievedDocumentChunk,
)
from backend.services.prototype_chat import (
    DemoAuthorizationConfigurationError,
    DemoUploadedDocumentAuthorizer,
    DocumentChatRetriever,
)

PDF_SOURCE_ID = "1461030a83ea47e78bd5e6f00aef5718.pdf"
OTHER_SOURCE_ID = "3125e113c0ef45a9ba09350716f167d4.pdf"


class FakeBedrockClient:
    def __init__(self, *, fail=False):
        self.fail = fail
        self.calls = []

    def retrieve(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("Bedrock unavailable")
        return {
            "retrievalResults": [
                {
                    "content": {"text": "A retrieved fact from page two."},
                    "location": {
                        "s3Location": {
                            "uri": f"s3://lesson-bucket/{PDF_SOURCE_ID}"
                        }
                    },
                    "metadata": {
                        "x-amz-bedrock-kb-document-page-number": 2,
                        "x-amz-bedrock-kb-chunk-id": "chunk-page-two",
                    },
                    "score": 0.88,
                }
            ]
        }


class FakeProvider:
    def __init__(self, *, fail=False):
        self.fail = fail
        self.evidence = []

    async def answer(self, question, evidence):
        self.evidence = evidence
        if self.fail:
            raise ProviderError("Model provider request failed.")
        return "The retrieved fact is on page two."


class FakeS3Client:
    def __init__(self, existing_keys=None, *, fail=False):
        self.existing_keys = (
            {PDF_SOURCE_ID}
            if existing_keys is None
            else set(existing_keys)
        )
        self.fail = fail
        self.calls = []

    def head_object(self, *, Bucket, Key):
        self.calls.append((Bucket, Key))
        if self.fail:
            raise ClientError(
                {"Error": {"Code": "503", "Message": "Unavailable"}},
                "HeadObject",
            )
        if Key not in self.existing_keys:
            raise ClientError(
                {"Error": {"Code": "404", "Message": "Not found"}},
                "HeadObject",
            )


class TestAuthorizer:
    async def ensure_allowed(self, session_id, source_ids):
        if not source_ids or any(source_id != PDF_SOURCE_ID for source_id in source_ids):
            raise AuthorizationError("Source is not permitted.")


def make_service(*, bedrock_client=None, provider=None):
    bedrock_client = bedrock_client or FakeBedrockClient()
    provider = provider or FakeProvider()
    return (
        GroundedChatService(
            authorizer=TestAuthorizer(),
            retriever=DocumentChatRetriever(
                BedrockDocumentRetriever(
                    "test-kb",
                    "lesson-bucket",
                    "ap-southeast-2",
                    client=bedrock_client,
                )
            ),
            provider=provider,
        ),
        bedrock_client,
        provider,
    )


def make_request(source_ids=None):
    return ChatRequest(
        session_id="demo-session",
        question="Where is the fact?",
        allowed_source_ids=source_ids or [PDF_SOURCE_ID],
    )


def test_demo_authorizer_checks_local_upload_and_configured_bucket(tmp_path):
    (tmp_path / PDF_SOURCE_ID).write_bytes(b"uploaded document")
    s3_client = FakeS3Client()
    authorizer = DemoUploadedDocumentAuthorizer(
        tmp_path,
        "lesson-bucket",
        s3_client=s3_client,
    )

    asyncio.run(
        authorizer.ensure_allowed("unverified-session-label", [PDF_SOURCE_ID])
    )

    assert s3_client.calls == [("lesson-bucket", PDF_SOURCE_ID)]


def test_demo_authorizer_rejects_ids_not_generated_by_upload_service(
    tmp_path,
):
    authorizer = DemoUploadedDocumentAuthorizer(
        tmp_path,
        "lesson-bucket",
        s3_client=FakeS3Client(),
    )

    with pytest.raises(AuthorizationError):
        asyncio.run(authorizer.ensure_allowed("any-session", [OTHER_SOURCE_ID]))

    assert not authorizer.s3_client.calls


def test_demo_authorizer_rejects_source_missing_from_s3(tmp_path):
    (tmp_path / PDF_SOURCE_ID).write_bytes(b"local copy")
    authorizer = DemoUploadedDocumentAuthorizer(
        tmp_path,
        "lesson-bucket",
        s3_client=FakeS3Client(existing_keys=set()),
    )

    with pytest.raises(AuthorizationError):
        asyncio.run(authorizer.ensure_allowed("any-session", [PDF_SOURCE_ID]))


def test_demo_authorizer_surfaces_s3_verification_failure(tmp_path):
    (tmp_path / PDF_SOURCE_ID).write_bytes(b"local copy")
    authorizer = DemoUploadedDocumentAuthorizer(
        tmp_path,
        "lesson-bucket",
        s3_client=FakeS3Client(fail=True),
    )

    with pytest.raises(RuntimeError, match="Could not verify"):
        asyncio.run(authorizer.ensure_allowed("any-session", [PDF_SOURCE_ID]))


def test_demo_authorizer_requires_bucket_configuration(monkeypatch, tmp_path):
    monkeypatch.delenv("S3_BUCKET_NAME", raising=False)
    monkeypatch.delenv("AWS_REGION", raising=False)

    with pytest.raises(
        DemoAuthorizationConfigurationError,
        match="S3_BUCKET_NAME, AWS_REGION",
    ):
        DemoUploadedDocumentAuthorizer.from_env(tmp_path)


def test_demo_authorizer_builds_s3_client_from_existing_configuration(
    monkeypatch, tmp_path
):
    client = FakeS3Client()
    calls = []
    monkeypatch.setenv("S3_BUCKET_NAME", "lesson-bucket")
    monkeypatch.setenv("AWS_REGION", "ap-southeast-2")
    monkeypatch.setattr(
        "backend.services.prototype_chat.boto3.client",
        lambda service, region_name: calls.append((service, region_name)) or client,
    )

    authorizer = DemoUploadedDocumentAuthorizer.from_env(tmp_path)

    assert authorizer.bucket_name == "lesson-bucket"
    assert authorizer.s3_client is client
    assert calls == [("s3", "ap-southeast-2")]


def test_retrieved_chunks_become_cited_chat_evidence():
    service, bedrock_client, provider = make_service()

    response = asyncio.run(service.answer(make_request()))

    assert response.grounded is True
    assert response.answer == "The retrieved fact is on page two."
    assert len(response.citations) == 1
    assert response.citations[0].source_id == PDF_SOURCE_ID
    assert response.citations[0].page == 2
    assert response.citations[0].chunk_id == "chunk-page-two"
    assert provider.evidence == ["A retrieved fact from page two."]
    assert len(bedrock_client.calls) == 1


def test_disallowed_source_is_rejected_before_bedrock_retrieval():
    service, bedrock_client, provider = make_service()

    with pytest.raises(AuthorizationError):
        asyncio.run(service.answer(make_request([OTHER_SOURCE_ID])))

    assert bedrock_client.calls == []
    assert provider.evidence == []


def test_incomplete_citation_metadata_is_not_sent_to_model():
    client = FakeBedrockClient()
    client.retrieve = lambda **kwargs: {
        "retrievalResults": [
            {
                "content": {"text": "Text without page provenance"},
                "location": {
                    "s3Location": {"uri": f"s3://lesson-bucket/{PDF_SOURCE_ID}"}
                },
                "metadata": {},
            }
        ]
    }
    service, _, provider = make_service(bedrock_client=client)

    response = asyncio.run(service.answer(make_request()))

    assert response.grounded is False
    assert response.citations == []
    assert provider.evidence == []


def test_pptx_retrieval_returns_slide_citation():
    class PptxRetriever:
        async def retrieve_documents(self, question, allowed_source_ids):
            return [
                RetrievedDocumentChunk(
                    source_id=OTHER_SOURCE_ID.replace(".pdf", ".pptx"),
                    source_type=SourceType.PPTX,
                    chunk_id="slide-seven",
                    text="A fact from the slide deck.",
                    score=0.9,
                    page=None,
                    slide=7,
                    document_id=None,
                    metadata={},
                )
            ]

    pptx_source_id = OTHER_SOURCE_ID.replace(".pdf", ".pptx")
    hits = asyncio.run(
        DocumentChatRetriever(PptxRetriever()).retrieve_documents(
            "Question", [pptx_source_id]
        )
    )

    assert len(hits) == 1
    assert hits[0].citation.source_type.value == "pptx"
    assert hits[0].citation.source_id == pptx_source_id
    assert hits[0].citation.slide == 7


def test_bedrock_failure_fails_closed_before_generation():
    service, _, provider = make_service(
        bedrock_client=FakeBedrockClient(fail=True)
    )

    with pytest.raises(ChatUnavailableError, match="retrieval"):
        asyncio.run(service.answer(make_request()))

    assert provider.evidence == []


def test_groq_provider_failure_fails_as_chat_unavailable():
    service, _, _ = make_service(provider=FakeProvider(fail=True))

    with pytest.raises(ChatUnavailableError, match="generation"):
        asyncio.run(service.answer(make_request()))


def test_app_startup_wires_document_demo_dependencies_without_cloud_calls(
    monkeypatch, tmp_path
):
    fake_provider = FakeProvider()
    bedrock_client = FakeBedrockClient()
    bedrock_retriever = BedrockDocumentRetriever(
        "test-kb",
        "lesson-bucket",
        "ap-southeast-2",
        client=bedrock_client,
    )
    (tmp_path / PDF_SOURCE_ID).write_bytes(b"uploaded document")
    demo_authorizer = DemoUploadedDocumentAuthorizer(
        tmp_path,
        "lesson-bucket",
        s3_client=FakeS3Client(),
    )
    monkeypatch.setattr(
        backend_main.DemoUploadedDocumentAuthorizer,
        "from_env",
        classmethod(lambda cls, upload_directory: demo_authorizer),
    )
    monkeypatch.setattr(
        backend_main.BedrockDocumentRetriever,
        "from_env",
        classmethod(lambda cls: bedrock_retriever),
    )
    monkeypatch.setattr(
        backend_main.ProviderConfig,
        "from_env",
        classmethod(
            lambda cls: ProviderConfig(
                endpoint="https://api.groq.com/openai/v1/chat/completions",
                model="test-model",
                api_key="test-key",
            )
        ),
    )
    monkeypatch.setattr(
        backend_main,
        "OpenAICompatibleProvider",
        lambda config: fake_provider,
    )
    monkeypatch.setattr(
        backend_main.app.state,
        "chat_service",
        None,
        raising=False,
    )

    with TestClient(
        backend_main.app,
        client=("127.0.0.1", 8001),
    ) as client:
        response = client.post(
            "/chat",
            json=make_request().model_dump(mode="json"),
        )

    assert response.status_code == 200
    assert response.json()["grounded"] is True
    assert response.json()["citations"][0]["page"] == 2


def test_chat_endpoint_rejects_non_private_peer(monkeypatch):
    monkeypatch.setattr(
        backend_main.app.state,
        "chat_service",
        make_service()[0],
        raising=False,
    )

    with TestClient(
        backend_main.app,
        client=("8.8.8.8", 8001),
    ) as client:
        response = client.post(
            "/chat",
            json=make_request().model_dump(mode="json"),
        )

    assert response.status_code == 403
    assert response.json()["detail"].startswith("Chat is restricted")


def test_invalid_chat_request_contract_still_rejects_video_without_timestamp():
    with pytest.raises(ValidationError):
        ChatRequest(
            session_id="demo",
            question="Question",
            allowed_source_ids=[PDF_SOURCE_ID],
            video_source_id=PDF_SOURCE_ID,
        )
