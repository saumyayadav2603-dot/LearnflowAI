import asyncio

import pytest
from botocore.exceptions import ClientError

from backend.services.document_retriever import (
    BedrockDocumentRetriever,
    BedrockRetrievalError,
    RetrievedDocumentChunk,
    SOURCE_URI_METADATA_KEY,
)


PDF_SOURCE_ID = "1461030a83ea47e78bd5e6f00aef5718.pdf"
PPTX_SOURCE_ID = "187453729b4349b191df275e7b65ea75.pptx"


class FakeBedrockClient:
    def __init__(self, results=None, error=None):
        self.results = results or []
        self.error = error
        self.response = None
        self.request = None

    def retrieve(self, **kwargs):
        self.request = kwargs
        if self.error:
            raise self.error
        if self.response is not None:
            return self.response
        return {"retrievalResults": self.results}


def result_for(source_id, *, text="Retrieved lesson text", metadata=None, score=0.91):
    return {
        "content": {"text": text},
        "location": {
            "s3Location": {"uri": f"s3://lesson-bucket/{source_id}"}
        },
        "metadata": metadata or {},
        "documentId": f"bedrock-document-{source_id}",
        "score": score,
    }


def make_retriever(client):
    return BedrockDocumentRetriever(
        "test-knowledge-base",
        "lesson-bucket",
        "ap-southeast-2",
        client=client,
    )


def test_retrieves_pdf_chunks_filtered_to_validated_source_ids():
    client = FakeBedrockClient([
        result_for(
            PDF_SOURCE_ID,
            metadata={
                "x-amz-bedrock-kb-document-page-number": 3,
                "x-amz-bedrock-kb-chunk-id": "pdf-chunk-3",
                "course": "lesson",
            },
        )
    ])
    retriever = make_retriever(client)

    chunks = asyncio.run(
        retriever.retrieve_documents("  What is covered?  ", [PDF_SOURCE_ID])
    )

    assert len(chunks) == 1
    chunk = chunks[0]
    assert isinstance(chunk, RetrievedDocumentChunk)
    assert chunk.source_id == PDF_SOURCE_ID
    assert chunk.source_type == "pdf"
    assert chunk.chunk_id == "pdf-chunk-3"
    assert chunk.page == 3
    assert chunk.slide is None
    assert chunk.text == "Retrieved lesson text"
    assert chunk.score == 0.91
    assert chunk.metadata["course"] == "lesson"

    request = client.request
    assert request["knowledgeBaseId"] == "test-knowledge-base"
    assert request["retrievalQuery"] == {"text": "What is covered?"}
    search = request["retrievalConfiguration"]["vectorSearchConfiguration"]
    assert search["numberOfResults"] == 8
    assert search["filter"]["in"]["key"] == SOURCE_URI_METADATA_KEY
    assert search["filter"]["in"]["value"] == [
        f"s3://lesson-bucket/{PDF_SOURCE_ID}"
    ]


def test_maps_powerpoint_slide_metadata():
    client = FakeBedrockClient([
        result_for(
            PPTX_SOURCE_ID,
            metadata={"slide_number": "5"},
        )
    ])

    chunks = asyncio.run(
        make_retriever(client).retrieve_documents("Explain this slide", [PPTX_SOURCE_ID])
    )

    assert len(chunks) == 1
    assert chunks[0].source_type == "pptx"
    assert chunks[0].slide == 5
    assert chunks[0].page is None
    assert chunks[0].chunk_id.startswith("derived-")
    assert chunks[0].document_id == f"bedrock-document-{PPTX_SOURCE_ID}"


def test_preserves_text_when_optional_page_metadata_is_missing():
    client = FakeBedrockClient([result_for(PDF_SOURCE_ID, metadata={"other": "value"})])

    chunks = asyncio.run(
        make_retriever(client).retrieve_documents("Question", [PDF_SOURCE_ID])
    )

    assert len(chunks) == 1
    assert chunks[0].text == "Retrieved lesson text"
    assert chunks[0].page is None
    assert chunks[0].metadata == {"other": "value"}


@pytest.mark.parametrize(
    "source_ids",
    [
        ["s3://attacker-bucket/private.pdf"],
        ["..\\private.pdf"],
        ["not-a-document-id"],
        [PDF_SOURCE_ID, "s3://attacker-bucket/private.pdf"],
        [],
        ["a" * 32 + ".ppt"],
    ],
)
def test_rejects_invalid_or_arbitrary_source_ids_before_aws_call(source_ids):
    client = FakeBedrockClient()

    with pytest.raises(ValueError, match="invalid document ID"):
        asyncio.run(
            make_retriever(client).retrieve_documents("Question", source_ids)
        )

    assert client.request is None


def test_empty_results_return_empty_list():
    chunks = asyncio.run(
        make_retriever(FakeBedrockClient()).retrieve_documents(
            "Question", [PDF_SOURCE_ID]
        )
    )
    assert chunks == []


@pytest.mark.parametrize("response", [None, {}, {"retrievalResults": None}])
def test_malformed_or_empty_bedrock_response_returns_empty_list(response):
    client = FakeBedrockClient()
    client.response = response

    chunks = asyncio.run(
        make_retriever(client).retrieve_documents("Question", [PDF_SOURCE_ID])
    )

    assert chunks == []


def test_discards_results_with_foreign_uri_or_incomplete_text():
    client = FakeBedrockClient([
        result_for("3125e113c0ef45a9ba09350716f167d4.pdf"),
        result_for(PDF_SOURCE_ID, text="  "),
        {"location": {"s3Location": {"uri": f"s3://lesson-bucket/{PDF_SOURCE_ID}"}}},
    ])

    chunks = asyncio.run(
        make_retriever(client).retrieve_documents("Question", [PDF_SOURCE_ID])
    )

    assert chunks == []


def test_invalid_scores_are_omitted():
    client = FakeBedrockClient([
        result_for(PDF_SOURCE_ID, score=float("nan")),
    ])

    chunks = asyncio.run(
        make_retriever(client).retrieve_documents("Question", [PDF_SOURCE_ID])
    )

    assert len(chunks) == 1
    assert chunks[0].score is None


def test_aws_client_error_is_wrapped_without_provider_message():
    client = FakeBedrockClient(error=ClientError(
        {"Error": {"Code": "AccessDenied", "Message": "private detail"}},
        "Retrieve",
    ))

    with pytest.raises(BedrockRetrievalError, match="retrieval failed") as error:
        asyncio.run(
            make_retriever(client).retrieve_documents("Question", [PDF_SOURCE_ID])
        )

    assert "private detail" not in str(error.value)


@pytest.mark.parametrize(
    "environment",
    [
        {},
        {"BEDROCK_KB_ID": "kb-id"},
        {"BEDROCK_KB_ID": "kb-id", "S3_BUCKET_NAME": "lesson-bucket"},
    ],
)
def test_missing_configuration_is_reported(monkeypatch, environment):
    for key in ("BEDROCK_KB_ID", "S3_BUCKET_NAME", "AWS_REGION"):
        monkeypatch.delenv(key, raising=False)
    for key, value in environment.items():
        monkeypatch.setenv(key, value)

    with pytest.raises(BedrockRetrievalError, match="configuration is missing"):
        BedrockDocumentRetriever.from_env()


def test_from_env_uses_configured_region_and_knowledge_base(monkeypatch):
    monkeypatch.setenv("BEDROCK_KB_ID", "configured-kb")
    monkeypatch.setenv("S3_BUCKET_NAME", "lesson-bucket")
    monkeypatch.setenv("AWS_REGION", "ap-southeast-2")
    created = {}

    def fake_client(service_name, *, region_name):
        created["service_name"] = service_name
        created["region_name"] = region_name
        return FakeBedrockClient()

    monkeypatch.setattr("backend.services.document_retriever.boto3.client", fake_client)

    retriever = BedrockDocumentRetriever.from_env()

    assert retriever.knowledge_base_id == "configured-kb"
    assert created == {
        "service_name": "bedrock-agent-runtime",
        "region_name": "ap-southeast-2",
    }


def test_empty_question_is_rejected_without_aws_call():
    client = FakeBedrockClient()

    with pytest.raises(ValueError, match="question must not be empty"):
        asyncio.run(
            make_retriever(client).retrieve_documents("  ", [PDF_SOURCE_ID])
        )

    assert client.request is None
