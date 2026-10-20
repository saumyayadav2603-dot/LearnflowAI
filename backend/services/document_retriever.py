"""Document-only retrieval from the configured Amazon Bedrock Knowledge Base."""
import asyncio
import hashlib
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Mapping

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from ..contracts import SourceType

DOCUMENT_SOURCE_ID_PATTERN = re.compile(r"^[0-9a-f]{32}\.(pdf|pptx)$")
SOURCE_URI_METADATA_KEY = "x-amz-bedrock-kb-source-uri"
CHUNK_ID_METADATA_KEY = "x-amz-bedrock-kb-chunk-id"
PAGE_METADATA_KEY = "x-amz-bedrock-kb-document-page-number"
SLIDE_METADATA_KEYS = ("slide_number", "slideNumber", "slide")
MAX_SOURCE_IDS = 30
MAX_RESULTS = 8


class BedrockRetrievalError(RuntimeError):
    """Bedrock retrieval configuration or service failure."""


@dataclass(frozen=True)
class RetrievedDocumentChunk:
    source_id: str
    source_type: SourceType
    chunk_id: str
    text: str
    score: float | None
    page: int | None
    slide: int | None
    document_id: str | None
    metadata: Mapping[str, Any]


class BedrockDocumentRetriever:
    """Retrieve chunks only from upload-generated PDF and PPTX object keys.

    This validates document ID shape and constrains the Bedrock metadata filter;
    it does not verify learner/session ownership.
    """

    def __init__(
        self,
        knowledge_base_id: str,
        bucket_name: str,
        region: str,
        *,
        client: Any | None = None,
    ) -> None:
        self.knowledge_base_id = knowledge_base_id
        self.bucket_name = bucket_name
        self.client = client or boto3.client(
            "bedrock-agent-runtime",
            region_name=region,
        )

    @classmethod
    def from_env(cls) -> "BedrockDocumentRetriever":
        knowledge_base_id = os.getenv("BEDROCK_KB_ID", "").strip()
        bucket_name = os.getenv("S3_BUCKET_NAME", "").strip()
        region = os.getenv("AWS_REGION", "").strip()
        missing = [
            name
            for name, value in (
                ("BEDROCK_KB_ID", knowledge_base_id),
                ("S3_BUCKET_NAME", bucket_name),
                ("AWS_REGION", region),
            )
            if not value
        ]
        if missing:
            raise BedrockRetrievalError(
                f"Bedrock retrieval configuration is missing: {', '.join(missing)}."
            )
        if "/" in bucket_name or "\\" in bucket_name:
            raise BedrockRetrievalError("S3_BUCKET_NAME is invalid.")
        return cls(knowledge_base_id, bucket_name, region)

    async def retrieve_documents(
        self,
        question: str,
        allowed_source_ids: list[str],
    ) -> list[RetrievedDocumentChunk]:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must not be empty.")
        if (
            not isinstance(allowed_source_ids, list)
            or not 1 <= len(allowed_source_ids) <= MAX_SOURCE_IDS
            or any(
                not isinstance(source_id, str)
                or DOCUMENT_SOURCE_ID_PATTERN.fullmatch(source_id) is None
                for source_id in allowed_source_ids
            )
        ):
            raise ValueError("allowed_source_ids contains an invalid document ID.")

        source_uris = {
            f"s3://{self.bucket_name}/{source_id}": source_id
            for source_id in dict.fromkeys(allowed_source_ids)
        }
        try:
            response = await asyncio.to_thread(
                self.client.retrieve,
                knowledgeBaseId=self.knowledge_base_id,
                retrievalQuery={"text": question.strip()},
                retrievalConfiguration={
                    "vectorSearchConfiguration": {
                        "numberOfResults": MAX_RESULTS,
                        "filter": {
                            "in": {
                                "key": SOURCE_URI_METADATA_KEY,
                                "value": list(source_uris),
                            }
                        },
                    }
                },
            )
        except (BotoCoreError, ClientError) as exc:
            raise BedrockRetrievalError(
                "Bedrock Knowledge Base retrieval failed."
            ) from exc

        if not isinstance(response, Mapping):
            return []
        results = response.get("retrievalResults", [])
        if not isinstance(results, list):
            return []

        chunks = []
        for result in results:
            if not isinstance(result, Mapping):
                continue
            chunk = self._map_result(result, source_uris)
            if chunk is not None:
                chunks.append(chunk)
        return chunks

    @classmethod
    def _map_result(
        cls,
        result: Mapping[str, Any],
        source_uris: Mapping[str, str],
    ) -> RetrievedDocumentChunk | None:
        location = result.get("location")
        if not isinstance(location, Mapping):
            return None
        s3_location = location.get("s3Location")
        if not isinstance(s3_location, Mapping):
            return None

        uri = s3_location.get("uri")
        source_id = source_uris.get(uri) if isinstance(uri, str) else None
        content = result.get("content")
        if not isinstance(content, Mapping):
            return None
        text = content.get("text")
        if not source_id or not isinstance(text, str) or not text.strip():
            return None

        raw_metadata = result.get("metadata", {})
        metadata = dict(raw_metadata) if isinstance(raw_metadata, Mapping) else {}
        extension = source_id.rsplit(".", 1)[-1]
        source_type = SourceType.PDF if extension == "pdf" else SourceType.PPTX
        document_page = cls._positive_int(metadata.get(PAGE_METADATA_KEY))
        page = document_page if source_type == SourceType.PDF else None
        slide = None
        if source_type == SourceType.PPTX:
            slide = next(
                (
                    parsed
                    for key in SLIDE_METADATA_KEYS
                    if (parsed := cls._positive_int(metadata.get(key))) is not None
                ),
                document_page,
            )

        raw_chunk_id = metadata.get(CHUNK_ID_METADATA_KEY)
        if isinstance(raw_chunk_id, str) and raw_chunk_id.strip():
            chunk_id = raw_chunk_id.strip()
        else:
            digest_input = f"{source_id}\0{page or ''}\0{slide or ''}\0{text}"
            chunk_id = f"derived-{hashlib.sha256(digest_input.encode('utf-8')).hexdigest()}"

        score_value = result.get("score")
        score = None
        if isinstance(score_value, (int, float)) and not isinstance(score_value, bool):
            try:
                parsed_score = float(score_value)
            except OverflowError:
                parsed_score = math.nan
            if math.isfinite(parsed_score) and parsed_score >= 0:
                score = parsed_score
        raw_document_id = result.get("documentId")
        document_id = (
            raw_document_id.strip()
            if isinstance(raw_document_id, str) and raw_document_id.strip()
            else None
        )

        return RetrievedDocumentChunk(
            source_id=source_id,
            source_type=source_type,
            chunk_id=chunk_id,
            text=text.strip(),
            score=score,
            page=page,
            slide=slide,
            document_id=document_id,
            metadata=metadata,
        )

    @staticmethod
    def _positive_int(value: Any) -> int | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            parsed = value
        elif isinstance(value, float) and value.is_integer():
            parsed = int(value)
        elif isinstance(value, str) and value.isdecimal():
            parsed = int(value)
        else:
            return None
        return parsed if parsed >= 1 else None
