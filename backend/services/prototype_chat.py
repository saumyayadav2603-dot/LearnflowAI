"""Prototype-only uploaded-document authorization and citation adapter."""
import asyncio
import os
import re
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from pydantic import ValidationError

from ..chat_service import AuthorizationError
from ..contracts import Citation, RetrievalHit, SourceType
from .document_retriever import BedrockDocumentRetriever

DOCUMENT_SOURCE_ID_PATTERN = re.compile(r"^[0-9a-f]{32}\.(pdf|pptx)$")


class DemoAuthorizationConfigurationError(RuntimeError):
    """Required prototype document authorization configuration is unavailable."""


class DemoUploadedDocumentAuthorizer:
    """Authorize only locally uploaded documents that exist in the configured S3 bucket.

    This is not user authentication or session ownership verification.
    """

    def __init__(
        self,
        upload_directory: Path,
        bucket_name: str,
        *,
        s3_client: Any,
    ) -> None:
        self.upload_directory = upload_directory.resolve()
        self.bucket_name = bucket_name
        self.s3_client = s3_client

    @classmethod
    def from_env(cls, upload_directory: Path) -> "DemoUploadedDocumentAuthorizer":
        bucket_name = os.getenv("S3_BUCKET_NAME", "").strip()
        region = os.getenv("AWS_REGION", "").strip()
        missing = [
            name
            for name, value in (
                ("S3_BUCKET_NAME", bucket_name),
                ("AWS_REGION", region),
            )
            if not value
        ]
        if missing:
            raise DemoAuthorizationConfigurationError(
                f"Prototype document authorization configuration is missing: {', '.join(missing)}."
            )
        return cls(
            upload_directory,
            bucket_name,
            s3_client=boto3.client("s3", region_name=region),
        )

    async def ensure_allowed(
        self,
        session_id: str,
        source_ids: list[str],
    ) -> None:
        # session_id is intentionally not interpreted as an identity or credential.
        if not source_ids:
            raise AuthorizationError("No uploaded document was requested.")
        for source_id in dict.fromkeys(source_ids):
            if DOCUMENT_SOURCE_ID_PATTERN.fullmatch(source_id) is None:
                raise AuthorizationError("Requested source is not an uploaded document.")

            local_path = (self.upload_directory / source_id).resolve()
            if local_path.parent != self.upload_directory or not local_path.is_file():
                raise AuthorizationError("Requested source is not an uploaded document.")

            try:
                await asyncio.to_thread(
                    self.s3_client.head_object,
                    Bucket=self.bucket_name,
                    Key=source_id,
                )
            except ClientError as exc:
                error_code = exc.response.get("Error", {}).get("Code")
                if error_code in {"404", "NoSuchKey", "NotFound"}:
                    raise AuthorizationError(
                        "Requested source is not available in the configured bucket."
                    ) from exc
                raise RuntimeError(
                    "Could not verify the uploaded document in the configured bucket."
                ) from exc
            except BotoCoreError as exc:
                raise RuntimeError(
                    "Could not verify the uploaded document in the configured bucket."
                ) from exc


class DocumentChatRetriever:
    """Adapt Bedrock document chunks to the chat service's strict hit contract."""

    def __init__(self, retriever: BedrockDocumentRetriever) -> None:
        self.retriever = retriever

    async def retrieve_documents(
        self,
        question: str,
        allowed_source_ids: list[str],
    ) -> list[RetrievalHit]:
        chunks = await self.retriever.retrieve_documents(
            question,
            allowed_source_ids,
        )
        hits = []
        for chunk in chunks:
            if chunk.source_id not in allowed_source_ids:
                continue
            try:
                if chunk.source_type == SourceType.PDF:
                    if chunk.page is None:
                        continue
                    citation = Citation(
                        source_id=chunk.source_id,
                        source_type=chunk.source_type,
                        chunk_id=chunk.chunk_id,
                        page=chunk.page,
                    )
                elif chunk.source_type == SourceType.PPTX:
                    if chunk.slide is None:
                        continue
                    citation = Citation(
                        source_id=chunk.source_id,
                        source_type=chunk.source_type,
                        chunk_id=chunk.chunk_id,
                        slide=chunk.slide,
                    )
                else:
                    continue

                hits.append(
                    RetrievalHit(
                        citation=citation,
                        text=chunk.text,
                        score=chunk.score,
                    )
                )
            except ValidationError:
                continue
        return hits

    async def retrieve_video(
        self,
        question: str,
        video_source_id: str,
        playback_seconds: float,
    ) -> list[RetrievalHit]:
        raise RuntimeError("Video retrieval is not enabled in the document demo.")
