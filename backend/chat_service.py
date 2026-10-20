"""Grounded Q&A coordinator with fail-closed source and playback safeguards.

This module does not provide source authorization or retrieval; those trusted
integrations must be supplied by the application at startup.
"""
from dataclasses import dataclass
from typing import Protocol

from .contracts import ChatRequest, ChatResponse, RetrievalHit, SourceType
from .llm_provider import ModelProvider, ProviderError


class AuthorizationError(PermissionError):
    """Requested learning sources are not accessible to the verified learner."""


class ChatUnavailableError(RuntimeError):
    """Required AI or retrieval dependency is not currently available."""


class SourceAuthorizer(Protocol):
    async def ensure_allowed(self, session_id: str, source_ids: list[str]) -> None:
        """Check trusted authenticated ownership/session association."""


class EvidenceRetriever(Protocol):
    async def retrieve_documents(
        self, question: str, allowed_source_ids: list[str]
    ) -> list[RetrievalHit]: ...

    async def retrieve_video(
        self, question: str, video_source_id: str, playback_seconds: float
    ) -> list[RetrievalHit]: ...


REFUSAL = "I cannot answer that from the material available at your current lesson position."


@dataclass
class GroundedChatService:
    authorizer: SourceAuthorizer
    retriever: EvidenceRetriever
    provider: ModelProvider

    async def answer(self, request: ChatRequest) -> ChatResponse:
        # Session IDs/allowed-source IDs supplied by a client are not proof of ownership.
        try:
            await self.authorizer.ensure_allowed(
                request.session_id, request.allowed_source_ids
            )
        except AuthorizationError:
            raise
        except Exception as exc:
            raise ChatUnavailableError("Source authorization unavailable.") from exc

        try:
            documents = await self.retriever.retrieve_documents(
                request.question, request.allowed_source_ids
            )
            video = []
            if request.video_source_id is not None:
                # ChatRequest guarantees playback_seconds exists with video_source_id.
                video = await self.retriever.retrieve_video(
                    request.question,
                    request.video_source_id,
                    request.playback_seconds,
                )
        except Exception as exc:
            raise ChatUnavailableError("Evidence retrieval unavailable.") from exc

        allowed = set(request.allowed_source_ids)
        evidence = []
        seen = set()
        # Never accept a video hit returned through the document interface, or vice versa.
        for hit, channel in (
            [(hit, "document") for hit in documents] +
            [(hit, "video") for hit in video]
        ):
            citation = hit.citation
            if citation.source_id not in allowed:
                continue
            if channel == "document":
                if citation.source_type not in (SourceType.PDF, SourceType.PPTX):
                    continue
            elif (
                citation.source_type != SourceType.VIDEO
                or citation.source_id != request.video_source_id
                or citation.end_seconds is None
                or request.playback_seconds is None
                or citation.end_seconds > request.playback_seconds
            ):
                # Exclude full chunks overlapping the current playback cutoff.
                continue

            key = (citation.source_id, citation.chunk_id)
            if key not in seen:
                seen.add(key)
                evidence.append(hit)
            if len(evidence) >= 8:
                break

        if not evidence:
            return ChatResponse(
                session_id=request.session_id,
                answer=REFUSAL,
                grounded=False,
                citations=[],
            )

        try:
            answer = await self.provider.answer(
                request.question, [hit.text for hit in evidence]
            )
        except ProviderError as exc:
            raise ChatUnavailableError("Answer generation unavailable.") from exc
        except Exception as exc:
            raise ChatUnavailableError("Answer generation unavailable.") from exc

        if answer is None:
            return ChatResponse(
                session_id=request.session_id,
                answer=REFUSAL,
                grounded=False,
                citations=[],
            )
        if not isinstance(answer, str) or not answer.strip():
            raise ChatUnavailableError("Answer generation returned no result.")

        # These are candidate evidence citations, NOT LLM-generated citations
        # and NOT an automated entailment verification of answer accuracy.
        return ChatResponse(
            session_id=request.session_id,
            answer=answer.strip(),
            grounded=True,
            citations=[hit.citation for hit in evidence],
        )
