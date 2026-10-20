import asyncio
import json

import httpx
import pytest

from backend.llm_provider import (
    DEFAULT_GROQ_MODEL,
    GROQ_CHAT_COMPLETIONS_ENDPOINT,
    OpenAICompatibleProvider,
    ProviderConfig,
    ProviderError,
)

CONFIG = ProviderConfig(
    endpoint=GROQ_CHAT_COMPLETIONS_ENDPOINT,
    model="test-groq-model",
    api_key="test-key",
)


def test_provider_config_uses_groq_environment(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_MODEL_NAME", "custom-test-model")

    config = ProviderConfig.from_env()

    assert config.endpoint == GROQ_CHAT_COMPLETIONS_ENDPOINT
    assert config.model == "custom-test-model"
    assert config.api_key == "test-key"


def test_provider_config_defaults_to_groq_quickstart_model(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.delenv("GROQ_MODEL_NAME", raising=False)

    config = ProviderConfig.from_env()

    assert config.endpoint == GROQ_CHAT_COMPLETIONS_ENDPOINT
    assert config.model == DEFAULT_GROQ_MODEL


def test_provider_config_requires_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_MODEL_NAME", "test-model")

    with pytest.raises(ProviderError, match="Groq model configuration"):
        ProviderConfig.from_env()


def test_provider_config_rejects_empty_model_name(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_MODEL_NAME", " ")

    with pytest.raises(ProviderError, match="Groq model configuration"):
        ProviderConfig.from_env()


def test_provider_sends_groq_request_and_parses_answer():
    async def mock(request):
        assert str(request.url) == GROQ_CHAT_COMPLETIONS_ENDPOINT
        assert request.headers["authorization"] == "Bearer test-key"
        payload = json.loads(request.content)
        assert payload["model"] == CONFIG.model
        assert "[E1] Lecture about caching" in payload["messages"][1]["content"]
        assert payload["temperature"] == 0
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": "SUPPORTED\nCaching stores data."}}
                ]
            },
        )

    provider = OpenAICompatibleProvider(
        CONFIG,
        transport=httpx.MockTransport(mock),
    )
    answer = asyncio.run(
        provider.answer("What is caching?", ["Lecture about caching"])
    )

    assert answer == "Caching stores data."


def test_provider_returns_abstention_for_unsupported_evidence():
    async def mock(request):
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "UNSUPPORTED"}}]},
        )

    provider = OpenAICompatibleProvider(
        CONFIG,
        transport=httpx.MockTransport(mock),
    )

    assert asyncio.run(provider.answer("Question?", ["Unrelated evidence"])) is None


def test_provider_rejects_unstructured_support_decision():
    async def mock(request):
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "Maybe the answer is 42."}}]},
        )

    provider = OpenAICompatibleProvider(
        CONFIG,
        transport=httpx.MockTransport(mock),
    )

    with pytest.raises(ProviderError, match="support decision"):
        asyncio.run(provider.answer("Question?", ["Evidence"]))


def test_no_evidence_never_calls_provider():
    provider = OpenAICompatibleProvider(CONFIG)

    with pytest.raises(ProviderError, match="approved evidence"):
        asyncio.run(provider.answer("Anything?", []))


def test_api_errors_do_not_expose_response_or_key():
    async def mock(request):
        assert request.headers["authorization"] == "Bearer test-key"
        return httpx.Response(
            401,
            json={"detail": "private provider error"},
        )

    provider = OpenAICompatibleProvider(
        CONFIG,
        transport=httpx.MockTransport(mock),
    )

    with pytest.raises(ProviderError, match="request failed") as error:
        asyncio.run(provider.answer("Question?", ["Evidence"]))

    assert "private provider error" not in str(error.value)
    assert CONFIG.api_key not in str(error.value)


def test_provider_retries_transient_error():
    calls = 0

    async def mock(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "SUPPORTED\nRetry worked"}}]},
        )

    provider = OpenAICompatibleProvider(
        CONFIG,
        transport=httpx.MockTransport(mock),
    )

    assert asyncio.run(provider.answer("Question?", ["Source"])) == "Retry worked"
    assert calls == 2


def test_provider_rejects_malformed_success_payload():
    async def mock(request):
        return httpx.Response(200, json={"choices": []})

    provider = OpenAICompatibleProvider(
        CONFIG,
        transport=httpx.MockTransport(mock),
    )

    with pytest.raises(ProviderError, match="invalid response"):
        asyncio.run(provider.answer("Explain this", ["Grounded source text"]))
