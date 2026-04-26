"""Тесты LLM-клиента (`OllamaClient`) и маппинга ошибок."""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

import httpx
import ollama
import pytest

from app.services.llm import (
    LLMBadResponse,
    LLMError,
    LLMTimeout,
    LLMUnavailable,
    OllamaClient,
)


def _make_client(mocker, *, generate_return=None, generate_side_effect=None) -> OllamaClient:
    inner = mocker.MagicMock()
    inner.generate = mocker.AsyncMock(
        return_value=generate_return,
        side_effect=generate_side_effect,
    )
    inner._client = mocker.MagicMock()
    inner._client.aclose = mocker.AsyncMock()
    return OllamaClient(base_url="http://x", timeout=1.0, client=inner)


def _make_chat_client(mocker, *, chat_return=None, chat_side_effect=None) -> OllamaClient:
    inner = mocker.MagicMock()
    inner.chat = mocker.AsyncMock(
        return_value=chat_return,
        side_effect=chat_side_effect,
    )
    inner._client = mocker.MagicMock()
    inner._client.aclose = mocker.AsyncMock()
    return OllamaClient(base_url="http://x", timeout=1.0, client=inner)


async def test_generate_success_returns_text(mocker) -> None:
    client = _make_client(mocker, generate_return=SimpleNamespace(response="hello"))

    result = await client.generate("ping", model="m", system_prompt=None)

    assert result == "hello"


async def test_generate_timeout_httpx(mocker) -> None:
    client = _make_client(
        mocker, generate_side_effect=httpx.TimeoutException("timeout")
    )

    with pytest.raises(LLMTimeout):
        await client.generate("ping", model="m", system_prompt=None)


async def test_generate_timeout_asyncio(mocker) -> None:
    client = _make_client(mocker, generate_side_effect=asyncio.TimeoutError())

    with pytest.raises(LLMTimeout):
        await client.generate("ping", model="m", system_prompt=None)


async def test_generate_connect_error_maps_to_unavailable(mocker) -> None:
    client = _make_client(mocker, generate_side_effect=httpx.ConnectError("no route"))

    with pytest.raises(LLMUnavailable):
        await client.generate("ping", model="m", system_prompt=None)


async def test_generate_404_maps_to_bad_response(mocker) -> None:
    err = ollama.ResponseError("model not found", 404)
    client = _make_client(mocker, generate_side_effect=err)

    with pytest.raises(LLMBadResponse, match="не найдена"):
        await client.generate("ping", model="m", system_prompt=None)


async def test_generate_5xx_maps_to_bad_response(mocker) -> None:
    err = ollama.ResponseError("internal", 503)
    client = _make_client(mocker, generate_side_effect=err)

    with pytest.raises(LLMBadResponse):
        await client.generate("ping", model="m", system_prompt=None)


async def test_generate_empty_response_raises_bad_response(mocker) -> None:
    client = _make_client(mocker, generate_return=SimpleNamespace(response="   "))

    with pytest.raises(LLMBadResponse):
        await client.generate("ping", model="m", system_prompt=None)


async def test_generate_success_logs_metrics(mocker, caplog) -> None:
    client = _make_client(mocker, generate_return=SimpleNamespace(response="hi there"))
    caplog.set_level(logging.INFO, logger="app.services.llm")

    await client.generate("prompt-text", model="qwen3.5:0.8b", system_prompt="sys")

    assert any(
        "model=qwen3.5:0.8b" in rec.message
        and "len_in=11" in rec.message
        and "len_out=8" in rec.message
        and "dur_ms=" in rec.message
        and "status=ok" in rec.message
        for rec in caplog.records
    )


async def test_generate_failure_logs_error_status(mocker, caplog) -> None:
    err = ollama.ResponseError("internal", 500)
    client = _make_client(mocker, generate_side_effect=err)
    caplog.set_level(logging.INFO, logger="app.services.llm")

    with pytest.raises(LLMBadResponse):
        await client.generate("ping", model="m", system_prompt=None)

    assert any("status=http_500" in rec.message for rec in caplog.records)


async def test_close_is_safe(mocker) -> None:
    client = _make_client(mocker, generate_return=SimpleNamespace(response="ok"))

    await client.close()

    client._client._client.aclose.assert_awaited()


async def test_exception_hierarchy() -> None:
    for cls in (LLMTimeout, LLMUnavailable, LLMBadResponse):
        assert issubclass(cls, LLMError)


def test_module_does_not_import_aiogram() -> None:
    import re

    import app.services.llm as mod

    src = open(mod.__file__, encoding="utf-8").read()
    # Модуль сам не должен импортировать aiogram.
    assert re.search(r"^\s*(from aiogram|import aiogram)", src, re.MULTILINE) is None


# --- chat(messages, model) ---


def _chat_response(content: str) -> SimpleNamespace:
    return SimpleNamespace(message=SimpleNamespace(content=content))


async def test_chat_success_returns_message_content(mocker) -> None:
    client = _make_chat_client(mocker, chat_return=_chat_response("здравствуй"))

    result = await client.chat(
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "привет"},
        ],
        model="qwen3.5:0.8b",
    )

    assert result == "здравствуй"


async def test_chat_success_logs_metrics(mocker, caplog) -> None:
    client = _make_chat_client(mocker, chat_return=_chat_response("ok-text"))
    caplog.set_level(logging.INFO, logger="app.services.llm")

    await client.chat(
        [
            {"role": "system", "content": "abc"},
            {"role": "user", "content": "defg"},
        ],
        model="m",
    )

    assert any(
        "llm.chat" in rec.message
        and "model=m" in rec.message
        and "messages=2" in rec.message
        and "len_in=7" in rec.message  # 3 + 4
        and "len_out=7" in rec.message
        and "status=ok" in rec.message
        for rec in caplog.records
    )


async def test_chat_timeout_httpx(mocker) -> None:
    client = _make_chat_client(
        mocker, chat_side_effect=httpx.TimeoutException("timeout")
    )

    with pytest.raises(LLMTimeout):
        await client.chat([{"role": "user", "content": "p"}], model="m")


async def test_chat_connect_error_maps_to_unavailable(mocker) -> None:
    client = _make_chat_client(
        mocker, chat_side_effect=httpx.ConnectError("no route")
    )

    with pytest.raises(LLMUnavailable):
        await client.chat([{"role": "user", "content": "p"}], model="m")


async def test_chat_404_maps_to_bad_response(mocker) -> None:
    err = ollama.ResponseError("model not found", 404)
    client = _make_chat_client(mocker, chat_side_effect=err)

    with pytest.raises(LLMBadResponse, match="не найдена"):
        await client.chat([{"role": "user", "content": "p"}], model="m")


async def test_chat_5xx_maps_to_bad_response(mocker) -> None:
    err = ollama.ResponseError("internal", 503)
    client = _make_chat_client(mocker, chat_side_effect=err)

    with pytest.raises(LLMBadResponse):
        await client.chat([{"role": "user", "content": "p"}], model="m")


async def test_chat_empty_response_raises_bad_response(mocker) -> None:
    client = _make_chat_client(mocker, chat_return=_chat_response("   "))

    with pytest.raises(LLMBadResponse):
        await client.chat([{"role": "user", "content": "p"}], model="m")


async def test_chat_supports_dict_response_shape(mocker) -> None:
    """Если ollama-клиент вернул dict (не объект) — content всё равно достаётся."""
    client = _make_chat_client(
        mocker, chat_return={"message": {"content": "ok"}}
    )

    result = await client.chat([{"role": "user", "content": "p"}], model="m")

    assert result == "ok"


# --- estimate_tokens ---


def test_estimate_tokens_for_string() -> None:
    from app.services.llm import estimate_tokens

    # 11 символов ('hello world') → ceil(11/4) == 3.
    assert estimate_tokens("hello world") == 3


def test_estimate_tokens_for_messages_sums_content() -> None:
    from app.services.llm import estimate_tokens

    messages = [
        {"role": "user", "content": "abc"},        # 3
        {"role": "assistant", "content": "defg"},  # 4
    ]
    # 7 символов суммарно → ceil(7/4) == 2.
    assert estimate_tokens(messages) == 2


def test_estimate_tokens_min_one() -> None:
    from app.services.llm import estimate_tokens

    assert estimate_tokens("") == 1
    assert estimate_tokens([]) == 1
