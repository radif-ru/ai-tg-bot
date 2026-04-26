"""Тесты `Summarizer` — сжатие истории через LLM."""

from __future__ import annotations

import pytest

from app.services.llm import LLMTimeout
from app.services.summarizer import Summarizer


def _make_summarizer(mocker, *, chat_return=None, chat_side_effect=None) -> tuple[
    Summarizer, "mocker.AsyncMock"
]:
    client = mocker.MagicMock()
    client.chat = mocker.AsyncMock(
        return_value=chat_return,
        side_effect=chat_side_effect,
    )
    summarizer = Summarizer(client, prompt="SUMMARY_PROMPT")
    return summarizer, client.chat


async def test_summarize_calls_chat_with_system_prompt_then_history(mocker) -> None:
    summarizer, chat_mock = _make_summarizer(mocker, chat_return="итог")

    history = [
        {"role": "user", "content": "привет"},
        {"role": "assistant", "content": "здравствуй"},
        {"role": "user", "content": "запомни моё имя"},
    ]

    await summarizer.summarize(history, model="m1")

    chat_mock.assert_awaited_once()
    call_kwargs = chat_mock.await_args.kwargs
    call_args = chat_mock.await_args.args
    payload = call_args[0]
    assert call_kwargs == {"model": "m1"}
    assert payload[0] == {"role": "system", "content": "SUMMARY_PROMPT"}
    assert payload[1:] == history


async def test_summarize_returns_chat_result_unmodified(mocker) -> None:
    summarizer, _ = _make_summarizer(mocker, chat_return="резюме диалога")

    result = await summarizer.summarize(
        [{"role": "user", "content": "x"}], model="m"
    )

    assert result == "резюме диалога"


async def test_summarize_propagates_llm_errors(mocker) -> None:
    summarizer, _ = _make_summarizer(mocker, chat_side_effect=LLMTimeout("boom"))

    with pytest.raises(LLMTimeout):
        await summarizer.summarize(
            [{"role": "user", "content": "x"}], model="m"
        )


async def test_summarize_with_empty_history_still_sends_system_prompt(mocker) -> None:
    """Граничный случай: пустая история — отправляется только system-prompt."""
    summarizer, chat_mock = _make_summarizer(mocker, chat_return="ничего")

    await summarizer.summarize([], model="m")

    payload = chat_mock.await_args.args[0]
    assert payload == [{"role": "system", "content": "SUMMARY_PROMPT"}]


def test_summarizer_rejects_empty_prompt(mocker) -> None:
    client = mocker.MagicMock()

    with pytest.raises(ValueError):
        Summarizer(client, prompt="")
    with pytest.raises(ValueError):
        Summarizer(client, prompt="   ")
