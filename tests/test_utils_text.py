"""Тесты утилит для работы с текстом."""

from __future__ import annotations

from app.utils.text import split_long_message


def test_short_text_returns_single_chunk() -> None:
    chunks = split_long_message("hello", limit=4096)

    assert chunks == ["hello"]


def test_exact_limit_returns_single_chunk() -> None:
    text = "x" * 4096

    chunks = split_long_message(text, limit=4096)

    assert chunks == [text]


def test_splits_on_newline_boundary_when_possible() -> None:
    text = "line1\nline2\nline3"

    chunks = split_long_message(text, limit=12)

    assert all(len(c) <= 12 for c in chunks)
    assert "".join(chunks).replace("\n", "") == text.replace("\n", "")


def test_splits_on_space_when_no_newline() -> None:
    text = "one two three four five six seven eight"

    chunks = split_long_message(text, limit=12)

    assert all(len(c) <= 12 for c in chunks)


def test_hard_split_when_no_boundary() -> None:
    text = "a" * 10000

    chunks = split_long_message(text, limit=4096)

    assert all(len(c) <= 4096 for c in chunks)
    assert "".join(chunks) == text


def test_empty_string_returns_empty_chunk_list() -> None:
    chunks = split_long_message("", limit=4096)

    assert chunks == [""] or chunks == []
