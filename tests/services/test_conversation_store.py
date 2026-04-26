"""Тесты `ConversationStore` — in-memory история диалога per-user."""

from __future__ import annotations

import pytest

from app.services.conversation import SUMMARY_PREFIX, ConversationStore


def test_get_history_for_unknown_user_is_empty() -> None:
    store = ConversationStore(max_messages=10)

    assert store.get_history(user_id=42) == []


def test_add_user_and_assistant_messages_preserve_order() -> None:
    store = ConversationStore(max_messages=10)

    store.add_user_message(user_id=1, content="привет")
    store.add_assistant_message(user_id=1, content="здравствуй")
    store.add_user_message(user_id=1, content="как дела")

    assert store.get_history(user_id=1) == [
        {"role": "user", "content": "привет"},
        {"role": "assistant", "content": "здравствуй"},
        {"role": "user", "content": "как дела"},
    ]


def test_truncate_drops_oldest_when_over_limit() -> None:
    store = ConversationStore(max_messages=3)

    for i in range(5):
        store.add_user_message(user_id=1, content=f"msg-{i}")

    history = store.get_history(user_id=1)
    assert len(history) == 3
    assert [m["content"] for m in history] == ["msg-2", "msg-3", "msg-4"]


def test_histories_are_isolated_per_user() -> None:
    store = ConversationStore(max_messages=10)

    store.add_user_message(user_id=1, content="a")
    store.add_user_message(user_id=2, content="b")

    assert store.get_history(user_id=1) == [{"role": "user", "content": "a"}]
    assert store.get_history(user_id=2) == [{"role": "user", "content": "b"}]


def test_clear_removes_user_history() -> None:
    store = ConversationStore(max_messages=10)
    store.add_user_message(user_id=1, content="a")

    store.clear(user_id=1)

    assert store.get_history(user_id=1) == []


def test_replace_with_summary_replaces_old_part_keeps_tail() -> None:
    store = ConversationStore(max_messages=10)
    for i in range(6):
        # 0..5 — три пары user/assistant
        role = "user" if i % 2 == 0 else "assistant"
        store._append(user_id=1, role=role, content=f"m{i}")

    store.replace_with_summary(user_id=1, summary="итог", kept_tail=2)

    history = store.get_history(user_id=1)
    assert len(history) == 1 + 2
    assert history[0]["role"] == "system"
    assert history[0]["content"] == f"{SUMMARY_PREFIX}итог"
    # Хвост сохранён без изменений.
    assert history[1] == {"role": "user", "content": "m4"}
    assert history[2] == {"role": "assistant", "content": "m5"}


def test_replace_with_summary_kept_tail_zero_drops_everything_into_summary() -> None:
    store = ConversationStore(max_messages=10)
    for i in range(4):
        store.add_user_message(user_id=1, content=f"m{i}")

    store.replace_with_summary(user_id=1, summary="итог", kept_tail=0)

    history = store.get_history(user_id=1)
    assert history == [{"role": "system", "content": f"{SUMMARY_PREFIX}итог"}]


def test_replace_with_summary_no_op_when_tail_covers_all() -> None:
    store = ConversationStore(max_messages=10)
    store.add_user_message(user_id=1, content="m0")
    store.add_assistant_message(user_id=1, content="m1")

    before = store.get_history(user_id=1)
    store.replace_with_summary(user_id=1, summary="итог", kept_tail=5)

    assert store.get_history(user_id=1) == before


def test_replace_with_summary_on_empty_history_is_noop() -> None:
    store = ConversationStore(max_messages=10)

    store.replace_with_summary(user_id=999, summary="итог", kept_tail=2)

    assert store.get_history(user_id=999) == []


def test_get_history_returns_copy_not_internal_list() -> None:
    store = ConversationStore(max_messages=10)
    store.add_user_message(user_id=1, content="hello")

    snapshot = store.get_history(user_id=1)
    snapshot.append({"role": "user", "content": "injected"})
    snapshot[0]["content"] = "MUTATED"

    fresh = store.get_history(user_id=1)
    assert fresh == [{"role": "user", "content": "hello"}]


def test_invalid_max_messages_raises() -> None:
    with pytest.raises(ValueError):
        ConversationStore(max_messages=0)
    with pytest.raises(ValueError):
        ConversationStore(max_messages=-1)


def test_invalid_kept_tail_raises() -> None:
    store = ConversationStore(max_messages=10)
    store.add_user_message(user_id=1, content="x")

    with pytest.raises(ValueError):
        store.replace_with_summary(user_id=1, summary="s", kept_tail=-1)
