"""Тесты in-memory реестра пользовательских настроек (модель + промпт)."""

from __future__ import annotations

from app.services.model_registry import UserSettingsRegistry

DEFAULT_MODEL = "qwen3.5:0.8b"
DEFAULT_PROMPT = "Ты — полезный ассистент."


def _make_registry() -> UserSettingsRegistry:
    return UserSettingsRegistry(
        default_model=DEFAULT_MODEL,
        default_prompt=DEFAULT_PROMPT,
    )


def test_get_model_returns_default_when_not_set() -> None:
    registry = _make_registry()

    assert registry.get_model(123) == DEFAULT_MODEL


def test_set_model_then_get_returns_value() -> None:
    registry = _make_registry()

    registry.set_model(123, "deepseek-r1:1.5b")

    assert registry.get_model(123) == "deepseek-r1:1.5b"


def test_get_prompt_returns_default_when_not_set() -> None:
    registry = _make_registry()

    assert registry.get_prompt(123) == DEFAULT_PROMPT


def test_set_prompt_then_get_returns_value() -> None:
    registry = _make_registry()

    registry.set_prompt(123, "custom")

    assert registry.get_prompt(123) == "custom"


def test_reset_restores_defaults_for_user() -> None:
    registry = _make_registry()
    registry.set_model(123, "deepseek-r1:1.5b")
    registry.set_prompt(123, "custom")

    registry.reset(123)

    assert registry.get_model(123) == DEFAULT_MODEL
    assert registry.get_prompt(123) == DEFAULT_PROMPT


def test_reset_prompt_only_restores_prompt() -> None:
    registry = _make_registry()
    registry.set_model(123, "deepseek-r1:1.5b")
    registry.set_prompt(123, "custom")

    registry.reset_prompt(123)

    assert registry.get_model(123) == "deepseek-r1:1.5b"
    assert registry.get_prompt(123) == DEFAULT_PROMPT


def test_users_are_isolated() -> None:
    registry = _make_registry()

    registry.set_model(111, "deepseek-r1:1.5b")
    registry.set_prompt(222, "private-222")

    assert registry.get_model(111) == "deepseek-r1:1.5b"
    assert registry.get_model(222) == DEFAULT_MODEL
    assert registry.get_prompt(111) == DEFAULT_PROMPT
    assert registry.get_prompt(222) == "private-222"


def test_reset_is_noop_for_unknown_user() -> None:
    registry = _make_registry()

    registry.reset(999)

    assert registry.get_model(999) == DEFAULT_MODEL
    assert registry.get_prompt(999) == DEFAULT_PROMPT
