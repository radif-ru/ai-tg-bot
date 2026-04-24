"""Утилиты работы с текстом (длинные сообщения, разбивка и т. п.)."""

from __future__ import annotations

TELEGRAM_MESSAGE_LIMIT = 4096


def split_long_message(text: str, limit: int = TELEGRAM_MESSAGE_LIMIT) -> list[str]:
    """Разбить `text` на чанки длиной не больше `limit`.

    Старается резать по переводам строк, затем по пробелам; при их отсутствии
    делает hard-split по границе `limit`. Сохраняет исходный порядок и
    содержимое (за исключением ведущих пробелов у чанков после newline/space).
    """
    if text == "":
        return [""]
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    remaining = text
    while len(remaining) > limit:
        break_at = remaining.rfind("\n", 0, limit)
        if break_at <= 0:
            break_at = remaining.rfind(" ", 0, limit)
        if break_at <= 0:
            break_at = limit
        chunks.append(remaining[:break_at])
        remaining = remaining[break_at:].lstrip("\n ")
    if remaining:
        chunks.append(remaining)
    return chunks
