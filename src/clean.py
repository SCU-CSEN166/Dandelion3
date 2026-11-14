"""
Media-leftover cleanup: WhatsApp exports leave placeholder text for photos,
videos, stickers, deleted messages, and missed calls -- these aren't
content and shouldn't be classified or extracted as if they were.
"""

import re

MEDIA_PLACEHOLDER_RE = re.compile(
    r"<\s*media omitted\s*>"
    r"|\b(image|video|gif|sticker|audio|document)\s+omitted\b"
    r"|\bthis message was deleted\b"
    r"|\byou deleted this message\b"
    r"|\bmissed (voice|video) call\b",
    re.IGNORECASE,
)


def clean_message_text(text: str) -> str:
    """Strip media-placeholder phrases, leaving any real surrounding text intact."""
    cleaned = MEDIA_PLACEHOLDER_RE.sub("", text)
    return re.sub(r"\s+", " ", cleaned).strip()


def is_media_only(raw_text: str, cleaned_text: str) -> bool:
    """True if the message was nothing but a media placeholder (no real content left)."""
    return bool(raw_text.strip()) and not cleaned_text
