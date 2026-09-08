"""Detect and redact common personal identifiers in processed text only."""

import re
from dataclasses import dataclass

EMAIL_PATTERN = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_PATTERN = re.compile(r"\b(?:\+1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b")
SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")


@dataclass(frozen=True, slots=True)
class RedactionResult:
    """Redacted text and the identifier classes that were removed."""

    text: str
    codes: tuple[str, ...]


def redact_personal_identifiers(text: str) -> RedactionResult:
    """Replace emails, US phone numbers, and SSN-shaped values with markers."""
    codes: list[str] = []
    redacted = text
    redacted, email_count = EMAIL_PATTERN.subn("[REDACTED_EMAIL]", redacted)
    if email_count:
        codes.append("email")
    redacted, phone_count = PHONE_PATTERN.subn("[REDACTED_PHONE]", redacted)
    if phone_count:
        codes.append("phone")
    redacted, ssn_count = SSN_PATTERN.subn("[REDACTED_SSN]", redacted)
    if ssn_count:
        codes.append("ssn")
    return RedactionResult(text=redacted, codes=tuple(codes))
