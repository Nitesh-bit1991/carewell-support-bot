"""Detect questions that ask for medical advice. The bot must not answer those.

This is a simple rule-based check. A production system would add a trained
classifier and have the pharmacist review the rules.
"""
import re

_PATTERNS = [
    r"\bdos(e|es|age|ing)\b",
    r"\bside[- ]effects?\b",
    r"\binteract\w*",
    r"\boverdos\w*",
    r"\bhow (many|much) (tablets?|pills?|capsules?|mg|ml|drops?)\b",
    r"\b(can|should|could|may) i (take|give|mix)\b",
    r"\bsafe (to|for)\b",
    r"\bpregnan\w*|\bbreast ?feed\w*",
    r"\b(which|what|best) (medicine|medication|drug|tablet|pill|antibiotic)s?\b",
    r"\bwhat (should|can) i take\b",
    r"\bsymptoms?\b|\bdiagnos\w*",
    r"\b(cure|treat|treatment)\b",
    r"\bmg\b",
]
_MEDICAL = re.compile("|".join(_PATTERNS), re.IGNORECASE)


def asks_for_medical_advice(question: str) -> bool:
    return bool(_MEDICAL.search(question))
