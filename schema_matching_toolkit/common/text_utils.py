import re
from difflib import SequenceMatcher


def normalize_text(s: str) -> str:
    return re.sub(r"[_\W]+", "", str(s).lower())


def fuzzy_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()
