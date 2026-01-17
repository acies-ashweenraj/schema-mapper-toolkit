from schema_mapper_toolkit.common.text_utils import fuzzy_ratio


def manual_score(src: str, tgt: str) -> float:
    if src == tgt:
        return 1.0
    return fuzzy_ratio(src, tgt)


def type_score(src_type: str, tgt_type: str) -> float:
    s = (src_type or "").lower()
    t = (tgt_type or "").lower()
    if s == t and s:
        return 1.0
    if ("int" in s and "int" in t) or ("char" in s and "char" in t) or ("date" in s and "date" in t):
        return 0.8
    return 0.0


def ensemble_score(manual_s: float, bm25_s: float, dense_s: float, type_s: float) -> float:
    return (
        0.15 * manual_s +
        0.30 * bm25_s +
        0.45 * dense_s +
        0.10 * type_s
    )
