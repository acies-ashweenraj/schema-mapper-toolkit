from typing import Dict, Any, List
from rank_bm25 import BM25Okapi


def _flatten_columns(schema: Dict[str, Any]) -> List[Dict[str, str]]:
    cols = []

    for t in schema.get("tables", []):
        table_name = t.get("table_name") or t.get("table") or t.get("name")
        if not table_name:
            continue

        for c in t.get("columns", []):
            col_name = c.get("column_name") or c.get("column") or c.get("name")
            dtype = c.get("data_type") or c.get("type") or ""

            if not col_name:
                continue

            col_id = f"{table_name}.{col_name}"
            text = f"{table_name} {col_name} {dtype}".lower()
            cols.append({"id": col_id, "text": text})

    return cols


def bm25_match(
    source_schema: Dict[str, Any],
    target_schema: Dict[str, Any],
    top_k: int = 5,   # ✅ keep it so old test will work
) -> Dict[str, Any]:
    """
    Returns only Top-1 match per source column (even if top_k is given)
    """
    source_cols = _flatten_columns(source_schema)
    target_cols = _flatten_columns(target_schema)

    if not source_cols or not target_cols:
        return {"method": "bm25", "top_k": 1, "matches": []}

    corpus = [t["text"].split() for t in target_cols]
    bm25 = BM25Okapi(corpus)

    results = []

    for src in source_cols:
        query = src["text"].split()
        scores = bm25.get_scores(query)

        # normalize 0..1
        max_score = max(scores) if len(scores) > 0 else 0
        if max_score > 0:
            scores = [s / max_score for s in scores]
        else:
            scores = [0.0 for _ in scores]

        ranked = sorted(
            zip(target_cols, scores),
            key=lambda x: x[1],
            reverse=True
        )

        if not ranked:
            results.append(
                {"source": src["id"], "best_match": None, "score": 0.0}
            )
            continue

        best_target, best_score = ranked[0]

        results.append(
            {
                "source": src["id"],
                "best_match": best_target["id"],
                "score": round(float(best_score), 4),
            }
        )

    return {
        "method": "bm25",
        "top_k": 1,  
        "matches": results
    }
