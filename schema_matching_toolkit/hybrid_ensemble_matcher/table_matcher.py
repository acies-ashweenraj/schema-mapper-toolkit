from typing import Dict, Any, List


def build_table_matches_from_column_matches(
    hybrid_result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Converts hybrid column matching output into table-level matches.

    Input:
      hybrid_result = output of hybrid_ensemble_match()

    Output:
      {
        "table_match_count": N,
        "table_matches": [
          {
            "source_table": "...",
            "best_match_table": "...",
            "confidence": 0.92,
            "column_match_count": 6
          }
        ],
        "column_matches": [...original hybrid matches...]
      }
    """
    matches = hybrid_result.get("matches", [])
    grouped: Dict[str, List[Dict[str, Any]]] = {}

    # group column matches by source table
    for m in matches:
        src = m.get("source", "")
        if "." not in src:
            continue
        src_table = src.split(".", 1)[0]
        grouped.setdefault(src_table, []).append(m)

    table_matches = []

    for src_table, col_matches in grouped.items():
        if not col_matches:
            continue

        # vote target table by best_match
        votes = {}
        for m in col_matches:
            best = m.get("best_match")
            if best and "." in best:
                tgt_table = best.split(".", 1)[0]
                votes[tgt_table] = votes.get(tgt_table, 0) + 1

        best_table = None
        if votes:
            best_table = sorted(votes.items(), key=lambda x: x[1], reverse=True)[0][0]

        # confidence = avg confidence of columns
        avg_conf = sum(m.get("confidence", 0.0) for m in col_matches) / len(col_matches)

        table_matches.append(
            {
                "source_table": src_table,
                "best_match_table": best_table,
                "confidence": round(avg_conf, 4),
                "column_match_count": len(col_matches),
            }
        )

    table_matches.sort(key=lambda x: x["confidence"], reverse=True)

    return {
        "table_match_count": len(table_matches),
        "table_matches": table_matches,
        "column_matches": matches,
    }
