"""Score the saved retrieval results against the gold evidence spans.

For each question only the in-budget retrieved units count (status != "out"; a truncated
unit counts with its truncated span). Gold spans are character offsets in paper_text.

- evidence_recall: gold characters covered by the retrieved text / total gold characters,
  per annotator, then the MAX over annotators (the "union" variant pools all annotators).
- full_coverage: 1 if every gold paragraph of at least one annotator is >= 80% covered.
- mrr: 1 / rank of the first unit overlapping any gold paragraph by >= 1 character, else 0.
- tokens_to_first_hit: tokens in the retrieved list before the first overlapping unit.
"""
import json
import statistics as st
from pathlib import Path

import config
from retrieve import STRATEGIES


def merge(spans: list[list[int]]) -> list[list[int]]:
    merged: list[list[int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def covered(retrieved: list[list[int]], start: int, end: int) -> int:
    """Characters of [start, end) covered by the merged retrieved spans."""
    return sum(max(0, min(end, e) - max(start, s)) for s, e in retrieved)


def recall_and_coverage(retrieved: list[list[int]], gold: list[list[int]]) -> tuple[float, bool]:
    total = sum(e - s for s, e in gold)
    hit = sum(covered(retrieved, s, e) for s, e in gold)
    full = all(covered(retrieved, s, e) / (e - s) >= config.PARAGRAPH_COVERAGE_THRESHOLD for s, e in gold)
    return hit / total, full


def score_question(entries: list[dict], item: dict) -> dict:
    used = [e for e in entries if e["status"] != "out"]
    retrieved = merge([[e["chunk_start"], e["chunk_end"]] for e in used])

    per_annotator = [recall_and_coverage(retrieved, g) for g in item["gold_spans_per_annotator"]]
    union_recall, union_full = recall_and_coverage(retrieved, item["gold_spans_union"])

    first_rank = tokens_before = None
    for e in used:
        if any(min(e["chunk_end"], ge) - max(e["chunk_start"], gs) > 0 for gs, ge in item["gold_spans_union"]):
            first_rank, tokens_before = e["rank"], e["cum_tokens"] - e["tokens"]
            break
    return {
        "question_id": item["question_id"],
        "evidence_recall": max(r for r, _ in per_annotator),
        "full_coverage": float(any(f for _, f in per_annotator)),
        "evidence_recall_union": union_recall,
        "full_coverage_union": float(union_full),
        "mrr": 1 / first_rank if first_rank else 0.0,
        "hit": first_rank is not None,
        "tokens_to_first_hit": tokens_before,
        "units_in_budget": len(used),
        "tokens_used": used[-1]["cum_tokens"] if used else 0,
    }


def aggregate(rows: list[dict]) -> dict:
    n = len(rows)
    mean = lambda key: round(sum(r[key] for r in rows) / n, 4)
    hits = [r["tokens_to_first_hit"] for r in rows if r["hit"]]
    return {
        "num_questions": n,
        "evidence_recall@B": mean("evidence_recall"),
        "full_coverage_rate@B": mean("full_coverage"),
        "mrr": mean("mrr"),
        "hit_rate": round(len(hits) / n, 4),
        "tokens_to_first_hit_mean": round(st.mean(hits), 1) if hits else None,
        "tokens_to_first_hit_median": st.median(hits) if hits else None,
        "tokens_to_first_hit_misses_as_budget_mean": round(
            sum(r["tokens_to_first_hit"] if r["hit"] else config.TOKEN_BUDGET for r in rows) / n, 1),
        "secondary_union": {
            "evidence_recall_union@B": mean("evidence_recall_union"),
            "full_coverage_rate_union@B": mean("full_coverage_union"),
        },
        "mean_units_in_budget": round(st.mean(r["units_in_budget"] for r in rows), 2),
    }


def main() -> None:
    items = {i["question_id"]: i for i in json.loads((Path(config.EVAL_DIR) / "eval_set.json").read_text())}
    results, per_question = {}, {}
    for strategy, collection in STRATEGIES.items():
        rows = []
        for line in open(Path(config.RETRIEVAL_DIR) / f"{collection}.jsonl"):
            r = json.loads(line)
            row = score_question(r["retrieved"], items[r["question_id"]])
            row["in_agentic_subset"] = items[r["question_id"]]["in_agentic_subset"]
            rows.append(row)
        per_question[strategy] = rows
        results[strategy] = {"collection": collection, "all_questions": aggregate(rows)}
        subset = [r for r in rows if r["in_agentic_subset"]]
        if strategy != "agentic_subset":  # same 155 questions for a like-for-like comparison with agentic
            results[strategy]["agentic_subset_questions"] = aggregate(subset)

    out = Path(config.METRICS_DIR)
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "token_budget": config.TOKEN_BUDGET,
        "paragraph_coverage_threshold": config.PARAGRAPH_COVERAGE_THRESHOLD,
        "definitions": {
            "evidence_recall@B": "gold-evidence characters covered by in-budget retrieved text / total gold characters, per annotator, max over annotators, mean over questions",
            "full_coverage_rate@B": "share of questions where every gold paragraph of at least one annotator is >= 80% covered",
            "mrr": "mean of 1/rank of the first in-budget unit overlapping any gold paragraph by >= 1 character (0 if none)",
            "tokens_to_first_hit": "tokens of the in-budget units ranked before the first overlapping unit (excludes that unit); mean/median over questions with a hit; the misses_as_budget variant counts a miss as the full budget",
        },
        "strategies": results,
    }
    (out / "retrieval_metrics.json").write_text(json.dumps(report, indent=1))
    (out / "per_question_metrics.json").write_text(json.dumps(per_question))
    print(json.dumps({s: r["all_questions"] for s, r in results.items()}, indent=1))


if __name__ == "__main__":
    main()
