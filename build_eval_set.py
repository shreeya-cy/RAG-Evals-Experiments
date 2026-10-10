"""Build the retrieval evaluation set from QASPER train questions.

An annotation (one annotator's answer to a question) is kept only if it is answerable,
has non-empty evidence, no evidence mentions "FLOAT SELECTED" (tables/figures are not
embedded), and every evidence string can be located in paper_text. A question is kept if
at least one annotation survives. Gold spans are character offsets into paper_text, the
same text the hierarchical and agentic collections use (see build_paper_text).
"""
import json
from pathlib import Path

from datasets import load_dataset

import config
from ingest_hierarchical import build_paper_text

AGENTIC_SUBSET_PAPERS = config.AGENTIC_SUBSET_SIZE


def merge_spans(spans: list[list[int]]) -> list[list[int]]:
    merged: list[list[int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def locate(evidence: str, paper_text: str, paragraphs: dict[str, list[int]]) -> list[int] | None:
    """Span of an evidence string in paper_text: an exact paragraph, else a substring."""
    text = evidence.strip()
    if text in paragraphs:
        return paragraphs[text]
    pos = paper_text.find(text)
    return [pos, pos + len(text)] if pos >= 0 and text else None


def main() -> None:
    dataset = load_dataset(config.DATASET_NAME, split=config.DATASET_SPLIT, trust_remote_code=True)
    stats = {k: 0 for k in (
        "questions_total", "annotations_total", "drop_unanswerable", "drop_empty_evidence",
        "drop_float_selected", "drop_unlocatable_evidence", "annotations_kept",
        "questions_kept", "questions_multi_annotator", "duplicate_paragraphs_in_papers",
    )}
    items = []
    for paper_idx, paper in enumerate(dataset):
        paper_text, parents = build_paper_text(paper)
        paragraphs: dict[str, list[int]] = {}
        for parent in parents:
            if parent["text"] in paragraphs:
                stats["duplicate_paragraphs_in_papers"] += 1  # same text twice: first occurrence is used
                continue
            paragraphs[parent["text"]] = [parent["start"], parent["end"]]

        qas = paper["qas"]
        for question, question_id, answers in zip(qas["question"], qas["question_id"], qas["answers"]):
            stats["questions_total"] += 1
            annotators = []
            for annotation_id, answer in zip(answers["annotation_id"], answers["answer"]):
                stats["annotations_total"] += 1
                evidence = answer["evidence"]
                if answer["unanswerable"]:
                    stats["drop_unanswerable"] += 1
                elif not evidence:
                    stats["drop_empty_evidence"] += 1
                elif any("FLOAT SELECTED" in e for e in evidence):
                    stats["drop_float_selected"] += 1
                else:
                    spans = [locate(e, paper_text, paragraphs) for e in evidence]
                    if any(s is None for s in spans):
                        stats["drop_unlocatable_evidence"] += 1
                        continue
                    stats["annotations_kept"] += 1
                    annotators.append(
                        {
                            "annotation_id": annotation_id,
                            "evidence": evidence,
                            "gold_spans": merge_spans(spans),
                        }
                    )
            if not annotators:
                continue
            stats["questions_kept"] += 1
            stats["questions_multi_annotator"] += len(annotators) > 1
            items.append(
                {
                    "question_id": question_id,
                    "question": question,
                    "paper_id": paper["id"],
                    "title": paper["title"],
                    "in_agentic_subset": paper_idx < AGENTIC_SUBSET_PAPERS,
                    "annotators": annotators,
                    "gold_spans_per_annotator": [a["gold_spans"] for a in annotators],
                    "gold_spans_union": merge_spans([s for a in annotators for s in a["gold_spans"]]),
                }
            )

    stats["questions_in_agentic_subset"] = sum(i["in_agentic_subset"] for i in items)
    out = Path(config.EVAL_DIR)
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval_set.json").write_text(json.dumps(items, indent=1))
    (out / "eval_set_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
