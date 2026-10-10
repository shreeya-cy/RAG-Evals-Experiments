"""Retrieve chunks for every evaluation question, for every chunking strategy.

Per question: embed the question (bge query prefix), run an exact search filtered to the
question's paper, express each hit as a (start, end) span in paper_text, and fill a token
budget in rank order (the last chunk that crosses the budget is truncated to fit).

Hierarchical: search 100 children, map them to parent paragraphs, drop repeated parents
(a parent's rank is its best child's rank) and fill the budget with parent text.
"""
import argparse
import json
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from datasets import load_dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import models
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import config
from ingest_fixed_size import build_document as fixed_document
from ingest_hierarchical import build_paper_text
from ingest_recursive import build_document as recursive_document
from qdrant_setup import get_client

STRATEGIES = {
    "fixed": config.COLLECTION_NAME,
    "paragraph": config.PARAGRAPH_COLLECTION_NAME,
    "recursive": config.RECURSIVE_COLLECTION_NAME,
    "hierarchical": config.HIERARCHICAL_COLLECTION_NAME,
    "agentic_subset": config.AGENTIC_SUBSET_COLLECTION_NAME,
}


class Paper:
    """paper_text plus lazily built helpers to express any strategy's chunk as a paper_text span."""

    def __init__(self, record: dict, tokenizer):
        self.record, self.tokenizer = record, tokenizer
        self.paper_text, self.parents = build_paper_text(record)
        self._fixed = self._recursive = None

    def paragraph_span(self, payload: dict) -> list[int]:
        if payload["chunk_type"] == "abstract":
            parent = self.parents[0]
        else:
            parent = next(
                p for p in self.parents
                if p["section_index"] == payload["section_index"]
                and p["paragraph_index"] == payload["paragraph_index"]
            )
        return [parent["start"], parent["end"]]

    def fixed_span(self, payload: dict) -> list[int]:
        """Chunk i is token window i; re-derive its characters from paper_text's token offsets."""
        if self._fixed is None:
            enc = self.tokenizer(self.paper_text, add_special_tokens=False, return_offsets_mapping=True)
            ref = self.tokenizer(fixed_document(self.record), add_special_tokens=False)["input_ids"]
            self._fixed = (enc["input_ids"], enc["offset_mapping"], enc["input_ids"] == ref)
        ids, offsets, same_tokens = self._fixed
        step = config.CHUNK_SIZE - config.CHUNK_OVERLAP
        lo = payload["chunk_index"] * step
        hi = min(lo + config.CHUNK_SIZE, len(ids))
        return [offsets[lo][0], offsets[hi - 1][1]], same_tokens

    def recursive_span(self, payload: dict) -> list[int]:
        """Re-split the recursive document (deterministic) and map it onto paper_text."""
        if self._recursive is None:
            splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
                self.tokenizer, chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP,
                separators=config.RECURSIVE_SEPARATORS, keep_separator="end",
            )
            doc = recursive_document(self.record)
            title_end = len(self.record["title"].strip())
            shift = lambda pos: pos if pos <= title_end else pos - 1  # recursive doc has "\n\n" after the title, paper_text "\n"
            spans, cursor = [], 0
            for chunk in splitter.split_text(doc):
                start = doc.find(chunk, max(cursor - 2 * len(chunk), 0))
                spans.append(([shift(start), shift(start + len(chunk))], chunk))
                cursor = start + 1
            self._recursive = spans
        return self._recursive[payload["chunk_index"]]


def norm(text: str) -> str:
    """Letters and digits only: the fixed-size text was decoded from the tokenizer, which drops
    unknown characters (e.g. Japanese) and re-spaces punctuation."""
    return re.sub(r"[^a-z0-9]", "", unicodedata.normalize("NFKD", text.lower()))


def fill_budget(entries: list[dict], paper: Paper) -> None:
    """Add rank-ordered token counts; truncate the chunk that crosses the budget."""
    used = 0
    for e in entries:
        enc = paper.tokenizer(
            paper.paper_text[e["chunk_start"] : e["chunk_end"]],
            add_special_tokens=False, return_offsets_mapping=True,
        )
        n = len(enc["input_ids"])
        e["full_tokens"] = n
        if used >= config.TOKEN_BUDGET or n == 0:
            e["status"], e["tokens"] = "out", 0
        elif used + n <= config.TOKEN_BUDGET:
            e["status"], e["tokens"] = "full", n
            used += n
        else:
            keep = config.TOKEN_BUDGET - used
            e["full_end"] = e["chunk_end"]
            e["chunk_end"] = e["chunk_start"] + enc["offset_mapping"][keep - 1][1]
            e["status"], e["tokens"] = "truncated", keep
            used = config.TOKEN_BUDGET
        e["cum_tokens"] = used


def search(client, collection, vector, paper_id, limit):
    for attempt in range(5):
        try:
            return client.query_points(
                collection, query=vector, limit=limit, with_payload=True,
                query_filter=models.Filter(must=[models.FieldCondition(
                    key="paper_id", match=models.MatchValue(value=paper_id))]),
                search_params=models.SearchParams(exact=True),
            ).points
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2**attempt)


def to_entries(strategy: str, hits, paper: Paper, checks: dict) -> list[dict]:
    entries = []
    if strategy == "hierarchical":
        seen = {}
        for hit in hits:  # rank order; first child of a parent is its best child
            p = hit.payload
            key = (p["parent_start"], p["parent_end"])
            if key not in seen:
                seen[key] = {"point_id": hit.id, "score": hit.score, "chunk_start": key[0],
                             "chunk_end": key[1], "num_children_hit": 0}
            seen[key]["num_children_hit"] += 1
        entries = list(seen.values())
    else:
        for hit in hits:
            p = hit.payload
            if strategy == "agentic_subset":
                span = [p["chunk_start"], p["chunk_end"]]
            elif strategy == "paragraph":
                span = paper.paragraph_span(p)
            elif strategy == "fixed":
                span, same = paper.fixed_span(p)
                checks["fixed_tokens_differ_from_paper_text"] += not same
                checks["fixed_text_mismatch"] += norm(paper.paper_text[span[0]:span[1]]) != norm(p["text"])
            else:  # recursive
                span, chunk = paper.recursive_span(p)
                checks["recursive_text_mismatch"] += norm(chunk) != norm(p["text"])
            entries.append({"point_id": hit.id, "score": hit.score,
                            "chunk_start": span[0], "chunk_end": span[1]})
    for rank, e in enumerate(entries, 1):
        e["rank"] = rank
    fill_budget(entries, paper)
    return entries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only first N questions")
    parser.add_argument("--strategies", nargs="+", default=list(STRATEGIES))
    args = parser.parse_args()

    items = json.loads((Path(config.EVAL_DIR) / "eval_set.json").read_text())
    if args.limit:
        items = items[: args.limit]
    out_dir = Path(config.RETRIEVAL_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = SentenceTransformer(config.EMBEDDING_MODEL)
    vectors = model.encode([config.QUERY_PREFIX + i["question"] for i in items],
                           batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    dataset = load_dataset(config.DATASET_NAME, split=config.DATASET_SPLIT, trust_remote_code=True)
    records = {p["id"]: p for p in dataset}
    papers: dict[str, Paper] = {}
    client = get_client()
    manifest = {"model": config.EMBEDDING_MODEL, "query_prefix": config.QUERY_PREFIX,
                "top_k": config.TOP_K, "top_k_children": config.TOP_K_CHILDREN,
                "token_budget": config.TOKEN_BUDGET, "search": "exact, filtered by paper_id",
                "budget_rule": "rank order; the chunk that crosses the budget is truncated to fit",
                "num_questions": len(items), "strategies": {}}

    for strategy in args.strategies:
        collection = STRATEGIES[strategy]
        rows = [(i, v) for i, v in zip(items, vectors)
                if strategy != "agentic_subset" or i["in_agentic_subset"]]
        k = config.TOP_K_CHILDREN if strategy == "hierarchical" else config.TOP_K
        start = time.perf_counter()
        with ThreadPoolExecutor(8) as pool:
            all_hits = list(tqdm(pool.map(
                lambda r: search(client, collection, r[1].tolist(), r[0]["paper_id"], k), rows),
                total=len(rows), desc=f"search {strategy}"))
        search_seconds = time.perf_counter() - start

        checks = {"fixed_tokens_differ_from_paper_text": 0, "fixed_text_mismatch": 0,
                  "recursive_text_mismatch": 0}
        with open(out_dir / f"{collection}.jsonl", "w") as f:
            for (item, _), hits in zip(rows, all_hits):
                pid = item["paper_id"]
                if pid not in papers:
                    papers[pid] = Paper(records[pid], model.tokenizer)
                entries = to_entries(strategy, hits, papers[pid], checks)
                f.write(json.dumps({"question_id": item["question_id"], "paper_id": pid,
                                    "num_retrieved": len(hits), "retrieved": entries}) + "\n")
        manifest["strategies"][strategy] = {"collection": collection, "num_questions": len(rows),
                                            "search_seconds": round(search_seconds, 1), "checks": checks}
        print(strategy, manifest["strategies"][strategy])

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
