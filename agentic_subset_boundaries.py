import argparse
import json
import time
from pathlib import Path

import nltk
import requests
from datasets import load_dataset
from transformers import AutoTokenizer
from nltk.tokenize import PunktTokenizer
from tqdm import tqdm

import config
from ingest_hierarchical import build_paper_text

SYSTEM_PROMPT = """You split scientific papers into chunks for a retrieval system.
You are shown numbered sentences from a paper. Decide after which sentences a chunk should END.

Rules:
- Each chunk must be self-contained and cover ONE topic, idea, method step or result.
- A line starting with "¶" begins a new paragraph; "##" lines are section titles. Paragraph ends are natural boundaries, but you may also split inside a long paragraph where the topic shifts, and you may merge short neighbouring paragraphs that belong together.
- Never separate a sentence from the sentence it depends on (e.g. a sentence starting with "This", "However", "These" or a table/figure reference stays with the previous sentence).
- Aim for chunks of roughly 100-300 tokens. The token count of each sentence is shown in parentheses.
- Do NOT list the final sentence shown; the last chunk may continue into the next window.
- Answer ONLY with JSON: {"boundaries": [ids]} where each id is the number of the LAST sentence of a chunk, in ascending order."""


def ask_llm(sentences: list[dict]) -> list[int]:
    """Return the sentence ids (from the window) after which the LLM ends a chunk."""
    lines, prev_section = [], object()
    for s in sentences:
        if s["section"] != prev_section and s["new_paragraph"] and s["section"]:
            lines.append(f"## {s['section']}")
        prev_section = s["section"]
        marker = "¶ " if s["new_paragraph"] else ""
        lines.append(f"[{s['id']}] {marker}{s['text']} ({s['tokens']} tok)")
    body = "\n".join(lines)
    resp = requests.post(
        f"{config.OLLAMA_URL}/api/chat",
        json={
            "model": config.OLLAMA_MODEL,
            "stream": False,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Sentences:\n{body}\n\nJSON:"},
            ],
            "format": {
                "type": "object",
                "properties": {"boundaries": {"type": "array", "items": {"type": "integer"}}},
                "required": ["boundaries"],
            },
            "options": {"temperature": 0, "num_ctx": 8192},
        },
        timeout=300,
    )
    resp.raise_for_status()
    ids = json.loads(resp.json()["message"]["content"])["boundaries"]
    valid = {s["id"] for s in sentences[:-1]}  # the last sentence of a window is never a boundary
    return sorted({i for i in ids if i in valid})


def get_sentences(paper: dict, sent_tok, tokenizer) -> tuple[str, list[dict]]:
    """Number every sentence of the paper with offsets into paper_text."""
    paper_text, parents = build_paper_text(paper)
    sentences = []
    for parent in parents:
        for k, (s, e) in enumerate(sent_tok.span_tokenize(parent["text"])):
            text = parent["text"][s:e]
            n = len(tokenizer(text, add_special_tokens=False)["input_ids"])
            if n == 0:
                continue
            sentences.append(
                {
                    "id": len(sentences),
                    "start": parent["start"] + s,
                    "end": parent["start"] + e,
                    "text": text,
                    "tokens": n,
                    "new_paragraph": k == 0,
                    "section": parent["section_name"],
                }
            )
    return paper_text, sentences


def llm_boundaries(sentences: list[dict]) -> tuple[list[int], int, int]:
    """Walk the paper in windows; return boundary ids, LLM call count and failure count."""
    boundaries, failures, calls, pos = [], 0, 0, 0
    while pos < len(sentences):
        window, total = [], 0
        for s in sentences[pos:]:
            if window and total + s["tokens"] > config.AGENTIC_WINDOW_TOKENS:
                break
            window.append(s)
            total += s["tokens"]
        is_last = pos + len(window) >= len(sentences)
        found = ask_safe(window)
        calls += 1
        if found is None:  # LLM unusable: fall back to paragraph ends in this window
            failures += 1
            found = [w["id"] for w, nxt in zip(window, window[1:]) if nxt["new_paragraph"]]
        if is_last:  # everything left is the final chunk
            boundaries += found
            break
        if not found:  # no boundary proposed: close the window so we always advance
            found = [window[-1]["id"]]
        boundaries += found
        pos = found[-1] + 1  # the unfinished tail is re-shown at the start of the next window
    return boundaries, calls, failures


def ask_safe(window: list[dict]) -> list[int] | None:
    for _ in range(2):
        try:
            return ask_llm(window)
        except (requests.RequestException, ValueError, KeyError):
            continue
    return None


def main() -> None:
    """Stage 1: ask the LLM for chunk boundaries on the first N train papers and record them."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=config.AGENTIC_SUBSET_SIZE)
    args = parser.parse_args()

    out_dir = Path(config.AGENTIC_ARTIFACTS_DIR)
    (out_dir / "boundaries").mkdir(parents=True, exist_ok=True)

    nltk.download("punkt_tab", quiet=True)
    sent_tok = PunktTokenizer("english")
    tokenizer = AutoTokenizer.from_pretrained(config.EMBEDDING_MODEL)
    dataset = load_dataset(
        config.DATASET_NAME, split=config.DATASET_SPLIT, trust_remote_code=True
    ).select(range(args.limit))

    records = []
    for paper in tqdm(dataset, desc="papers"):
        path = out_dir / "boundaries" / f"{paper['id']}.json"
        if path.exists():  # resume: keep earlier results and their recorded latency
            records.append(json.loads(path.read_text()))
            continue
        _, sentences = get_sentences(paper, sent_tok, tokenizer)
        start = time.perf_counter()
        boundaries, calls, failures = llm_boundaries(sentences)
        latency = time.perf_counter() - start
        record = {
            "paper_id": paper["id"],
            "title": paper["title"],
            "num_sentences": len(sentences),
            "num_tokens": sum(s["tokens"] for s in sentences),
            "boundaries": boundaries,  # id of the LAST sentence of each chunk (final chunk implicit)
            "llm_calls": calls,
            "llm_failed_windows": failures,
            "latency_seconds": round(latency, 2),
        }
        path.write_text(json.dumps(record, indent=1))
        records.append(record)

    lat = [r["latency_seconds"] for r in records]
    summary = {
        "model": config.OLLAMA_MODEL,
        "num_papers": len(records),
        "window_tokens": config.AGENTIC_WINDOW_TOKENS,
        "total_latency_seconds": round(sum(lat), 1),
        "mean_latency_seconds": round(sum(lat) / len(lat), 1),
        "max_latency_seconds": max(lat),
        "total_llm_calls": sum(r["llm_calls"] for r in records),
        "total_failed_windows": sum(r["llm_failed_windows"] for r in records),
        "total_boundaries": sum(len(r["boundaries"]) for r in records),
        "total_sentences": sum(r["num_sentences"] for r in records),
        "total_tokens": sum(r["num_tokens"] for r in records),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
