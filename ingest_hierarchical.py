import argparse
import uuid

import nltk
from datasets import load_dataset
from nltk.tokenize import PunktTokenizer
from qdrant_client import models
from tqdm import tqdm

import config
from embedding import add_model_arg, chunk_tokenizer, embed, load_embedder
from qdrant_setup import get_client, setup_collection, upsert_with_retry

UPSERT_BATCH = 128


def build_paper_text(paper: dict) -> tuple[str, list[dict]]:
    """Return paper_text and its parents (one per paragraph) with offsets into it.

    paper_text is "\\n\\n"-separated blocks: the abstract paragraph (title attached
    at its start), then for each section its name followed by its paragraphs.
    Section names are part of paper_text but are not parents.
    """
    blocks: list[tuple[str, dict | None]] = []
    title = paper["title"].strip()
    blocks.append(
        (
            f"{title}\n{paper['abstract'].strip()}",
            {"section_name": "abstract", "section_index": None, "paragraph_index": None},
        )
    )
    full_text = paper["full_text"]
    for s_idx, (name, paragraphs) in enumerate(
        zip(full_text["section_name"], full_text["paragraphs"])
    ):
        name = (name or "").strip()
        if name:
            blocks.append((name, None))
        for p_idx, para in enumerate(paragraphs):
            if para and para.strip():
                meta = {"section_name": name, "section_index": s_idx, "paragraph_index": p_idx}
                blocks.append((para.strip(), meta))

    paper_text, parents, pos = "", [], 0
    for text, meta in blocks:
        if paper_text:
            paper_text += "\n\n"
            pos += 2
        if meta is not None:
            parents.append({"start": pos, "end": pos + len(text), **meta})
        paper_text += text
        pos += len(text)
    for parent in parents:
        parent["text"] = paper_text[parent["start"] : parent["end"]]
    return paper_text, parents


def split_children(para: str, sent_tok, tokenizer) -> list[dict]:
    """Children of one paragraph as spans relative to the paragraph.

    Sentences are merged left to right until a child has >= CHILD_MIN_TOKENS tokens.
    A sentence over CHILD_MAX_SENTENCE_TOKENS becomes 128-token windows. Leftover
    short text at the end of a paragraph is appended to the previous child, unless
    that child is a window of a long sentence (then it stays on its own).
    """

    def n_tokens(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    children: list[dict] = []
    buf: dict | None = None

    def flush_leftover() -> None:
        nonlocal buf
        if buf is None:
            return
        if children and not children[-1]["window"]:
            children[-1]["end"] = buf["end"]
            children[-1]["tokens"] += buf["tokens"]
        else:
            children.append(buf)
        buf = None

    for s, e in sent_tok.span_tokenize(para):
        sentence = para[s:e]
        enc = tokenizer(sentence, add_special_tokens=False, return_offsets_mapping=True)
        n = len(enc["input_ids"])
        if n == 0:
            continue
        if n > config.CHILD_MAX_SENTENCE_TOKENS:
            flush_leftover()
            offsets, step = enc["offset_mapping"], config.CHILD_MAX_SENTENCE_TOKENS
            for i in range(0, n, step):
                j = min(i + step, n)
                children.append(
                    {"start": s + offsets[i][0], "end": s + offsets[j - 1][1],
                     "tokens": j - i, "window": True}
                )
            continue
        if buf is None:
            buf = {"start": s, "end": e, "tokens": n, "window": False}
        else:
            buf["end"] = e
            buf["tokens"] += n
        if buf["tokens"] >= config.CHILD_MIN_TOKENS:
            children.append(buf)
            buf = None
    flush_leftover()
    return children


def build_children(paper: dict, sent_tok, tokenizer) -> list[dict]:
    paper_text, parents = build_paper_text(paper)
    out = []
    for parent_index, parent in enumerate(parents):
        for child in split_children(parent["text"], sent_tok, tokenizer):
            start = parent["start"] + child["start"]
            end = parent["start"] + child["end"]
            out.append(
                {
                    "text": paper_text[start:end],
                    "child_start": start,
                    "child_end": end,
                    "child_token_count": child["tokens"],
                    "parent_index": parent_index,
                    "parent_start": parent["start"],
                    "parent_end": parent["end"],
                    "parent_text": parent["text"],
                    "section_name": parent["section_name"],
                    "section_index": parent["section_index"],
                    "paragraph_index": parent["paragraph_index"],
                }
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only first N papers")
    parser.add_argument("--recreate", action="store_true")
    add_model_arg(parser)
    args = parser.parse_args()

    model, spec = load_embedder(args.model)
    name = spec["collections"]["hierarchical"]
    client = get_client()
    setup_collection(client, name=name, recreate=args.recreate, vector_size=spec["dim"])

    nltk.download("punkt_tab", quiet=True)
    sent_tok = PunktTokenizer("english")

    tokenizer = chunk_tokenizer()

    dataset = load_dataset(
        config.DATASET_NAME, split=config.DATASET_SPLIT, trust_remote_code=True
    )
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    total = 0
    for paper in tqdm(dataset, desc="papers"):
        children = build_children(paper, sent_tok, tokenizer)
        vectors = embed(model, spec, [c["text"] for c in children])
        points = [
            models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"hier:{paper['id']}:{i}")),
                vector=vec.tolist(),
                payload={
                    "title": paper["title"],
                    "paper_id": paper["id"],
                    "child_index": i,
                    **child,
                },
            )
            for i, (child, vec) in enumerate(zip(children, vectors))
        ]
        for i in range(0, len(points), UPSERT_BATCH):
            upsert_with_retry(client, name, points[i : i + UPSERT_BATCH])
        total += len(points)

    print(f"Upserted {total} children from {len(dataset)} papers")
    print("Collection count:", client.count(name, exact=True).count)


if __name__ == "__main__":
    main()
