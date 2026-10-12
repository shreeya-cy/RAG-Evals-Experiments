import argparse
import json
import uuid
from pathlib import Path

import nltk
from datasets import load_dataset
from nltk.tokenize import PunktTokenizer
from qdrant_client import models
from tqdm import tqdm

import config
from embedding import add_model_arg, chunk_tokenizer, embed, load_embedder
from agentic_subset_boundaries import get_sentences
from qdrant_setup import get_client, setup_collection, upsert_with_retry

UPSERT_BATCH = 128


def build_chunks(paper_text: str, sentences: list[dict], boundaries: list[int]) -> list[dict]:
    """Group sentences by the recorded LLM boundaries; split any group over the token cap."""
    ends = set(boundaries)
    groups, cur = [], []
    for s in sentences:
        cur.append(s)
        if s["id"] in ends:
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)

    chunks = []
    for group in groups:
        pieces, piece, tokens = [], [], 0
        for s in group:
            if piece and tokens + s["tokens"] > config.AGENTIC_MAX_CHUNK_TOKENS:
                pieces.append(piece)
                piece, tokens = [], 0
            piece.append(s)
            tokens += s["tokens"]
        pieces.append(piece)
        for piece in pieces:
            start, end = piece[0]["start"], piece[-1]["end"]
            chunks.append(
                {
                    "text": paper_text[start:end],
                    "chunk_start": start,
                    "chunk_end": end,
                    "first_sentence": piece[0]["id"],
                    "last_sentence": piece[-1]["id"],
                    "chunk_token_count": sum(s["tokens"] for s in piece),
                    "num_sentences": len(piece),
                    "split_by_cap": len(pieces) > 1,
                }
            )
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recreate", action="store_true")
    add_model_arg(parser)
    args = parser.parse_args()

    boundary_dir = Path(config.AGENTIC_ARTIFACTS_DIR) / "boundaries"
    records = {p.stem: json.loads(p.read_text()) for p in boundary_dir.glob("*.json")}
    if not records:
        raise SystemExit("No recorded boundaries; run agentic_subset_boundaries.py first")

    model, spec = load_embedder(args.model)
    name = spec["collections"]["agentic_subset"]
    client = get_client()
    setup_collection(client, name=name, recreate=args.recreate, vector_size=spec["dim"])

    nltk.download("punkt_tab", quiet=True)
    sent_tok = PunktTokenizer("english")
    tokenizer = chunk_tokenizer()

    dataset = load_dataset(
        config.DATASET_NAME, split=config.DATASET_SPLIT, trust_remote_code=True
    ).filter(lambda p: p["id"] in records)

    all_chunks, total = 0, 0
    for paper in tqdm(dataset, desc="papers"):
        paper_text, sentences = get_sentences(paper, sent_tok, tokenizer)
        record = records[paper["id"]]
        assert record["num_sentences"] == len(sentences), "sentence split changed since boundaries were recorded"
        chunks = build_chunks(paper_text, sentences, record["boundaries"])
        vectors = embed(model, spec, [c["text"] for c in chunks])
        points = [
            models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"agentic:{paper['id']}:{i}")),
                vector=vec.tolist(),
                payload={"title": paper["title"], "paper_id": paper["id"], "chunk_index": i, **chunk},
            )
            for i, (chunk, vec) in enumerate(zip(chunks, vectors))
        ]
        for i in range(0, len(points), UPSERT_BATCH):
            upsert_with_retry(client, name, points[i : i + UPSERT_BATCH])
        total += len(points)

    print(f"Upserted {total} chunks from {len(dataset)} papers")
    print("Collection count:", client.count(name, exact=True).count)


if __name__ == "__main__":
    main()
