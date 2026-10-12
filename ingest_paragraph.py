import argparse
import uuid

from datasets import load_dataset
from qdrant_client import models
from tqdm import tqdm

import config
from embedding import add_model_arg, embed, load_embedder
from qdrant_setup import get_client, setup_collection, upsert_with_retry

UPSERT_BATCH = 128


def build_chunks(paper: dict) -> list[dict]:
    """One chunk for title+abstract, then one chunk per non-empty paragraph."""
    title = paper["title"].strip()
    chunks = [
        {
            "chunk_type": "abstract",
            "section_name": "abstract",
            "section_index": None,
            "paragraph_index": None,
            "text": f"{title}\n{paper['abstract'].strip()}",
        }
    ]
    full_text = paper["full_text"]
    for s_idx, (name, paragraphs) in enumerate(
        zip(full_text["section_name"], full_text["paragraphs"])
    ):
        for p_idx, para in enumerate(paragraphs):
            if not para or not para.strip():
                continue
            chunks.append(
                {
                    "chunk_type": "paragraph",
                    "section_name": (name or "").strip(),
                    "section_index": s_idx,
                    "paragraph_index": p_idx,
                    "text": para.strip(),
                }
            )
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only first N papers")
    parser.add_argument("--recreate", action="store_true")
    add_model_arg(parser)
    args = parser.parse_args()

    model, spec = load_embedder(args.model)
    name = spec["collections"]["paragraph"]
    client = get_client()
    setup_collection(client, name=name, recreate=args.recreate, vector_size=spec["dim"])

    dataset = load_dataset(
        config.DATASET_NAME, split=config.DATASET_SPLIT, trust_remote_code=True
    )
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    total = 0
    for paper in tqdm(dataset, desc="papers"):
        chunks = build_chunks(paper)
        vectors = embed(model, spec, [c["text"] for c in chunks])
        points = [
            models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"para:{paper['id']}:{i}")),
                vector=vec.tolist(),
                payload={
                    "title": paper["title"],
                    "paper_id": paper["id"],
                    "chunk_index": i,
                    **chunk,
                },
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
