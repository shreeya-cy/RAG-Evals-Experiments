import argparse
import uuid

from datasets import load_dataset
from qdrant_client import models
from tqdm import tqdm

import config
from embedding import add_model_arg, chunk_tokenizer, embed, load_embedder
from qdrant_setup import get_client, setup_collection, upsert_with_retry

UPSERT_BATCH = 128


def build_document(paper: dict) -> str:
    """Title, abstract, then every section (name + paragraphs) as one long text."""
    parts = [paper["title"].strip(), paper["abstract"].strip()]
    full_text = paper["full_text"]
    for name, paragraphs in zip(full_text["section_name"], full_text["paragraphs"]):
        body = "\n".join(p.strip() for p in paragraphs if p and p.strip())
        section = f"{name.strip()}\n{body}".strip() if name else body
        if section:
            parts.append(section)
    return "\n\n".join(parts)


def chunk_text(text: str, tokenizer, size: int, overlap: int) -> list[str]:
    """Fixed-size sliding window over tokens; the last chunk is the remainder."""
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    step = size - overlap
    chunks = []
    for start in range(0, len(ids), step):
        window = ids[start : start + size]
        chunks.append(tokenizer.decode(window, skip_special_tokens=True))
        if start + size >= len(ids):
            break
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only first N papers")
    parser.add_argument("--recreate", action="store_true")
    add_model_arg(parser)
    args = parser.parse_args()

    model, spec = load_embedder(args.model)
    name = spec["collections"]["fixed"]
    client = get_client()
    setup_collection(client, name=name, recreate=args.recreate, vector_size=spec["dim"])

    tokenizer = chunk_tokenizer()

    dataset = load_dataset(
        config.DATASET_NAME, split=config.DATASET_SPLIT, trust_remote_code=True
    )
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    total = 0
    for paper in tqdm(dataset, desc="papers"):
        chunks = chunk_text(
            build_document(paper), tokenizer, config.CHUNK_SIZE, config.CHUNK_OVERLAP
        )
        vectors = embed(model, spec, chunks)
        points = [
            models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{paper['id']}:{i}")),
                vector=vec.tolist(),
                payload={
                    "title": paper["title"],
                    "paper_id": paper["id"],
                    "chunk_index": i,
                    "text": chunk,
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
