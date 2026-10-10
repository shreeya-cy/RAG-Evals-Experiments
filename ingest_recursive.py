import argparse
import uuid

from datasets import load_dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import models
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import config
from qdrant_setup import get_client, setup_collection, upsert_with_retry

UPSERT_BATCH = 128


def build_document(paper: dict) -> str:
    """Title, abstract, then each section name and paragraph as "\\n\\n"-separated blocks."""
    parts = [paper["title"].strip(), paper["abstract"].strip()]
    full_text = paper["full_text"]
    for name, paragraphs in zip(full_text["section_name"], full_text["paragraphs"]):
        if name and name.strip():
            parts.append(name.strip())
        parts.extend(p.strip() for p in paragraphs if p and p.strip())
    return "\n\n".join(p for p in parts if p)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only first N papers")
    parser.add_argument("--recreate", action="store_true")
    args = parser.parse_args()

    name = config.RECURSIVE_COLLECTION_NAME
    client = get_client()
    setup_collection(client, name=name, recreate=args.recreate)

    model = SentenceTransformer(config.EMBEDDING_MODEL)
    model.max_seq_length = 512

    # Chunk size/overlap are measured in bge tokens, as in the fixed-size strategy.
    # keep_separator="end" leaves the ". " at the end of a sentence, not the start of the next.
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        model.tokenizer,
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=config.RECURSIVE_SEPARATORS,
        keep_separator="end",
    )

    dataset = load_dataset(
        config.DATASET_NAME, split=config.DATASET_SPLIT, trust_remote_code=True
    )
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    total = 0
    for paper in tqdm(dataset, desc="papers"):
        chunks = splitter.split_text(build_document(paper))
        vectors = model.encode(
            chunks, batch_size=64, normalize_embeddings=True, show_progress_bar=False
        )
        points = [
            models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"rec:{paper['id']}:{i}")),
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
