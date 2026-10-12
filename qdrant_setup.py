import argparse
import time

from qdrant_client import QdrantClient, models

import config


def get_client() -> QdrantClient:
    if not config.QDRANT_URL or not config.QDRANT_API_KEY:
        raise SystemExit("Set QDRANT_URL and QDRANT_API_KEY in .env")
    return QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)


def setup_collection(
    client: QdrantClient,
    name: str = config.COLLECTION_NAME,
    recreate: bool = False,
    vector_size: int = config.VECTOR_SIZE,
) -> None:
    exists = client.collection_exists(name)
    if exists and recreate:
        client.delete_collection(name)
        exists = False
    if exists:
        print(f"Collection '{name}' already exists")
        return
    client.create_collection(
        collection_name=name,
        vectors_config=models.VectorParams(
            size=vector_size, distance=models.Distance.COSINE
        ),
    )
    for field in ("paper_id", "title", "chunk_type", "section_name"):
        client.create_payload_index(
            collection_name=name,
            field_name=field,
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
    print(f"Created collection '{name}'")



def upsert_with_retry(client: QdrantClient, name: str, points, retries: int = 5) -> None:
    """Upsert a batch, retrying with backoff on transient network errors."""
    for attempt in range(retries):
        try:
            client.upsert(name, points=points)
            return
        except Exception as exc:  # network drops surface as several exception types
            if attempt == retries - 1:
                raise
            wait = 2**attempt
            print(f"upsert failed ({exc!r}); retrying in {wait}s")
            time.sleep(wait)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default=config.COLLECTION_NAME)
    parser.add_argument("--recreate", action="store_true", help="drop and recreate")
    parser.add_argument("--vector-size", type=int, default=config.VECTOR_SIZE)
    args = parser.parse_args()
    setup_collection(get_client(), name=args.name, recreate=args.recreate, vector_size=args.vector_size)
