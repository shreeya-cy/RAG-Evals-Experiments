"""Embedding model selection shared by the ingest and retrieval scripts."""
import argparse

import numpy as np
from transformers import AutoTokenizer

import config

OPENAI_BATCH = 256  # inputs per request (API limit 2048 inputs / 300k tokens)


def add_model_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", choices=list(config.EMBEDDING_MODELS), default="bge")


def load_embedder(key: str):
    """Return (model, spec). model is a SentenceTransformer, or an OpenAI client."""
    spec = config.EMBEDDING_MODELS[key]
    if spec.get("provider") == "openai":
        from openai import OpenAI

        return OpenAI(max_retries=8), spec
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(spec["name"])
    model.max_seq_length = 512
    return model, spec


def chunk_tokenizer():
    """Tokenizer that measures chunk sizes, the same for every embedding model."""
    return AutoTokenizer.from_pretrained(config.CHUNK_TOKENIZER)


def embed(model, spec: dict, texts: list[str]) -> np.ndarray:
    """L2-normalised vectors, one row per text."""
    if spec.get("provider") == "openai":
        texts = [t if t.strip() else "." for t in texts]  # the API rejects empty input
        rows = []
        for i in range(0, len(texts), OPENAI_BATCH):
            resp = model.embeddings.create(
                model=spec["name"], input=texts[i : i + OPENAI_BATCH], dimensions=spec["dim"]
            )
            rows.extend(d.embedding for d in sorted(resp.data, key=lambda d: d.index))
        vecs = np.array(rows, dtype=np.float32)
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    return model.encode(
        texts, batch_size=spec["batch_size"], normalize_embeddings=True, show_progress_bar=False
    )
