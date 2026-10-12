"""Embedding model selection shared by the ingest scripts."""
import argparse

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

import config


def add_model_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", choices=list(config.EMBEDDING_MODELS), default="bge")


def load_embedder(key: str) -> tuple[SentenceTransformer, dict]:
    spec = config.EMBEDDING_MODELS[key]
    model = SentenceTransformer(spec["name"])
    model.max_seq_length = 512
    return model, spec


def chunk_tokenizer():
    """Tokenizer that measures chunk sizes, the same for every embedding model."""
    return AutoTokenizer.from_pretrained(config.CHUNK_TOKENIZER)


def embed(model: SentenceTransformer, spec: dict, texts: list[str]):
    return model.encode(
        texts, batch_size=spec["batch_size"], normalize_embeddings=True, show_progress_bar=False
    )
