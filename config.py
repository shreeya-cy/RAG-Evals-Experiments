import os

from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")

COLLECTION_NAME = "qasper_train_bge_small_256"
PARAGRAPH_COLLECTION_NAME = "qasper_train_bge_small_paragraph"
RECURSIVE_COLLECTION_NAME = "qasper_train_bge_small_recursive_256"
HIERARCHICAL_COLLECTION_NAME = "qasper_train_bge_small_hierarchical"
AGENTIC_SUBSET_COLLECTION_NAME = "qasper_train_bge_small_agentic_subset"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
VECTOR_SIZE = 384

CHUNK_SIZE = 256  # tokens
CHUNK_OVERLAP = 32  # tokens

DATASET_NAME = "allenai/qasper"
DATASET_SPLIT = "train"

RECURSIVE_SEPARATORS = ["\n\n", ". ", " "]  # paragraph -> sentence -> word

CHILD_MIN_TOKENS = 30  # merge neighbouring sentences until a child has at least this many
CHILD_MAX_SENTENCE_TOKENS = 128  # longer sentences are split into windows of this size

OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:7b"
AGENTIC_WINDOW_TOKENS = 1500  # sentence tokens shown to the LLM per call
AGENTIC_MAX_CHUNK_TOKENS = 400  # safety cap (bge reads at most 512); longer chunks are split at sentence boundaries
AGENTIC_SUBSET_SIZE = 50  # first N train papers
AGENTIC_ARTIFACTS_DIR = "artifacts/agentic_subset"

EVAL_DIR = "artifacts/eval"

RETRIEVAL_DIR = "artifacts/retrieval"
TOP_K = 50  # chunks retrieved per question (fixed, paragraph, recursive, agentic)
TOP_K_CHILDREN = 100  # hierarchical: children retrieved before deduplicating to parents
TOKEN_BUDGET = 1024
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

METRICS_DIR = "artifacts/metrics"
PARAGRAPH_COVERAGE_THRESHOLD = 0.8


# Embedding models. Chunking always uses the bge tokenizer (CHUNK_TOKENIZER) so every model
# embeds identical chunks; only the vectors differ.
CHUNK_TOKENIZER = EMBEDDING_MODEL
QWEN_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
QWEN_QUERY_PROMPT = (
    "Instruct: Given a question about a scientific paper, retrieve the passages that answer it\n"
    "Query: "
)
EMBEDDING_MODELS = {
    "bge": {
        "name": EMBEDDING_MODEL,
        "dim": VECTOR_SIZE,
        "batch_size": 64,
        "retrieval_dir": RETRIEVAL_DIR,
        "metrics_dir": METRICS_DIR,
        "query_prefix": QUERY_PREFIX,
        "collections": {
            "fixed": COLLECTION_NAME,
            "paragraph": PARAGRAPH_COLLECTION_NAME,
            "recursive": RECURSIVE_COLLECTION_NAME,
            "hierarchical": HIERARCHICAL_COLLECTION_NAME,
            "agentic_subset": AGENTIC_SUBSET_COLLECTION_NAME,
        },
    },
    "qwen": {
        "name": QWEN_EMBEDDING_MODEL,
        "dim": 1024,
        "batch_size": 16,
        "retrieval_dir": "artifacts/retrieval_qwen3",
        "metrics_dir": "artifacts/metrics_qwen3",
        "query_prefix": QWEN_QUERY_PROMPT,
        "collections": {
            "fixed": "qasper_train_qwen3_06b_256",
            "paragraph": "qasper_train_qwen3_06b_paragraph",
            "recursive": "qasper_train_qwen3_06b_recursive_256",
            "hierarchical": "qasper_train_qwen3_06b_hierarchical",
            "agentic_subset": "qasper_train_qwen3_06b_agentic_subset",
        },
    },
}
