# RAG-Evals-Experiments

Knowledge sources built from QASPER (train split), embedded with `BAAI/bge-small-en-v1.5`
and stored in Qdrant Cloud.

| Chunking strategy | Script | Collection |
|---|---|---|
| Fixed size (256 tokens, 32 overlap) over title + abstract + full text | `ingest_fixed_size.py` | `qasper_train_bge_small_256` |
| One chunk per paragraph (+ title/abstract chunk) | `ingest_paragraph.py` | `qasper_train_bge_small_paragraph` |
| Recursive (`\n\n` → `. ` → ` `), 256 tokens, 32 overlap, LangChain `RecursiveCharacterTextSplitter` | `ingest_recursive.py` | `qasper_train_bge_small_recursive_256` |
| Hierarchical: parent = paragraph, child = merged NLTK Punkt sentences (>=30 tokens, long sentences cut into 128-token windows) | `ingest_hierarchical.py` | `qasper_train_bge_small_hierarchical` |

```
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in QDRANT_URL and QDRANT_API_KEY
python ingest_fixed_size.py --limit 5   # smoke test (creates the collection if missing)
python ingest_fixed_size.py             # full train split
python ingest_paragraph.py              # paragraph-level collection
python ingest_recursive.py              # recursive-chunk collection
python ingest_hierarchical.py           # hierarchical (parent/child) collection
```

`qdrant_setup.py --name <collection> [--recreate]` creates a collection on its own.
