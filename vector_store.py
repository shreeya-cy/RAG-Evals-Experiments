"""One interface over the vector databases: a Qdrant collection or a Pinecone namespace.

store.setup(recreate), store.upsert(points), store.count(), store.search(vector, paper_id, limit).
Points are anything with .id, .vector and .payload (e.g. qdrant_client.models.PointStruct).
Search hits have .id, .score and .payload.
"""
import json
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from types import SimpleNamespace

from qdrant_client import models

import config
from qdrant_setup import get_client, setup_collection, upsert_with_retry


class QdrantStore:
    UPSERT_BATCH = 128

    def __init__(self, collection: str, dim: int):
        self.collection, self.dim = collection, dim
        self.client = get_client()

    def setup(self, recreate: bool = False) -> None:
        setup_collection(self.client, name=self.collection, recreate=recreate, vector_size=self.dim)

    def upsert(self, points) -> None:
        for i in range(0, len(points), self.UPSERT_BATCH):
            upsert_with_retry(self.client, self.collection, points[i : i + self.UPSERT_BATCH])

    def count(self, expected: int | None = None) -> int:
        return self.client.count(self.collection, exact=True).count

    def search(self, vector, paper_id: str, limit: int):
        for attempt in range(5):
            try:
                return self.client.query_points(
                    self.collection, query=vector, limit=limit, with_payload=True,
                    query_filter=models.Filter(must=[models.FieldCondition(
                        key="paper_id", match=models.MatchValue(value=paper_id))]),
                    search_params=models.SearchParams(exact=True),
                ).points
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2**attempt)


class WriteUnitLedger:
    """Running total of Pinecone write units this month, kept on disk across runs.

    Upsert cost is 1 WU per KB of request (min 5), plus the same again for each record that
    replaces an existing one. This only guards against blowing the free monthly allowance.
    """

    def __init__(self, path: str = config.PINECONE_LEDGER, cap: int = config.PINECONE_WU_CAP):
        self.path, self.cap, self.lock = Path(path), cap, threading.Lock()

    def _load(self) -> dict:
        month = date.today().strftime("%Y-%m")
        try:
            data = json.loads(self.path.read_text())
            if data.get("month") == month:
                return data
        except (OSError, ValueError):
            pass
        return {"month": month, "write_units": 0}

    def total(self) -> int:
        with self.lock:
            return self._load()["write_units"]

    def charge(self, units: int) -> None:
        with self.lock:
            data = self._load()
            if data["write_units"] + units > self.cap:
                raise SystemExit(
                    f"Stopping: {data['write_units']} + {units} write units would pass the "
                    f"safety cap of {self.cap} (free plan: 2,000,000 per month)"
                )
            data["write_units"] += units
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(data))


class PineconeStore:
    UPSERT_BATCH = 100
    WORKERS = 4
    _index_lock = threading.Lock()

    def __init__(self, index_name: str, namespace: str, dim: int):
        from pinecone import Pinecone

        self.index_name, self.namespace, self.dim = index_name, namespace, dim
        self.pc = Pinecone(api_key=config.PINECONE_API_KEY)
        self.index = None
        self.ledger = WriteUnitLedger()
        self.pool = ThreadPoolExecutor(self.WORKERS)
        self.pending: list = []

    def _open_index(self, create: bool):
        from pinecone import ServerlessSpec

        if not self.pc.has_index(self.index_name):
            if not create:
                raise SystemExit(f"Pinecone index '{self.index_name}' does not exist")
            self.pc.create_index(
                name=self.index_name, dimension=self.dim, metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),  # the only free-plan region
            )
            print(f"Created Pinecone index '{self.index_name}'")
        desc = self.pc.describe_index(self.index_name)
        if desc.dimension != self.dim:
            raise SystemExit(f"Index '{self.index_name}' has dimension {desc.dimension}, expected {self.dim}")
        self.index = self.pc.Index(host=desc.host)

    def setup(self, recreate: bool = False) -> None:
        with self._index_lock:
            self._open_index(create=True)
        if recreate:
            try:
                self.index.delete(delete_all=True, namespace=self.namespace)
            except Exception as exc:  # a namespace that does not exist yet raises 404
                if "404" not in str(exc) and "not found" not in str(exc).lower():
                    raise

    def _record(self, point) -> dict:
        meta = {k: v for k, v in point.payload.items()
                if v is not None and k not in config.PINECONE_DROP_FIELDS}
        return {"id": point.id, "values": point.vector, "metadata": meta}

    @staticmethod
    def _kb(record: dict) -> int:
        size = len(record["id"]) + len(json.dumps(record["metadata"])) + len(record["values"]) * 4
        return math.ceil(size / 1024)

    def _send(self, records: list[dict]) -> None:
        self.ledger.charge(max(5, sum(self._kb(r) for r in records)))
        for attempt in range(6):
            try:
                self.index.upsert(vectors=records, namespace=self.namespace)
                return
            except Exception as exc:
                if attempt == 5:
                    raise
                wait = 2**attempt
                print(f"pinecone upsert failed ({exc!r}); retrying in {wait}s")
                time.sleep(wait)

    def upsert(self, points) -> None:
        if self.index is None:
            self._open_index(create=False)
        records = [self._record(p) for p in points]
        for i in range(0, len(records), self.UPSERT_BATCH):
            self.pending.append(self.pool.submit(self._send, records[i : i + self.UPSERT_BATCH]))
        self.pending = [f for f in self.pending if not (f.done() and f.result() is None)]  # re-raises errors

    def flush(self) -> None:
        for f in self.pending:
            f.result()
        self.pending = []

    def count(self, expected: int | None = None) -> int:
        """Vectors in this namespace. Stats lag behind writes, so wait for `expected` if given."""
        self.flush()
        for _ in range(40):
            ns = self.index.describe_index_stats().namespaces.get(self.namespace)
            n = ns.vector_count if ns else 0
            if expected is None or n >= expected:
                return n
            time.sleep(3)
        return n

    def search(self, vector, paper_id: str, limit: int):
        if self.index is None:
            self._open_index(create=False)
        for attempt in range(5):
            try:
                res = self.index.query(
                    vector=vector, top_k=limit, namespace=self.namespace,
                    filter={"paper_id": {"$eq": paper_id}}, include_metadata=True,
                )
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2**attempt)
        # Pinecone returns every number as a float; ints are what the rest of the code expects.
        fix = lambda v: int(v) if isinstance(v, float) and v.is_integer() else v
        return [SimpleNamespace(id=m.id, score=m.score,
                                payload={k: fix(v) for k, v in (m.metadata or {}).items()})
                for m in res.matches]


def open_store(spec: dict, strategy: str, recreate: bool = False, setup: bool = True):
    """The store for one chunking strategy: a Qdrant collection or a Pinecone namespace."""
    target = spec["collections"][strategy]
    if spec.get("store") == "pinecone":
        store = PineconeStore(spec["index"], target, spec["dim"])
    else:
        store = QdrantStore(target, spec["dim"])
    if setup:
        store.setup(recreate=recreate)
    return store
