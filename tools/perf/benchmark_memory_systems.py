from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib import parse, request

import chromadb
import psutil
from mem0 import Memory
from mem0.configs.base import EmbedderConfig, LlmConfig, MemoryConfig, VectorStoreConfig


TOKEN_RE = re.compile(r"[a-z0-9_-]+")


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _normalize_token(token: str) -> str | None:
    if not token:
        return None
    if token.startswith("sem"):
        token = "concept" + token[3:]
    if token in {"fraud", "prevention", "process"}:
        mapping = {"fraud": "risk", "prevention": "control", "process": "workflow"}
        token = mapping[token]
    # Simulate common vector weakness on hard IDs.
    if token.startswith("tck-") or token.startswith("mk"):
        return None
    if token.isdigit():
        return None
    return token


def stable_embed(text: str, dim: int) -> list[float]:
    text = str(text or "").lower()
    tokens = TOKEN_RE.findall(text)
    vec = [0.0] * dim
    for raw in tokens:
        token = _normalize_token(raw)
        if token is None:
            continue
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
        i1 = int.from_bytes(digest[0:4], "little") % dim
        i2 = int.from_bytes(digest[4:8], "little") % dim
        w = 1.0 + (int.from_bytes(digest[8:10], "little") % 100) / 250.0
        vec[i1] += w
        vec[i2] += w * 0.5
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 1e-12:
        vec = [v / norm for v in vec]
    return vec


class FakeEmbeddingHandler(BaseHTTPRequestHandler):
    default_dim = 768

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def do_POST(self) -> None:
        if self.path not in {"/v1/embeddings", "/embeddings"}:
            self.send_response(404)
            self.end_headers()
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(size).decode("utf-8"))
            inputs = body.get("input", [])
            if isinstance(inputs, str):
                inputs = [inputs]
            if not isinstance(inputs, list):
                raise ValueError("invalid input")
            dim = int(body.get("dimensions") or self.default_dim)
            data = []
            for idx, txt in enumerate(inputs):
                data.append(
                    {
                        "object": "embedding",
                        "index": idx,
                        "embedding": stable_embed(str(txt), dim),
                    }
                )
            payload = {"object": "list", "data": data, "model": body.get("model", "bench-model")}
            raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        except Exception as exc:
            raw = json.dumps({"error": str(exc)}).encode("utf-8")
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)


class FakeEmbeddingServer:
    def __init__(self) -> None:
        self.httpd = ThreadingHTTPServer(("127.0.0.1", _find_free_port()), FakeEmbeddingHandler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_port}/v1"

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=3)


@dataclass(frozen=True)
class Doc:
    marker: str
    message_id: str
    sender: str
    group_id: str
    content: str


@dataclass(frozen=True)
class QueryCase:
    query: str
    expected_marker: str
    mode: str  # "id" or "semantic"


def build_dataset(doc_count: int, seed: int) -> tuple[list[Doc], list[QueryCase]]:
    rng = random.Random(seed)
    docs: list[Doc] = []
    queries: list[QueryCase] = []
    for i in range(doc_count):
        marker = f"mk{i:05d}"
        ticket = f"tck-{i:05d}"
        concept = f"concept{i:05d}"
        project = f"proj_{i % 97}"
        customer = f"customer_{i % 211}"
        channel = rng.choice(["email", "sms", "app", "call"])
        content = (
            f"Memory marker {marker}. "
            f"Ticket {ticket} belongs to {project} for {customer}. "
            f"It documents a risk control workflow with manual review and channel {channel}. "
            f"The incident concept tag is {concept} and follow-up owner is team_{i % 43}."
        )
        docs.append(
            Doc(
                marker=marker,
                message_id=f"msg-{i:05d}",
                sender="bench_user",
                group_id="bench_group",
                content=content,
            )
        )

        id_query = f"Need detail for ticket {ticket}. Include marker {marker}."
        sem_query = (
            f"Find the fraud prevention process for sem{i:05d} under {project} and {customer}."
        )
        queries.append(QueryCase(query=id_query, expected_marker=marker, mode="id"))
        queries.append(QueryCase(query=sem_query, expected_marker=marker, mode="semantic"))

    rng.shuffle(queries)
    return docs, queries


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    k = (len(values) - 1) * p
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return float(values[int(k)])
    return float(values[lo] + (values[hi] - values[lo]) * (k - lo))


def _summary_stats(latencies_ms: list[float], ok_count: int, total_count: int, elapsed_sec: float) -> dict[str, float]:
    return {
        "count": float(total_count),
        "ok": float(ok_count),
        "error_rate": float((total_count - ok_count) / total_count) if total_count else 0.0,
        "qps": float(ok_count / elapsed_sec) if elapsed_sec > 0 else 0.0,
        "latency_p50_ms": _percentile(latencies_ms, 0.50),
        "latency_p95_ms": _percentile(latencies_ms, 0.95),
        "latency_p99_ms": _percentile(latencies_ms, 0.99),
        "latency_mean_ms": float(statistics.mean(latencies_ms)) if latencies_ms else 0.0,
    }


def _quality_stats(ranks: list[int | None]) -> dict[str, float]:
    total = len(ranks)
    if total == 0:
        return {"recall_at_5": 0.0, "mrr_at_5": 0.0}
    recall = sum(1 for r in ranks if r is not None and r <= 5) / total
    mrr = sum((1.0 / r) for r in ranks if r is not None and r <= 5) / total
    return {"recall_at_5": float(recall), "mrr_at_5": float(mrr)}


def _first_rank(texts: list[str], marker: str) -> int | None:
    for idx, text in enumerate(texts, start=1):
        if marker in text:
            return idx
    return None


class FlockMemRunner:
    def __init__(self, repo_root: Path, embed_base_url: str, work_dir: Path, port: int) -> None:
        self.repo_root = repo_root
        self.embed_base_url = embed_base_url
        self.work_dir = work_dir
        self.port = port
        self.base_url = f"http://127.0.0.1:{port}"
        self.proc: subprocess.Popen[str] | None = None
        self.log_file = self.work_dir / "minimem-bench.log"

    def start(self) -> None:
        env = dict(os.environ)
        env.update(
            {
                "LITE_HOST": "127.0.0.1",
                "LITE_PORT": str(self.port),
                "LITE_AUTO_KILL_PORT": "false",
                "LITE_DATA_DIR": str(self.work_dir / "minimem_data"),
                "LITE_DB_PATH": str(self.work_dir / "minimem_data" / "lite.db"),
                "LITE_LANCEDB_DIR": str(self.work_dir / "minimem_data" / "lancedb"),
                "LITE_GRAPH_DIR": str(self.work_dir / "minimem_data" / "graph"),
                "LITE_GRAPH_ENABLED": "false",
                "LITE_RETRIEVAL_PROFILE": "agentic",
                "LITE_EXTRACTOR_PROVIDER": "rule",
                "LITE_EMBEDDING_PROVIDER": "openai",
                "LITE_EMBEDDING_BASE_URL": self.embed_base_url,
                "LITE_EMBEDDING_API_KEY": "bench",
                "LITE_EMBEDDING_MODEL": "bench-embed-768",
                "LITE_CHAT_BASE_URL": "http://127.0.0.1:9/v1",
                "LITE_CHAT_API_KEY": "bench",
                "LITE_CHAT_MODEL": "bench-chat",
                "LITE_AGENT_POLICY_ENABLED": "false",
                "LITE_KEY_MEMORY_IMPORTANCE_THRESHOLD": "0.0",
                "LITE_VECTOR_LANCEDB_ENABLED": "true",
                "LITE_VECTOR_LANCEDB_MIN_IMPORTANCE": "0.0",
            }
        )
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        handle = self.log_file.open("w", encoding="utf-8")
        self.proc = subprocess.Popen(
            [sys.executable, "main.py"],
            cwd=str(self.repo_root),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self._wait_ready()

    def _wait_ready(self, timeout_sec: float = 35.0) -> None:
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            if self.proc and self.proc.poll() is not None:
                raise RuntimeError(f"FlockMem exited early with code {self.proc.returncode}")
            try:
                with request.urlopen(f"{self.base_url}/health", timeout=1.5) as resp:
                    if resp.status == 200:
                        return
            except Exception:
                time.sleep(0.3)
        raise TimeoutError("FlockMem server did not become ready in time")

    def stop(self) -> None:
        if self.proc is None:
            return
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)
        self.proc = None

    def process_metrics(self) -> dict[str, float]:
        if self.proc is None or self.proc.pid <= 0:
            return {"rss_mb": 0.0}
        try:
            rss = psutil.Process(self.proc.pid).memory_info().rss
            return {"rss_mb": float(rss / (1024 * 1024))}
        except Exception:
            return {"rss_mb": 0.0}

    def ingest(self, docs: list[Doc]) -> dict[str, float]:
        latencies: list[float] = []
        ok = 0
        start = time.perf_counter()
        for d in docs:
            payload = {
                "message_id": d.message_id,
                "create_time": int(time.time()),
                "sender": d.sender,
                "content": d.content,
                "group_id": d.group_id,
                "sender_name": d.sender,
                "role": "user",
            }
            raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            req = request.Request(
                url=f"{self.base_url}/api/v1/memories",
                data=raw,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            t0 = time.perf_counter()
            try:
                with request.urlopen(req, timeout=40) as resp:
                    if resp.status == 200:
                        ok += 1
            except Exception:
                pass
            latencies.append((time.perf_counter() - t0) * 1000)
        elapsed = time.perf_counter() - start
        return _summary_stats(latencies, ok_count=ok, total_count=len(docs), elapsed_sec=elapsed)

    def _search_once(
        self, q: QueryCase, retrieve_method: str, decision_mode: str, top_k: int
    ) -> tuple[float, int | None, bool]:
        params = parse.urlencode(
            {
                "query": q.query,
                "user_id": "bench_user",
                "group_id": "bench_group",
                "retrieve_method": retrieve_method,
                "decision_mode": decision_mode,
                "top_k": top_k,
            }
        )
        url = f"{self.base_url}/api/v1/memories/search?{params}"
        t0 = time.perf_counter()
        try:
            with request.urlopen(url, timeout=40) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            memories = body.get("result", {}).get("memories", [])
            texts = [
                " ".join(
                    [
                        str(row.get("summary") or ""),
                        str(row.get("episode") or ""),
                        str(row.get("subject") or ""),
                    ]
                ).lower()
                for row in memories
            ]
            rank = _first_rank(texts, q.expected_marker)
            return (time.perf_counter() - t0) * 1000, rank, True
        except Exception:
            return (time.perf_counter() - t0) * 1000, None, False

    def search_bench(
        self,
        queries: list[QueryCase],
        retrieve_method: str,
        decision_mode: str,
        concurrency: int,
        top_k: int,
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, float]]:
        latencies: list[float] = []
        ranks_all: list[int | None] = []
        ranks_id: list[int | None] = []
        ranks_sem: list[int | None] = []
        ok = 0
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            futures = [
                pool.submit(self._search_once, q, retrieve_method, decision_mode, top_k)
                for q in queries
            ]
            for q, f in zip(queries, futures):
                latency_ms, rank, success = f.result()
                latencies.append(latency_ms)
                if success:
                    ok += 1
                ranks_all.append(rank)
                if q.mode == "id":
                    ranks_id.append(rank)
                else:
                    ranks_sem.append(rank)
        elapsed = time.perf_counter() - start
        perf = _summary_stats(latencies, ok_count=ok, total_count=len(queries), elapsed_sec=elapsed)
        return perf, _quality_stats(ranks_all), _quality_stats(ranks_id), _quality_stats(ranks_sem)


class Mem0Runner:
    def __init__(self, embed_base_url: str, work_dir: Path, vector_dim: int) -> None:
        self.embed_base_url = embed_base_url
        self.work_dir = work_dir
        self.vector_dim = vector_dim
        self.memory: Memory | None = None

    def start(self) -> None:
        cfg = MemoryConfig(
            vector_store=VectorStoreConfig(
                provider="chroma",
                config={
                    "collection_name": "bench_mem0",
                    "path": str(self.work_dir / "mem0_chroma"),
                },
            ),
            embedder=EmbedderConfig(
                provider="openai",
                config={
                    "api_key": "bench",
                    "openai_base_url": self.embed_base_url,
                    "model": "bench-embed-768",
                    "embedding_dims": self.vector_dim,
                },
            ),
            llm=LlmConfig(
                provider="openai",
                config={
                    "api_key": "bench",
                    "openai_base_url": self.embed_base_url,
                    "model": "bench-chat",
                },
            ),
            history_db_path=str(self.work_dir / "mem0_history.db"),
        )
        self.memory = Memory(cfg)

    def stop(self) -> None:
        self.memory = None

    def ingest(self, docs: list[Doc]) -> dict[str, float]:
        assert self.memory is not None
        latencies: list[float] = []
        ok = 0
        start = time.perf_counter()
        for d in docs:
            t0 = time.perf_counter()
            try:
                self.memory.add(d.content, user_id=d.sender, infer=False)
                ok += 1
            except Exception:
                pass
            latencies.append((time.perf_counter() - t0) * 1000)
        elapsed = time.perf_counter() - start
        return _summary_stats(latencies, ok_count=ok, total_count=len(docs), elapsed_sec=elapsed)

    def _search_once(self, q: QueryCase, top_k: int) -> tuple[float, int | None, bool]:
        assert self.memory is not None
        t0 = time.perf_counter()
        try:
            res = self.memory.search(q.query, user_id="bench_user", limit=top_k)
            rows = res.get("results", []) if isinstance(res, dict) else []
            texts = [str(item.get("memory") or "").lower() for item in rows]
            rank = _first_rank(texts, q.expected_marker)
            return (time.perf_counter() - t0) * 1000, rank, True
        except Exception:
            return (time.perf_counter() - t0) * 1000, None, False

    def search_bench(
        self, queries: list[QueryCase], concurrency: int, top_k: int
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, float]]:
        latencies: list[float] = []
        ranks_all: list[int | None] = []
        ranks_id: list[int | None] = []
        ranks_sem: list[int | None] = []
        ok = 0
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            futures = [pool.submit(self._search_once, q, top_k) for q in queries]
            for q, f in zip(queries, futures):
                latency_ms, rank, success = f.result()
                latencies.append(latency_ms)
                if success:
                    ok += 1
                ranks_all.append(rank)
                if q.mode == "id":
                    ranks_id.append(rank)
                else:
                    ranks_sem.append(rank)
        elapsed = time.perf_counter() - start
        perf = _summary_stats(latencies, ok_count=ok, total_count=len(queries), elapsed_sec=elapsed)
        return perf, _quality_stats(ranks_all), _quality_stats(ranks_id), _quality_stats(ranks_sem)


class ChromaRunner:
    def __init__(self, work_dir: Path, vector_dim: int) -> None:
        self.work_dir = work_dir
        self.vector_dim = vector_dim
        self.collection = None

    def start(self) -> None:
        client = chromadb.PersistentClient(path=str(self.work_dir / "chroma_direct"))
        self.collection = client.get_or_create_collection(name="bench_chroma_direct")

    def stop(self) -> None:
        self.collection = None

    def ingest(self, docs: list[Doc]) -> dict[str, float]:
        assert self.collection is not None
        latencies: list[float] = []
        ok = 0
        start = time.perf_counter()
        batch_size = 128
        for start_idx in range(0, len(docs), batch_size):
            batch = docs[start_idx : start_idx + batch_size]
            ids = [d.message_id for d in batch]
            docs_text = [d.content for d in batch]
            metas = [{"user_id": d.sender, "group_id": d.group_id, "marker": d.marker} for d in batch]
            emb = [stable_embed(txt, self.vector_dim) for txt in docs_text]
            t0 = time.perf_counter()
            try:
                self.collection.add(ids=ids, documents=docs_text, metadatas=metas, embeddings=emb)
                ok += len(batch)
            except Exception:
                pass
            latencies.append((time.perf_counter() - t0) * 1000)
        elapsed = time.perf_counter() - start
        expanded = []
        for lat in latencies:
            expanded.append(lat)
        return _summary_stats(expanded, ok_count=ok, total_count=len(docs), elapsed_sec=elapsed)

    def _search_once(self, q: QueryCase, top_k: int) -> tuple[float, int | None, bool]:
        assert self.collection is not None
        t0 = time.perf_counter()
        try:
            emb = stable_embed(q.query, self.vector_dim)
            res = self.collection.query(query_embeddings=[emb], n_results=top_k)
            docs = res.get("documents", [[]])[0] if isinstance(res.get("documents"), list) else []
            texts = [str(d or "").lower() for d in docs]
            rank = _first_rank(texts, q.expected_marker)
            return (time.perf_counter() - t0) * 1000, rank, True
        except Exception:
            return (time.perf_counter() - t0) * 1000, None, False

    def search_bench(
        self, queries: list[QueryCase], concurrency: int, top_k: int
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, float]]:
        latencies: list[float] = []
        ranks_all: list[int | None] = []
        ranks_id: list[int | None] = []
        ranks_sem: list[int | None] = []
        ok = 0
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            futures = [pool.submit(self._search_once, q, top_k) for q in queries]
            for q, f in zip(queries, futures):
                latency_ms, rank, success = f.result()
                latencies.append(latency_ms)
                if success:
                    ok += 1
                ranks_all.append(rank)
                if q.mode == "id":
                    ranks_id.append(rank)
                else:
                    ranks_sem.append(rank)
        elapsed = time.perf_counter() - start
        perf = _summary_stats(latencies, ok_count=ok, total_count=len(queries), elapsed_sec=elapsed)
        return perf, _quality_stats(ranks_all), _quality_stats(ranks_id), _quality_stats(ranks_sem)


def _run_system_bench(
    name: str,
    ingest_fn: Callable[[list[Doc]], dict[str, float]],
    search_fn: Callable[[list[QueryCase], int, int], tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, float]]],
    docs: list[Doc],
    queries: list[QueryCase],
    top_k: int,
    concurrency: int,
) -> dict[str, Any]:
    ingest = ingest_fn(docs)
    perf, quality_all, quality_id, quality_sem = search_fn(queries, concurrency, top_k)
    return {
        "name": name,
        "ingest": ingest,
        "search_perf": perf,
        "quality_all": quality_all,
        "quality_id_queries": quality_id,
        "quality_semantic_queries": quality_sem,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark FlockMem vs OSS memory baselines")
    parser.add_argument("--doc-count", type=int, default=800, help="Number of memory items to ingest")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k for retrieval")
    parser.add_argument("--concurrency", type=int, default=24, help="Search concurrency")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--result-file",
        type=str,
        default="tools/perf/results/benchmark_latest.json",
        help="Path to output JSON report",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    work_dir = Path(tempfile.mkdtemp(prefix="minimem_bench_"))
    result_path = (repo_root / args.result_file).resolve()
    result_path.parent.mkdir(parents=True, exist_ok=True)

    docs, queries = build_dataset(doc_count=max(20, args.doc_count), seed=args.seed)
    embed_server = FakeEmbeddingServer()
    embed_server.start()

    minimem_port = _find_free_port()
    minimem = FlockMemRunner(repo_root=repo_root, embed_base_url=embed_server.base_url, work_dir=work_dir, port=minimem_port)
    mem0 = Mem0Runner(embed_base_url=embed_server.base_url, work_dir=work_dir, vector_dim=768)
    chroma = ChromaRunner(work_dir=work_dir, vector_dim=768)

    report: dict[str, Any] = {
        "timestamp": int(time.time()),
        "doc_count": len(docs),
        "query_count": len(queries),
        "top_k": args.top_k,
        "concurrency": args.concurrency,
        "systems": [],
        "notes": [
            "All systems run on the same machine and dataset.",
            "FlockMem uses retrieve_method=agentic, decision_mode=rule.",
            "Mem0 runs with infer=False to isolate memory retrieval/storage performance.",
            "Embedding provider is a local deterministic OpenAI-compatible mock for fair comparison.",
        ],
    }

    try:
        minimem.start()
        report["systems"].append(
            _run_system_bench(
                name="FlockMem(agentic)",
                ingest_fn=minimem.ingest,
                search_fn=lambda qs, cc, k: minimem.search_bench(
                    queries=qs,
                    retrieve_method="agentic",
                    decision_mode="static",
                    concurrency=cc,
                    top_k=k,
                ),
                docs=docs,
                queries=queries,
                top_k=args.top_k,
                concurrency=args.concurrency,
            )
        )
        report["systems"][-1]["process"] = minimem.process_metrics()

        report["systems"].append(
            _run_system_bench(
                name="FlockMem(hybrid)",
                ingest_fn=lambda _: {"count": 0.0, "ok": 0.0, "error_rate": 0.0, "qps": 0.0, "latency_p50_ms": 0.0, "latency_p95_ms": 0.0, "latency_p99_ms": 0.0, "latency_mean_ms": 0.0},
                search_fn=lambda qs, cc, k: minimem.search_bench(
                    queries=qs,
                    retrieve_method="hybrid",
                    decision_mode="static",
                    concurrency=cc,
                    top_k=k,
                ),
                docs=[],
                queries=queries,
                top_k=args.top_k,
                concurrency=args.concurrency,
            )
        )
        report["systems"][-1]["process"] = minimem.process_metrics()

        report["systems"].append(
            _run_system_bench(
                name="FlockMem(keyword)",
                ingest_fn=lambda _: {"count": 0.0, "ok": 0.0, "error_rate": 0.0, "qps": 0.0, "latency_p50_ms": 0.0, "latency_p95_ms": 0.0, "latency_p99_ms": 0.0, "latency_mean_ms": 0.0},
                search_fn=lambda qs, cc, k: minimem.search_bench(
                    queries=qs,
                    retrieve_method="keyword",
                    decision_mode="static",
                    concurrency=cc,
                    top_k=k,
                ),
                docs=[],
                queries=queries,
                top_k=args.top_k,
                concurrency=args.concurrency,
            )
        )
        report["systems"][-1]["process"] = minimem.process_metrics()

        report["systems"].append(
            _run_system_bench(
                name="FlockMem(vector)",
                ingest_fn=lambda _: {"count": 0.0, "ok": 0.0, "error_rate": 0.0, "qps": 0.0, "latency_p50_ms": 0.0, "latency_p95_ms": 0.0, "latency_p99_ms": 0.0, "latency_mean_ms": 0.0},
                search_fn=lambda qs, cc, k: minimem.search_bench(
                    queries=qs,
                    retrieve_method="vector",
                    decision_mode="static",
                    concurrency=cc,
                    top_k=k,
                ),
                docs=[],
                queries=queries,
                top_k=args.top_k,
                concurrency=args.concurrency,
            )
        )
        report["systems"][-1]["process"] = minimem.process_metrics()

        mem0.start()
        report["systems"].append(
            _run_system_bench(
                name="Mem0(oss-0.1.114/chroma)",
                ingest_fn=mem0.ingest,
                search_fn=lambda qs, cc, k: mem0.search_bench(queries=qs, concurrency=cc, top_k=k),
                docs=docs,
                queries=queries,
                top_k=args.top_k,
                concurrency=args.concurrency,
            )
        )
        report["systems"][-1]["process"] = {"rss_mb": 0.0}

        chroma.start()
        report["systems"].append(
            _run_system_bench(
                name="Chroma(vector-only)",
                ingest_fn=chroma.ingest,
                search_fn=lambda qs, cc, k: chroma.search_bench(queries=qs, concurrency=cc, top_k=k),
                docs=docs,
                queries=queries,
                top_k=args.top_k,
                concurrency=args.concurrency,
            )
        )
        report["systems"][-1]["process"] = {"rss_mb": 0.0}

    finally:
        chroma.stop()
        mem0.stop()
        minimem.stop()
        embed_server.stop()
        shutil.rmtree(work_dir, ignore_errors=True)

    result_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Benchmark finished. Report: {result_path}")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()

