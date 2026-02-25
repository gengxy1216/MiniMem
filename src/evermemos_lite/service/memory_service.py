from __future__ import annotations

import json
import math
import re
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock
from typing import Any, Protocol

from evermemos_lite.domain.policy import EffectivePolicy
from evermemos_lite.domain.retrieval.fusion import reciprocal_rank_fusion
from evermemos_lite.infra.graph.kuzu_store import KuzuGraphStore
from evermemos_lite.infra.sqlite.db import SQLiteEngine
from evermemos_lite.infra.sqlite.memory_repository import MemoryRepository
from evermemos_lite.infra.vector.lancedb_store import LanceVectorStore
from evermemos_lite.service.extractor import MemoryExtractor
from evermemos_lite.service.formation_enhancer import (
    BoundaryDecision,
    FormationEnhancerProtocol,
)
from evermemos_lite.service.graph_extractor import extract_graph_triples
from evermemos_lite.service.query_rewriter import (
    QueryExpansionDecision,
    QueryRewriterProtocol,
    RewriteDecision,
)
from evermemos_lite.service.retrieval_verifier import (
    RetrievalVerifierProtocol,
    SufficiencyDecision,
)
from evermemos_lite.service.semantic_consolidator import ConsolidateInput, SemanticConsolidator


class EmbeddingProviderProtocol(Protocol):
    def embed(self, text: str) -> list[float]:
        ...


@dataclass(frozen=True)
class MemorizeInput:
    message_id: str
    create_time: int
    sender: str
    content: str
    group_id: str | None
    group_name: str | None
    sender_name: str | None
    role: str


@dataclass(frozen=True)
class ChatTurnInput:
    conversation_id: str
    user_id: str
    group_id: str | None
    user_text: str
    assistant_text: str
    timestamp: int


@dataclass(frozen=True)
class RetrievalSufficiency:
    sufficient: bool
    source: str
    confidence: float
    reason: str


@dataclass(frozen=True)
class AgenticRewritePlan:
    query: str
    source: str
    confidence: float
    reason: str


class MemoryService:
    def __init__(
        self,
        engine: SQLiteEngine,
        vector_store: LanceVectorStore,
        embedding_provider: EmbeddingProviderProtocol,
        extractor: MemoryExtractor,
        graph_store: KuzuGraphStore,
        formation_enhancer: FormationEnhancerProtocol | None = None,
        semantic_boundary_min_confidence: float = 0.68,
        retrieval_verifier: RetrievalVerifierProtocol | None = None,
        retrieval_verifier_min_confidence: float = 0.66,
        query_rewriter: QueryRewriterProtocol | None = None,
        query_rewriter_min_confidence: float = 0.62,
        phase4_reasoning_enabled: bool = True,
        temporal_rerank_weight: float = 0.35,
        multi_hop_max_queries: int = 3,
        graph_top_k: int = 3,
        graph_write_min_importance: float = 0.6,
        key_memory_importance_threshold: float = 0.5,
        vector_write_min_importance: float = 0.3,
        vector_embed_chunk_chars: int = 600,
        vector_embed_max_chunks: int = 8,
        search_budget_factor: int = 4,
        search_min_probe_k: int = 12,
        keyword_confident_best_score: float = 9.0,
        keyword_confident_kth_score: float = 2.8,
        semantic_vector_budget_cap: int = 32,
        semantic_keyword_budget_cap: int = 16,
        query_embed_cache_size: int = 256,
        query_embed_cache_ttl_sec: int = 900,
        search_trace_enabled: bool = False,
        search_trace_slow_ms: int = 0,
    ) -> None:
        self.engine = engine
        self.repo = MemoryRepository(engine)
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.extractor = extractor
        self.formation_enhancer = formation_enhancer
        self.semantic_boundary_min_confidence = max(
            0.0, min(1.0, float(semantic_boundary_min_confidence))
        )
        self.retrieval_verifier = retrieval_verifier
        self.retrieval_verifier_min_confidence = max(
            0.0, min(1.0, float(retrieval_verifier_min_confidence))
        )
        self.query_rewriter = query_rewriter
        self.query_rewriter_min_confidence = max(
            0.0, min(1.0, float(query_rewriter_min_confidence))
        )
        self.phase4_reasoning_enabled = bool(phase4_reasoning_enabled)
        self.temporal_rerank_weight = max(0.0, min(1.0, float(temporal_rerank_weight)))
        self.multi_hop_max_queries = max(1, int(multi_hop_max_queries))
        self.semantic_consolidator = SemanticConsolidator(
            self.repo, embedding_provider=embedding_provider
        )
        self.graph_store = graph_store
        self.graph_top_k = max(1, graph_top_k)
        self.graph_write_min_importance = max(0.0, min(1.0, graph_write_min_importance))
        self.key_memory_importance_threshold = max(
            0.0, min(1.0, float(key_memory_importance_threshold))
        )
        self.vector_write_min_importance = max(
            0.0, min(1.0, float(vector_write_min_importance))
        )
        self.vector_embed_chunk_chars = max(120, int(vector_embed_chunk_chars))
        self.vector_embed_max_chunks = max(1, int(vector_embed_max_chunks))
        self.search_budget_factor = max(2, int(search_budget_factor))
        self.search_min_probe_k = max(4, int(search_min_probe_k))
        self.keyword_confident_best_score = max(
            0.1, float(keyword_confident_best_score)
        )
        self.keyword_confident_kth_score = max(
            0.0, float(keyword_confident_kth_score)
        )
        self.semantic_vector_budget_cap = max(8, int(semantic_vector_budget_cap))
        self.semantic_keyword_budget_cap = max(4, int(semantic_keyword_budget_cap))
        self.query_embed_cache_size = max(32, int(query_embed_cache_size))
        self.query_embed_cache_ttl_sec = max(30, int(query_embed_cache_ttl_sec))
        self.search_trace_enabled = bool(search_trace_enabled)
        self.search_trace_slow_ms = max(0, int(search_trace_slow_ms))
        self._query_embed_cache: OrderedDict[str, tuple[int, list[float]]] = OrderedDict()
        self._query_embed_lock = Lock()
        self._vector_index_stats: dict[str, int] = {}
        self._vector_index_lock = Lock()

    @staticmethod
    def estimate_importance(content: str) -> float:
        score = min(1.0, len(content.strip()) / 400.0 + 0.2)
        return round(float(score), 3)

    def memorize(self, payload: MemorizeInput, request_id: str) -> dict[str, Any]:
        extracted = self.extractor.extract(
            content=payload.content,
            sender=payload.sender,
            group_id=payload.group_id,
        )
        extracted_importance = float(extracted.importance_score)
        if extracted_importance <= 0.01:
            extracted_importance = self.estimate_importance(extracted.episode)
        triples = extract_graph_triples(
            facts=list(extracted.atomic_facts) or [extracted.episode],
            user_id=payload.sender,
        )
        storage_tier = self._decide_storage_tier(
            importance=float(extracted_importance),
            has_entity_relation=bool(triples),
        )
        episode = self.repo.save_message_as_memory(
            message_id=payload.message_id,
            create_time=int(payload.create_time),
            sender=payload.sender,
            content=extracted.episode,
            user_id=payload.sender,
            group_id=payload.group_id,
            group_name=payload.group_name,
            sender_name=payload.sender_name,
            role=payload.role,
            importance_score=float(extracted_importance),
            storage_tier=storage_tier,
            summary=extracted.summary,
            subject=extracted.subject,
            atomic_facts=extracted.atomic_facts,
            foresights=extracted.foresights,
            profile_patch=extracted.profile_patch,
            event_id=request_id,
        )

        consolidate_result = self.semantic_consolidator.consolidate(
            ConsolidateInput(
                user_id=payload.sender,
                group_id=payload.group_id,
                event_id=episode["event_id"],
                memory_id=episode["id"],
                timestamp=int(payload.create_time),
                episode=extracted.episode,
                summary=extracted.summary,
                importance_score=float(extracted_importance),
                atomic_facts=list(extracted.atomic_facts),
                profile_patch=extracted.profile_patch,
                storage_tier=storage_tier,
            )
        )
        if consolidate_result.get("scene_id"):
            episode["scene_id"] = consolidate_result.get("scene_id")

        if self.graph_store.enabled and self._tier_supports_graph(storage_tier):
            if triples:
                self.graph_store.upsert_triples(
                    triples,
                    event_id=episode["event_id"],
                    timestamp=int(payload.create_time),
                    user_id=payload.sender,
                    group_id=payload.group_id,
                )

        return {
            "memory": episode,
            "summary": extracted.summary,
            "subject": extracted.subject,
            "importance_score": extracted_importance,
            "storage_tier": storage_tier,
            "scene_id": episode.get("scene_id"),
        }

    def maybe_index_vector(self, policy: EffectivePolicy, episode: dict[str, Any]) -> dict[str, Any]:
        if not policy.vector_enabled:
            self._record_vector_index("skip.policy_disabled")
            return {"status": "skipped", "reason": "policy_disabled"}
        tier = str(episode.get("storage_tier") or "text_only")
        if not self._tier_supports_vector(tier):
            self._record_vector_index("skip.tier_not_vector")
            return {"status": "skipped", "reason": "tier_not_vector", "tier": tier}
        importance = float(episode.get("importance_score", 0.0))
        min_vector_threshold = max(
            float(policy.importance_threshold), self.vector_write_min_importance
        )
        if importance < min_vector_threshold:
            self._record_vector_index("skip.importance_below_threshold")
            return {
                "status": "skipped",
                "reason": "importance_below_threshold",
                "importance": round(float(importance), 4),
                "threshold": round(float(min_vector_threshold), 4),
            }
        try:
            vector = self._embed_episode_for_index(str(episode.get("episode", "")))
        except Exception as exc:
            self._record_vector_index("error.embed_failed")
            return {
                "status": "error",
                "reason": "embed_failed",
                "detail": str(exc)[:260],
            }
        vector = self._fit_vector_dim(vector)
        vector_id = uuid.uuid4().hex
        now = int(time.time())
        try:
            self.vector_store.upsert(
                vector_id,
                episode["id"],
                vector,
                {
                    "id": episode["id"],
                    "user_id": episode.get("user_id"),
                    "group_id": episode.get("group_id"),
                    "timestamp": int(episode.get("timestamp", now)),
                    "importance_score": float(episode.get("importance_score", 0.0)),
                },
            )
        except Exception as exc:
            self._record_vector_index("error.vector_store_upsert_failed")
            return {
                "status": "error",
                "reason": "vector_store_upsert_failed",
                "detail": str(exc)[:260],
            }
        self._record_vector_index("indexed.ok")
        return {"status": "indexed", "reason": "ok"}

    def get_vector_index_stats(self) -> dict[str, int]:
        with self._vector_index_lock:
            return dict(self._vector_index_stats)

    def _record_vector_index(self, key: str) -> None:
        with self._vector_index_lock:
            self._vector_index_stats[key] = int(self._vector_index_stats.get(key, 0)) + 1

    def _fit_vector_dim(self, vector: list[float]) -> list[float]:
        if len(vector) == self.vector_store.vector_dim:
            return vector
        if len(vector) > self.vector_store.vector_dim:
            return vector[: self.vector_store.vector_dim]
        return vector + [0.0] * (self.vector_store.vector_dim - len(vector))

    def _embed_episode_for_index(self, episode: str) -> list[float]:
        chunks = self._split_embedding_chunks(episode)
        vectors: list[list[float]] = []
        errors: list[str] = []
        for chunk in chunks:
            text = str(chunk or "").strip()
            if not text:
                continue
            try:
                vec = self.embedding_provider.embed(text)
                safe = [float(x) for x in vec if isinstance(x, (int, float))]
                if safe:
                    vectors.append(safe)
            except Exception as exc:
                errors.append(str(exc))
                continue
        if not vectors:
            detail = errors[-1] if errors else "empty embedding output"
            raise RuntimeError(detail)
        min_dim = min(len(v) for v in vectors)
        if min_dim <= 0:
            raise RuntimeError("invalid embedding dimensions")
        avg = [0.0] * min_dim
        for vec in vectors:
            for i in range(min_dim):
                avg[i] += float(vec[i])
        denom = float(len(vectors))
        return [x / denom for x in avg]

    def _split_embedding_chunks(self, text: str) -> list[str]:
        raw = str(text or "").strip()
        if not raw:
            return []
        max_chars = self.vector_embed_chunk_chars
        max_chunks = self.vector_embed_max_chunks
        if len(raw) <= max_chars:
            return [raw]
        parts = [p.strip() for p in re.split(r"[。！？!?；;\n]+", raw) if p.strip()]
        chunks: list[str] = []
        buf = ""
        for part in parts:
            candidate = part if not buf else f"{buf} {part}"
            if len(candidate) <= max_chars:
                buf = candidate
                continue
            if buf:
                chunks.append(buf)
                buf = ""
            if len(part) <= max_chars:
                buf = part
                continue
            for start in range(0, len(part), max_chars):
                chunks.append(part[start : start + max_chars])
                if len(chunks) >= max_chunks:
                    return chunks[:max_chunks]
        if buf:
            chunks.append(buf)
        return chunks[:max_chunks]

    def fetch(
        self,
        user_id: str | None,
        group_id: str | None,
        limit: int,
        as_of_ts: int | None = None,
        start_ts: int | None = None,
        end_ts: int | None = None,
    ) -> list[dict[str, Any]]:
        rows = self.repo.fetch_episodes(user_id=user_id, group_id=group_id, limit=max(1, limit))
        return self._attach_valid_foresight(
            rows=rows,
            user_id=user_id,
            group_id=group_id,
            as_of_ts=as_of_ts,
            start_ts=start_ts,
            end_ts=end_ts,
        )

    def get_profile_snapshot(self, user_id: str | None, group_id: str | None) -> dict[str, Any] | None:
        return self.repo.get_latest_profile_snapshot(user_id=user_id, group_id=group_id)

    def search_graph(
        self, query: str, user_id: str | None, group_id: str | None, limit: int
    ) -> list[dict[str, Any]]:
        if not self.graph_store.enabled:
            return []
        rows = self.graph_store.search(
            query=query, user_id=user_id, group_id=group_id, limit=max(1, limit)
        )
        return [
            {
                "subject": row["subject"],
                "relation": row["relation"],
                "obj": row["obj"],
                "event_id": row["event_id"],
                "timestamp": row["timestamp"],
                "confidence": row["confidence"],
            }
            for row in rows
        ]

    def graph_neighbors(
        self, entity_name: str, user_id: str | None, group_id: str | None, limit: int
    ) -> list[dict[str, Any]]:
        if not self.graph_store.enabled:
            return []
        rows = self.graph_store.neighbors(
            entity_name=entity_name,
            user_id=user_id,
            group_id=group_id,
            limit=max(1, limit),
        )
        return [
            {
                "subject": row["subject"],
                "relation": row["relation"],
                "obj": row["obj"],
                "event_id": row["event_id"],
                "timestamp": row["timestamp"],
                "confidence": row["confidence"],
            }
            for row in rows
        ]

    def search(
        self,
        policy: EffectivePolicy,
        query: str,
        user_id: str | None,
        group_id: str | None,
        top_k: int,
        as_of_ts: int | None = None,
        start_ts: int | None = None,
        end_ts: int | None = None,
    ) -> list[dict[str, Any]]:
        start = time.perf_counter()
        if policy.agentic_enabled:
            base = self._agentic_search(
                policy,
                query,
                user_id,
                group_id,
                top_k,
                as_of_ts=as_of_ts,
                start_ts=start_ts,
                end_ts=end_ts,
            )
        else:
            if as_of_ts is None and start_ts is None and end_ts is None:
                base = self._basic_search(policy, query, user_id, group_id, top_k, None)
            else:
                base = self._basic_search(
                    policy,
                    query,
                    user_id,
                    group_id,
                    top_k,
                    None,
                    as_of_ts=as_of_ts,
                    start_ts=start_ts,
                    end_ts=end_ts,
                )
        after_base = time.perf_counter()
        profile_hints = self._profile_hint_hits(query=query, user_id=user_id, group_id=group_id)
        after_profile = time.perf_counter()
        merged_base = self._merge_profile_hits(base, profile_hints, top_k)
        merged = self._merge_graph_hits(query, user_id, group_id, merged_base, top_k)
        end = time.perf_counter()
        self._emit_search_trace(
            "search_total",
            query_len=len(str(query or "")),
            top_k=max(1, int(top_k)),
            profile=str(policy.profile),
            agentic=bool(policy.agentic_enabled),
            output_count=len(merged),
            ms_total=round((end - start) * 1000.0, 3),
            ms_base=round((after_base - start) * 1000.0, 3),
            ms_profile=round((after_profile - after_base) * 1000.0, 3),
            ms_graph=round((end - after_profile) * 1000.0, 3),
        )
        return merged

    def retrieve_live_segment_context(
        self, *, conversation_id: str, query: str, limit: int = 2
    ) -> list[dict[str, Any]]:
        segment = self.repo.get_conversation_segment(conversation_id)
        if not segment:
            return []
        candidates = self._recent_user_queries_from_segment(segment, limit=6)
        if not candidates:
            return []
        is_identity = self._is_identity_query(query)
        rows: list[dict[str, Any]] = []
        for idx, text in enumerate(reversed(candidates), start=1):
            snippet = str(text or "").strip()
            if not snippet:
                continue
            overlap = self._lexical_overlap_ratio(query, snippet)
            if is_identity and re.search(r"我叫([^\s，。；;！？!?]{1,32})", snippet):
                overlap = max(overlap, 0.96)
            if overlap < 0.08 and not is_identity:
                continue
            rows.append(
                {
                    "id": f"live:{conversation_id}:{idx}",
                    "event_id": f"live:{conversation_id}",
                    "timestamp": int(segment.get("last_time") or time.time()),
                    "summary": snippet[:220],
                    "subject": segment.get("user_id") or "user",
                    "episode": snippet,
                    "importance_score": 0.82 if is_identity else 0.62,
                    "storage_tier": "text_only",
                    "score": overlap,
                    "source": "live_segment",
                }
            )
        rows.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        return rows[: max(1, int(limit))]

    def _basic_search(
        self,
        policy: EffectivePolicy,
        query: str,
        user_id: str | None,
        group_id: str | None,
        top_k: int,
        candidate_episode_ids: set[str] | None,
        attach_foresight: bool = True,
        as_of_ts: int | None = None,
        start_ts: int | None = None,
        end_ts: int | None = None,
    ) -> list[dict[str, Any]]:
        started_at = time.perf_counter()
        kw_ms = 0.0
        vec_ms = 0.0
        hydrate_ms = 0.0
        foresight_ms = 0.0
        keyword_hits: list[dict[str, Any]] = []
        vector_hits: list[dict[str, Any]] = []
        top = max(1, int(top_k))
        budget_multiplier = max(2, int(self.search_budget_factor))
        min_probe = max(4, int(self.search_min_probe_k))
        id_like_query = self._is_id_like_query(query)
        keyword_budget = max(
            min_probe,
            min(200, max(top * budget_multiplier, int(policy.keyword_top_k))),
        )
        vector_budget = max(
            min_probe,
            min(200, max(top * budget_multiplier, int(policy.vector_top_k))),
        )
        if not id_like_query:
            keyword_budget = min(keyword_budget, self.semantic_keyword_budget_cap)
            vector_budget = min(vector_budget, self.semantic_vector_budget_cap)

        if policy.keyword_enabled:
            kw_started = time.perf_counter()
            keyword_hits = self.repo.search_keyword(
                query=query,
                user_id=user_id,
                group_id=group_id,
                top_k=keyword_budget,
                candidate_episode_ids=candidate_episode_ids,
            )
            kw_ms = (time.perf_counter() - kw_started) * 1000.0
            for row in keyword_hits:
                row["source"] = "keyword"
                row["in_keyword_hits"] = True

        run_vector_probe = (
            policy.vector_enabled
            and self.vector_store.enabled
            and not (
                id_like_query
                and self._is_keyword_confident(keyword_hits, top)
            )
        )
        if run_vector_probe:
            try:
                vec_started = time.perf_counter()
                vector = self._embed_query(query)
                vector_hits = self.vector_store.search(
                    vector=vector,
                    top_k=vector_budget,
                    user_id=user_id,
                    group_id=group_id,
                    candidate_episode_ids=candidate_episode_ids,
                )
                vec_ms = (time.perf_counter() - vec_started) * 1000.0
                for row in vector_hits:
                    row["source"] = "vector"
                    row["in_vector_hits"] = True
            except Exception:
                vector_hits = []

        merge_strategy = "none"
        if keyword_hits and vector_hits:
            if self._should_prefer_vector_for_semantic(
                query=query,
                id_like_query=id_like_query,
                keyword_hits=keyword_hits,
                vector_hits=vector_hits,
                top_k=top,
            ):
                hit_rows = self._vector_priority_merge(
                    vector_hits=vector_hits,
                    keyword_hits=keyword_hits,
                    top_k=top,
                )
                merge_strategy = "semantic_vector_priority"
            else:
                fused = reciprocal_rank_fusion(
                    [keyword_hits, vector_hits], key="memory_id", rrf_k=policy.rrf_k
                )
                for row in fused:
                    row["source"] = "hybrid_rrf"
                    row.setdefault(
                        "in_keyword_hits",
                        any(x.get("memory_id") == row.get("memory_id") for x in keyword_hits),
                    )
                    row.setdefault(
                        "in_vector_hits",
                        any(x.get("memory_id") == row.get("memory_id") for x in vector_hits),
                    )
                hit_rows = fused[:top]
                merge_strategy = "hybrid_rrf"
        elif keyword_hits:
            hit_rows = keyword_hits[:top]
            merge_strategy = "keyword_only"
        elif vector_hits:
            hit_rows = vector_hits[:top]
            merge_strategy = "vector_only"
        else:
            hit_rows = []

        hydrate_started = time.perf_counter()
        episode_ids = [str(r.get("memory_id")) for r in hit_rows if r.get("memory_id")]
        hydrated = self._hydrate_vector_results(episode_ids)
        hydrate_ms = (time.perf_counter() - hydrate_started) * 1000.0
        hydrated_map = {str(row["id"]): row for row in hydrated}

        rows: list[dict[str, Any]] = []
        for hit in hit_rows:
            mid = str(hit.get("memory_id", ""))
            base = dict(hydrated_map.get(mid, {}))
            if not base:
                continue
            base["score"] = self._adjust_hybrid_score(base, float(hit.get("score", 0.0)))
            base["source"] = hit.get("source", "keyword")
            base["in_vector_hits"] = bool(hit.get("in_vector_hits", False))
            base["in_keyword_hits"] = bool(hit.get("in_keyword_hits", False))
            rows.append(base)

        if not rows:
            rows = self._fallback_recent(
                user_id,
                group_id,
                top_k,
                candidate_episode_ids,
                attach_foresight=attach_foresight,
                as_of_ts=as_of_ts,
                start_ts=start_ts,
                end_ts=end_ts,
            )
        limited = rows[:top]
        if not attach_foresight:
            self._emit_search_trace(
                "basic_search",
                query_len=len(str(query or "")),
                top_k=top,
                id_like=bool(id_like_query),
                candidate_count=0 if candidate_episode_ids is None else len(candidate_episode_ids),
                keyword_budget=keyword_budget,
                vector_budget=vector_budget,
                keyword_hits=len(keyword_hits),
                vector_hits=len(vector_hits),
                merge_strategy=merge_strategy,
                output_count=len(limited),
                foresight_attached=False,
                ms_total=round((time.perf_counter() - started_at) * 1000.0, 3),
                ms_keyword=round(kw_ms, 3),
                ms_vector=round(vec_ms, 3),
                ms_hydrate=round(hydrate_ms, 3),
                ms_foresight=0.0,
            )
            return limited
        foresight_started = time.perf_counter()
        final_rows = self._attach_valid_foresight(
            rows=limited,
            user_id=user_id,
            group_id=group_id,
            as_of_ts=as_of_ts,
            start_ts=start_ts,
            end_ts=end_ts,
        )
        foresight_ms = (time.perf_counter() - foresight_started) * 1000.0
        self._emit_search_trace(
            "basic_search",
            query_len=len(str(query or "")),
            top_k=top,
            id_like=bool(id_like_query),
            candidate_count=0 if candidate_episode_ids is None else len(candidate_episode_ids),
            keyword_budget=keyword_budget,
            vector_budget=vector_budget,
            keyword_hits=len(keyword_hits),
            vector_hits=len(vector_hits),
            merge_strategy=merge_strategy,
            output_count=len(final_rows),
            foresight_attached=True,
            ms_total=round((time.perf_counter() - started_at) * 1000.0, 3),
            ms_keyword=round(kw_ms, 3),
            ms_vector=round(vec_ms, 3),
            ms_hydrate=round(hydrate_ms, 3),
            ms_foresight=round(foresight_ms, 3),
        )
        return final_rows

    def append_chat_turn(
        self, *, payload: ChatTurnInput, policy: EffectivePolicy
    ) -> dict[str, Any]:
        now_ts = int(payload.timestamp)
        segment = self.repo.get_conversation_segment(payload.conversation_id)
        turn_markdown = self._format_turn_markdown(
            user_text=payload.user_text,
            assistant_text=payload.assistant_text,
            timestamp=now_ts,
        )
        next_segment_seq = int(segment.get("segment_seq") or 1) if segment else 1
        boundary = False
        boundary_reason = ""
        boundary_confidence = 0.0
        auto_memory_saved = False
        auto_memory_error: str | None = None

        if segment:
            should_cut, boundary_reason, boundary_confidence = self._decide_segment_boundary(
                segment=segment,
                query=payload.user_text,
                now_ts=now_ts,
            )
        else:
            should_cut = False
        if segment and should_cut:
            boundary = True
            try:
                committed = self._commit_segment_as_memory(
                    conversation_id=payload.conversation_id,
                    segment=segment,
                    user_id=payload.user_id,
                    group_id=payload.group_id,
                    policy=policy,
                )
                auto_memory_saved = bool(committed)
            except Exception as exc:
                auto_memory_error = str(exc)
            next_segment_seq = int(segment.get("segment_seq") or 1) + 1
            self.repo.delete_conversation_segment(payload.conversation_id)
            segment = None

        if segment is None:
            segment_seq = next_segment_seq
            turns_markdown = turn_markdown
            turn_count = 1
            start_time = now_ts
        else:
            segment_seq = int(segment.get("segment_seq") or 1)
            turns_markdown = str(segment.get("turns_markdown") or "").strip()
            turns_markdown = (
                f"{turns_markdown}\n\n{turn_markdown}".strip()
                if turns_markdown
                else turn_markdown
            )
            turn_count = int(segment.get("turn_count") or 0) + 1
            start_time = int(segment.get("start_time") or now_ts)

        self.repo.upsert_conversation_segment(
            conversation_id=payload.conversation_id,
            user_id=payload.user_id,
            group_id=payload.group_id,
            segment_seq=segment_seq,
            turns_markdown=turns_markdown,
            last_query=payload.user_text,
            turn_count=turn_count,
            start_time=start_time,
            last_time=now_ts,
        )

        if turn_count >= 4:
            live = self.repo.get_conversation_segment(payload.conversation_id)
            if live:
                try:
                    committed = self._commit_segment_as_memory(
                        conversation_id=payload.conversation_id,
                        segment=live,
                        user_id=payload.user_id,
                        group_id=payload.group_id,
                        policy=policy,
                    )
                    auto_memory_saved = auto_memory_saved or bool(committed)
                    self.repo.delete_conversation_segment(payload.conversation_id)
                except Exception as exc:
                    auto_memory_error = str(exc)

        return {
            "memory_saved": auto_memory_saved,
            "memory_error": auto_memory_error,
            "boundary_detected": boundary,
            "boundary_reason": boundary_reason,
            "boundary_confidence": round(float(boundary_confidence), 4),
            "segment_turn_count": turn_count,
        }

    def _agentic_search(
        self,
        policy: EffectivePolicy,
        query: str,
        user_id: str | None,
        group_id: str | None,
        top_k: int,
        as_of_ts: int | None = None,
        start_ts: int | None = None,
        end_ts: int | None = None,
    ) -> list[dict[str, Any]]:
        started_at = time.perf_counter()
        # Round-1: full-scope retrieval with low overhead.
        if as_of_ts is None and start_ts is None and end_ts is None:
            hits = self._basic_search(
                policy,
                query,
                user_id,
                group_id,
                top_k,
                None,
                attach_foresight=False,
            )
        else:
            hits = self._basic_search(
                policy,
                query,
                user_id,
                group_id,
                top_k,
                None,
                attach_foresight=False,
                as_of_ts=as_of_ts,
                start_ts=start_ts,
                end_ts=end_ts,
            )
        sufficiency = self._assess_retrieval_sufficiency(query=query, hits=hits, top_k=top_k)
        if self._should_skip_agentic_second_round(
            query=query,
            hits=hits,
            top_k=top_k,
            sufficiency=sufficiency,
        ):
            final_rows = self._attach_valid_foresight(
                rows=hits[: max(1, top_k)],
                user_id=user_id,
                group_id=group_id,
                as_of_ts=as_of_ts,
                start_ts=start_ts,
                end_ts=end_ts,
            )
            self._emit_search_trace(
                "agentic_search",
                query_len=len(str(query or "")),
                top_k=max(1, int(top_k)),
                second_round=False,
                reason="first_round_sufficient",
                sufficiency_source=sufficiency.source,
                sufficiency_confidence=round(float(sufficiency.confidence), 3),
                first_round_count=len(hits),
                output_count=len(final_rows),
                ms_total=round((time.perf_counter() - started_at) * 1000.0, 3),
            )
            return final_rows
        # Round-2: scene-guided candidate narrowing + rewrite only when needed.
        scene_started = time.perf_counter()
        scene = self._scene_guided_candidate_ids(query, user_id, group_id, top_scene_n=6)
        scene_ms = (time.perf_counter() - scene_started) * 1000.0
        candidate_ids = set(scene.get("episode_ids", []))
        rewrite_plan = self._build_agentic_rewrite_plan(
            query=query, hits=hits, sufficiency=sufficiency
        )
        rewritten = rewrite_plan.query
        if not candidate_ids and (
            rewritten.strip() == query.strip() or bool(sufficiency.sufficient)
        ):
            final_rows = self._attach_valid_foresight(
                rows=hits[: max(1, top_k)],
                user_id=user_id,
                group_id=group_id,
                as_of_ts=as_of_ts,
                start_ts=start_ts,
                end_ts=end_ts,
            )
            self._emit_search_trace(
                "agentic_search",
                query_len=len(str(query or "")),
                top_k=max(1, int(top_k)),
                second_round=False,
                reason="no_scene_and_noop_rewrite",
                sufficiency_source=sufficiency.source,
                sufficiency_confidence=round(float(sufficiency.confidence), 3),
                rewrite_source=rewrite_plan.source,
                rewrite_confidence=round(float(rewrite_plan.confidence), 3),
                first_round_count=len(hits),
                candidate_count=0,
                output_count=len(final_rows),
                ms_scene=round(scene_ms, 3),
                ms_total=round((time.perf_counter() - started_at) * 1000.0, 3),
            )
            return final_rows
        second = self._run_agentic_second_round(
            policy=policy,
            original_query=query,
            rewritten_query=rewritten,
            first_round_hits=hits,
            insufficiency_reason=sufficiency.reason,
            user_id=user_id,
            group_id=group_id,
            top_k=top_k,
            candidate_episode_ids=candidate_ids or None,
            as_of_ts=as_of_ts,
            start_ts=start_ts,
            end_ts=end_ts,
        )
        self._annotate_scene_info(
            rows=second,
            episode_scene_map=scene.get("episode_scene_map", {}),
            scene_score_map=scene.get("scene_score_map", {}),
        )
        merged = reciprocal_rank_fusion([hits, second], key="id", rrf_k=policy.rrf_k)
        for row in merged:
            row.setdefault("source", "agentic_two_round")
            row["agentic_round"] = 2
            row["agentic_rewrite"] = rewritten
            row["agentic_rewrite_source"] = rewrite_plan.source
            row["agentic_rewrite_confidence"] = round(float(rewrite_plan.confidence), 3)
            row["agentic_scene_count"] = len(candidate_ids)
        if not merged:
            final_rows = self._fallback_recent(
                user_id,
                group_id,
                top_k,
                candidate_ids,
                as_of_ts=as_of_ts,
                start_ts=start_ts,
                end_ts=end_ts,
            )
            self._emit_search_trace(
                "agentic_search",
                query_len=len(str(query or "")),
                top_k=max(1, int(top_k)),
                second_round=True,
                reason="fallback_recent",
                first_round_count=len(hits),
                second_round_count=len(second),
                candidate_count=len(candidate_ids),
                output_count=len(final_rows),
                ms_scene=round(scene_ms, 3),
                ms_total=round((time.perf_counter() - started_at) * 1000.0, 3),
            )
            return final_rows
        final_rows = self._attach_valid_foresight(
            rows=merged[: max(1, top_k)],
            user_id=user_id,
            group_id=group_id,
            as_of_ts=as_of_ts,
            start_ts=start_ts,
            end_ts=end_ts,
        )
        if self.phase4_reasoning_enabled and self._is_temporal_query(query):
            final_rows = self._rerank_temporal_rows(query=query, rows=final_rows)
        self._emit_search_trace(
            "agentic_search",
            query_len=len(str(query or "")),
            top_k=max(1, int(top_k)),
            second_round=True,
            reason="rrf_merge",
            first_round_count=len(hits),
            second_round_count=len(second),
            candidate_count=len(candidate_ids),
            output_count=len(final_rows),
            rewrite_source=rewrite_plan.source,
            rewrite_confidence=round(float(rewrite_plan.confidence), 3),
            ms_scene=round(scene_ms, 3),
            ms_total=round((time.perf_counter() - started_at) * 1000.0, 3),
        )
        return final_rows

    def _is_keyword_confident(self, keyword_hits: list[dict[str, Any]], top_k: int) -> bool:
        top = max(1, int(top_k))
        if len(keyword_hits) < top:
            return False
        scores = [float(x.get("score", 0.0)) for x in keyword_hits[:top]]
        if not scores:
            return False
        best = max(scores)
        kth = scores[-1]
        return (
            best >= self.keyword_confident_best_score
            and kth >= self.keyword_confident_kth_score
        )

    def _should_prefer_vector_for_semantic(
        self,
        *,
        query: str,
        id_like_query: bool,
        keyword_hits: list[dict[str, Any]],
        vector_hits: list[dict[str, Any]],
        top_k: int,
    ) -> bool:
        if id_like_query or self._is_identity_query(query):
            return False
        if len(str(query or "").strip()) < 16:
            return False
        if not vector_hits:
            return False
        if self._is_keyword_confident(keyword_hits, min(max(1, int(top_k)), 2)):
            return False
        return True

    @staticmethod
    def _vector_priority_merge(
        *,
        vector_hits: list[dict[str, Any]],
        keyword_hits: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        top = max(1, int(top_k))
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        keyword_set = {
            str(row.get("memory_id")) for row in keyword_hits if row.get("memory_id")
        }
        vector_set = {
            str(row.get("memory_id")) for row in vector_hits if row.get("memory_id")
        }

        for row in vector_hits:
            mid = str(row.get("memory_id", "")).strip()
            if not mid or mid in seen:
                continue
            seen.add(mid)
            item = dict(row)
            item["source"] = "semantic_vector_priority"
            item["in_vector_hits"] = True
            item["in_keyword_hits"] = mid in keyword_set
            out.append(item)
            if len(out) >= top:
                return out[:top]

        for row in keyword_hits:
            mid = str(row.get("memory_id", "")).strip()
            if not mid or mid in seen:
                continue
            seen.add(mid)
            item = dict(row)
            item["source"] = "semantic_vector_priority"
            item["in_vector_hits"] = mid in vector_set
            item["in_keyword_hits"] = True
            out.append(item)
            if len(out) >= top:
                break
        return out[:top]

    def _should_skip_agentic_second_round(
        self,
        *,
        query: str,
        hits: list[dict[str, Any]],
        top_k: int,
        sufficiency: RetrievalSufficiency | None = None,
    ) -> bool:
        if sufficiency and sufficiency.source == "llm_verifier":
            return bool(sufficiency.sufficient)
        if self._is_id_like_query(query):
            if sufficiency is not None:
                return bool(sufficiency.sufficient)
            return self._is_sufficient(hits, top_k)
        if not hits:
            return False
        if sufficiency is not None and bool(sufficiency.sufficient):
            return True
        best = max(float(h.get("score", 0.0)) for h in hits)
        if best >= 0.12:
            return True
        if len(hits) >= max(2, min(3, top_k)):
            return True
        return False

    def _assess_retrieval_sufficiency(
        self, *, query: str, hits: list[dict[str, Any]], top_k: int
    ) -> RetrievalSufficiency:
        verifier = self.retrieval_verifier
        if verifier is not None:
            try:
                decision = verifier.judge_sufficiency(query=query, hits=hits, top_k=top_k)
            except Exception:
                decision = None
            if (
                isinstance(decision, SufficiencyDecision)
                and float(decision.confidence) >= self.retrieval_verifier_min_confidence
            ):
                return RetrievalSufficiency(
                    sufficient=bool(decision.sufficient),
                    source="llm_verifier",
                    confidence=float(decision.confidence),
                    reason=str(decision.reason or "llm_verifier")[:80],
                )
        fallback = self._is_sufficient(hits, top_k)
        return RetrievalSufficiency(
            sufficient=bool(fallback),
            source="heuristic",
            confidence=1.0 if fallback else 0.0,
            reason="heuristic",
        )

    @staticmethod
    def _is_id_like_query(query: str) -> bool:
        q = str(query or "").strip().lower()
        if not q:
            return False
        if re.search(r"\b(?:tck|ticket|case|order|id|mk)[-_]\d{3,}\b", q):
            return True
        if any(term in q for term in ("ticket", "order", "case", "工单", "编号", "单号")):
            if re.search(r"\d{3,}", q):
                return True
        return False

    @staticmethod
    def _is_sufficient(hits: list[dict[str, Any]], top_k: int) -> bool:
        if len(hits) < min(3, top_k):
            return False
        best = max(float(h.get("score", 0.0)) for h in hits) if hits else 0.0
        if best < 0.05:
            return False
        if all(str(h.get("source", "")) == "fallback_recent" for h in hits):
            return False
        return True

    def _embed_query(self, query: str) -> list[float]:
        key = str(query or "").strip()
        if not key:
            return self.embedding_provider.embed(key)
        now = int(time.time())
        with self._query_embed_lock:
            cached = self._query_embed_cache.get(key)
            if cached and now - int(cached[0]) <= self.query_embed_cache_ttl_sec:
                self._query_embed_cache.move_to_end(key)
                return list(cached[1])
            if cached:
                self._query_embed_cache.pop(key, None)
        vec = self.embedding_provider.embed(key)
        safe_vec = [float(x) for x in vec if isinstance(x, (int, float))]
        with self._query_embed_lock:
            self._query_embed_cache[key] = (now, safe_vec)
            self._query_embed_cache.move_to_end(key)
            while len(self._query_embed_cache) > self.query_embed_cache_size:
                self._query_embed_cache.popitem(last=False)
        return list(safe_vec)

    def _emit_search_trace(self, event: str, **payload: Any) -> None:
        if not self.search_trace_enabled:
            return
        total = float(payload.get("ms_total", 0.0) or 0.0)
        if self.search_trace_slow_ms > 0 and total < float(self.search_trace_slow_ms):
            return
        row = {"event": event, "ts_ms": int(time.time() * 1000)}
        row.update(payload)
        try:
            print(
                "[search-trace] "
                + json.dumps(row, ensure_ascii=False, separators=(",", ":"))
            )
        except Exception:
            return

    def _build_agentic_rewrite_plan(
        self, *, query: str, hits: list[dict[str, Any]], sufficiency: RetrievalSufficiency
    ) -> AgenticRewritePlan:
        if sufficiency.sufficient:
            return AgenticRewritePlan(
                query=query,
                source="noop_sufficient",
                confidence=float(sufficiency.confidence),
                reason=sufficiency.reason,
            )
        rewriter = self.query_rewriter
        if rewriter is not None:
            try:
                decision = rewriter.rewrite(
                    query=query, hits=hits, insufficiency_reason=sufficiency.reason
                )
            except Exception:
                decision = None
            if (
                isinstance(decision, RewriteDecision)
                and decision.query.strip()
                and float(decision.confidence) >= self.query_rewriter_min_confidence
            ):
                return AgenticRewritePlan(
                    query=decision.query.strip(),
                    source="llm_rewriter",
                    confidence=float(decision.confidence),
                    reason=str(decision.reason or "llm_rewriter")[:80],
                )
        rewritten = self._rewrite_query(query, hits)
        return AgenticRewritePlan(
            query=rewritten,
            source="heuristic_rewrite" if rewritten.strip() != query.strip() else "heuristic_noop",
            confidence=0.5 if rewritten.strip() != query.strip() else 0.0,
            reason="heuristic",
        )

    def _run_agentic_second_round(
        self,
        *,
        policy: EffectivePolicy,
        original_query: str,
        rewritten_query: str,
        first_round_hits: list[dict[str, Any]],
        insufficiency_reason: str,
        user_id: str | None,
        group_id: str | None,
        top_k: int,
        candidate_episode_ids: set[str] | None,
        as_of_ts: int | None,
        start_ts: int | None,
        end_ts: int | None,
    ) -> list[dict[str, Any]]:
        queries = self._build_agentic_query_variants(
            original_query=original_query,
            rewritten_query=rewritten_query,
            first_round_hits=first_round_hits,
            insufficiency_reason=insufficiency_reason,
        )

        rounds: list[list[dict[str, Any]]] = []
        for q in queries:
            rows = self._call_basic_search_with_optional_as_of(
                policy=policy,
                query=q,
                user_id=user_id,
                group_id=group_id,
                top_k=top_k,
                candidate_episode_ids=candidate_episode_ids,
                as_of_ts=as_of_ts,
                start_ts=start_ts,
                end_ts=end_ts,
            )
            rounds.append(rows)
        if not rounds:
            return []
        if len(rounds) == 1:
            second = rounds[0]
        else:
            second = reciprocal_rank_fusion(rounds, key="id", rrf_k=policy.rrf_k)
            second = second[: max(1, int(top_k))]
        second = self._rerank_query_overlap_rows(rows=second, query_variants=queries)
        if self.phase4_reasoning_enabled and self._is_temporal_query(original_query):
            second = self._rerank_temporal_rows(query=original_query, rows=second)
        return second

    def _build_agentic_query_variants(
        self,
        *,
        original_query: str,
        rewritten_query: str,
        first_round_hits: list[dict[str, Any]],
        insufficiency_reason: str,
    ) -> list[str]:
        seed = " ".join(str(rewritten_query or "").strip().split())
        if not seed:
            seed = " ".join(str(original_query or "").strip().split())
        queries = [seed] if seed else []
        max_queries = max(1, int(self.multi_hop_max_queries))
        rewriter = self.query_rewriter
        if max_queries > 1 and rewriter is not None and hasattr(rewriter, "expand_queries"):
            try:
                decision = rewriter.expand_queries(  # type: ignore[attr-defined]
                    query=seed,
                    hits=first_round_hits,
                    insufficiency_reason=insufficiency_reason,
                    max_queries=max_queries - 1,
                )
            except Exception:
                decision = None
            if (
                isinstance(decision, QueryExpansionDecision)
                and float(decision.confidence) >= self.query_rewriter_min_confidence
            ):
                seen = {q.lower() for q in queries}
                for item in decision.queries:
                    q = " ".join(str(item or "").strip().split())
                    if not q:
                        continue
                    key = q.lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    queries.append(q)
                    if len(queries) >= max_queries:
                        break
        if (
            self.phase4_reasoning_enabled
            and self._is_multi_hop_query(original_query)
            and len(queries) < max_queries
        ):
            extras = self._expand_multi_hop_queries(seed)
            seen = {q.lower() for q in queries}
            for item in extras:
                q = " ".join(str(item or "").strip().split())
                if not q:
                    continue
                key = q.lower()
                if key in seen:
                    continue
                seen.add(key)
                queries.append(q)
                if len(queries) >= max_queries:
                    break
        return queries[:max_queries]

    def _rerank_query_overlap_rows(
        self, *, rows: list[dict[str, Any]], query_variants: list[str]
    ) -> list[dict[str, Any]]:
        if not rows:
            return rows
        variant_tokens = [
            self._tokenize_search_text(q)
            for q in query_variants
            if str(q or "").strip()
        ]
        variant_tokens = [x for x in variant_tokens if x]
        if not variant_tokens:
            return rows
        weight = 0.18
        weighted: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            text = " ".join(
                [
                    str(item.get("summary", "")),
                    str(item.get("episode", "")),
                    str(item.get("subject", "")),
                ]
            ).strip()
            row_tokens = self._tokenize_search_text(text)
            overlap = (
                max(self._token_overlap(qt, row_tokens) for qt in variant_tokens)
                if row_tokens
                else 0.0
            )
            base = float(item.get("score", 0.0))
            item["query_overlap_bonus"] = round(float(overlap), 4)
            item["score"] = float(base) + weight * float(overlap)
            weighted.append(item)
        weighted.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        return weighted

    def _call_basic_search_with_optional_as_of(
        self,
        *,
        policy: EffectivePolicy,
        query: str,
        user_id: str | None,
        group_id: str | None,
        top_k: int,
        candidate_episode_ids: set[str] | None,
        as_of_ts: int | None,
        start_ts: int | None,
        end_ts: int | None,
    ) -> list[dict[str, Any]]:
        return self._basic_search(
            policy,
            query,
            user_id,
            group_id,
            top_k,
            candidate_episode_ids,
            attach_foresight=False,
            as_of_ts=as_of_ts,
            start_ts=start_ts,
            end_ts=end_ts,
        )

    @staticmethod
    def _is_multi_hop_query(query: str) -> bool:
        q = str(query or "").strip().lower()
        if not q:
            return False
        connectors = (" and ", " then ", "之后", "然后", "以及", "并且", "同时", "再")
        return any(x in q for x in connectors) and len(q) >= 10

    def _expand_multi_hop_queries(self, query: str) -> list[str]:
        text = str(query or "").strip()
        if not text:
            return []
        parts = re.split(r"(?:\s+and\s+|\s+then\s+|之后|然后|以及|并且|同时|再)", text)
        out: list[str] = []
        for part in parts:
            p = str(part or "").strip(" ，,。;；")
            if len(p) < 4:
                continue
            out.append(p)
            if len(out) >= self.multi_hop_max_queries:
                break
        return out

    @staticmethod
    def _is_temporal_query(query: str) -> bool:
        q = str(query or "").strip().lower()
        if not q:
            return False
        if re.search(r"\b(19|20)\d{2}\b", q):
            return True
        terms = (
            "最近",
            "刚刚",
            "刚才",
            "上周",
            "上个月",
            "去年",
            "明年",
            "when",
            "before",
            "after",
            "timeline",
            "earliest",
            "latest",
        )
        return any(t in q for t in terms)

    def _rerank_temporal_rows(self, *, query: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not rows:
            return rows
        q = str(query or "").lower()
        target_year = self._extract_target_year(q)
        latest_pref = any(x in q for x in ("最近", "latest", "recent", "刚刚", "newest"))
        earliest_pref = any(x in q for x in ("最早", "earliest", "first"))
        weighted: list[dict[str, Any]] = []
        ts_values = [int(r.get("timestamp") or 0) for r in rows if int(r.get("timestamp") or 0) > 0]
        max_ts = max(ts_values) if ts_values else 0
        min_ts = min(ts_values) if ts_values else 0
        for row in rows:
            item = dict(row)
            base = float(item.get("score", 0.0))
            ts = int(item.get("timestamp") or 0)
            temporal_bonus = 0.0
            if target_year is not None and ts > 0:
                row_year = int(time.strftime("%Y", time.localtime(ts)))
                year_gap = abs(row_year - target_year)
                temporal_bonus += max(0.0, 1.0 - 0.18 * float(year_gap))
            if latest_pref and ts > 0 and max_ts > min_ts:
                temporal_bonus += float(ts - min_ts) / float(max_ts - min_ts)
            if earliest_pref and ts > 0 and max_ts > min_ts:
                temporal_bonus += float(max_ts - ts) / float(max_ts - min_ts)
            item["temporal_bonus"] = round(float(temporal_bonus), 4)
            item["score"] = float(base) + self.temporal_rerank_weight * float(temporal_bonus)
            weighted.append(item)
        weighted.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        return weighted

    @staticmethod
    def _extract_target_year(query: str) -> int | None:
        m = re.search(r"((?:19|20)\d{2})\s*年?", str(query or ""))
        if not m:
            return None
        try:
            return int(m.group(1))
        except Exception:
            return None

    @staticmethod
    def _rewrite_query(query: str, hits: list[dict[str, Any]]) -> str:
        clues: list[str] = []
        for row in hits[:3]:
            summary = str(row.get("summary", "")).strip()
            subject = str(row.get("subject", "")).strip()
            if summary:
                clues.append(summary[:24])
            if subject:
                clues.append(subject)
        if not clues:
            return query
        return f"{query} 相关时间 人物 地点\n相关背景 {'; '.join(clues[:3])}\n时间线索"

    def _hydrate_vector_results(self, episode_ids: list[str]) -> list[dict[str, Any]]:
        if not episode_ids:
            return []
        unique_ids = [x for x in dict.fromkeys(episode_ids) if x]
        return self.repo.fetch_episodes_by_ids(unique_ids)

    def _fallback_recent(
        self,
        user_id: str | None,
        group_id: str | None,
        top_k: int,
        candidate_episode_ids: set[str] | None,
        attach_foresight: bool = True,
        as_of_ts: int | None = None,
        start_ts: int | None = None,
        end_ts: int | None = None,
    ) -> list[dict[str, Any]]:
        rows = self.repo.fetch_episodes(
            user_id=user_id,
            group_id=group_id,
            limit=max(1, top_k),
            candidate_episode_ids=candidate_episode_ids,
        )
        total = max(1, len(rows))
        for idx, row in enumerate(rows):
            row["score"] = max(0.0, 1.0 - (idx / (total + 0.01)))
            row["source"] = "fallback_recent"
        if not attach_foresight:
            return rows
        return self._attach_valid_foresight(
            rows=rows,
            user_id=user_id,
            group_id=group_id,
            as_of_ts=as_of_ts,
            start_ts=start_ts,
            end_ts=end_ts,
        )

    def _scene_guided_candidate_ids(
        self, query: str, user_id: str | None, group_id: str | None, top_scene_n: int
    ) -> dict[str, Any]:
        candidates = self.repo.get_memscene_candidates(
            user_id=user_id, group_id=group_id, limit=max(20, top_scene_n * 4)
        )
        try:
            q_vec = _safe_vector(self._embed_query(query))
        except Exception:
            q_vec = []
        scored: list[tuple[str, float]] = []
        for row in candidates:
            scene_vec = _safe_vector(row.get("centroid_vector_json"))
            if not scene_vec:
                continue
            sim = _cosine_similarity(q_vec, scene_vec)
            if sim >= 0.05:
                scored.append((str(row["id"]), sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        top_scene_ids = [sid for sid, _ in scored[: max(1, top_scene_n)]]
        episode_ids = self.repo.get_episode_ids_by_memscene_ids(top_scene_ids, limit=500)
        return {
            "episode_ids": episode_ids,
            "scene_score_map": {sid: s for sid, s in scored},
            "episode_scene_map": self.repo.get_episode_scene_map(episode_ids),
        }

    def _decide_storage_tier(self, *, importance: float, has_entity_relation: bool) -> str:
        is_vector_candidate = float(importance) >= self.vector_write_min_importance
        if has_entity_relation and is_vector_candidate:
            return "vector_graph"
        if has_entity_relation:
            return "graph_text"
        if is_vector_candidate:
            return "vector_only"
        return "text_only"

    @staticmethod
    def _format_turn_markdown(
        *, user_text: str, assistant_text: str, timestamp: int
    ) -> str:
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
        return (
            f"### Turn @ {ts}\n"
            f"**User**: {user_text.strip()}\n\n"
            f"**Assistant**: {assistant_text.strip()}"
        )

    def _decide_segment_boundary(
        self, *, segment: dict[str, Any], query: str, now_ts: int
    ) -> tuple[bool, str, float]:
        semantic = self._detect_semantic_boundary(segment=segment, query=query, now_ts=now_ts)
        if (
            semantic
            and semantic.should_cut
            and float(semantic.confidence) >= self.semantic_boundary_min_confidence
        ):
            return True, f"semantic:{semantic.reason}", float(semantic.confidence)

        should_cut = self._should_cut_segment(segment=segment, query=query, now_ts=now_ts)
        if should_cut:
            if semantic:
                return (
                    True,
                    f"heuristic_after_semantic:{semantic.reason}",
                    float(semantic.confidence),
                )
            return True, "heuristic", 0.0
        if semantic:
            return False, f"semantic_no_cut:{semantic.reason}", float(semantic.confidence)
        return False, "none", 0.0

    def _detect_semantic_boundary(
        self, *, segment: dict[str, Any], query: str, now_ts: int
    ) -> BoundaryDecision | None:
        enhancer = self.formation_enhancer
        if enhancer is None:
            return None
        recent_queries = self._recent_user_queries_from_segment(segment, limit=5)
        last_ts = int(segment.get("last_time") or now_ts)
        idle_seconds = max(0, int(now_ts - last_ts))
        turn_count = int(segment.get("turn_count") or 0)
        try:
            return enhancer.detect_boundary(
                query=query,
                recent_user_queries=recent_queries,
                turn_count=turn_count,
                idle_seconds=idle_seconds,
            )
        except Exception:
            return None

    def _should_cut_segment(
        self, *, segment: dict[str, Any], query: str, now_ts: int
    ) -> bool:
        last_ts = int(segment.get("last_time") or now_ts)
        turn_count = int(segment.get("turn_count") or 0)
        if now_ts - last_ts > 1800:
            return True
        q = query.strip().lower()
        if any(k in q for k in ("另外", "换个话题", "by the way", "new topic")):
            return True
        if turn_count >= 6:
            return True
        if turn_count < 2:
            return False
        prev_q = str(segment.get("last_query") or "").strip()
        recent_queries = self._recent_user_queries_from_segment(segment, limit=3)
        reference = " ".join([x for x in recent_queries if x.strip()]) or prev_q
        try:
            a = _safe_vector(self.embedding_provider.embed(reference))
            b = _safe_vector(self.embedding_provider.embed(query))
            return _cosine_similarity(a, b) < 0.42
        except Exception:
            # lexical fallback for environments where embedding service is unavailable
            base_tokens = {t for t in reference.lower().split() if t}
            cur_tokens = {t for t in query.lower().split() if t}
            if len(base_tokens) < 3 or len(cur_tokens) < 3:
                return False
            overlap = len(base_tokens & cur_tokens) / max(1, len(base_tokens | cur_tokens))
            return overlap < 0.18

    @staticmethod
    def _recent_user_queries_from_segment(
        segment: dict[str, Any], limit: int = 3
    ) -> list[str]:
        turns = str(segment.get("turns_markdown") or "")
        if not turns:
            return []
        lines = [line.strip() for line in turns.splitlines() if line.strip()]
        out: list[str] = []
        for line in reversed(lines):
            if not line.startswith("**User**:"):
                continue
            text = line.split(":", 1)[-1].strip()
            if text:
                out.append(text)
            if len(out) >= max(1, limit):
                break
        out.reverse()
        return out

    def _commit_segment_as_memory(
        self,
        *,
        conversation_id: str,
        segment: dict[str, Any],
        user_id: str,
        group_id: str | None,
        policy: EffectivePolicy,
    ) -> bool:
        markdown = str(segment.get("turns_markdown") or "").strip()
        if not markdown:
            return False
        memory_content = self._compose_segment_memory_content(
            turns_markdown=markdown,
            user_id=user_id,
            group_id=group_id,
        )
        seq = int(segment.get("segment_seq") or 1)
        now_ts = int(segment.get("last_time") or time.time())
        result = self.memorize(
            MemorizeInput(
                message_id=f"{conversation_id}-seg-{seq}",
                create_time=now_ts,
                sender=user_id,
                content=memory_content,
                group_id=group_id,
                group_name=None,
                sender_name=user_id,
                role="assistant",
            ),
            request_id=f"{conversation_id}-seg-{seq}-{uuid.uuid4().hex[:8]}",
        )
        if policy.vector_enabled and result.get("memory"):
            try:
                self.maybe_index_vector(policy, result["memory"])
            except Exception:
                pass
        return True

    def _compose_segment_memory_content(
        self, *, turns_markdown: str, user_id: str, group_id: str | None
    ) -> str:
        base = str(turns_markdown or "").strip()
        if not base:
            return ""
        enhancer = self.formation_enhancer
        if enhancer is None:
            return base
        try:
            narrative = enhancer.synthesize_narrative(
                turns_markdown=base,
                user_id=user_id,
                group_id=group_id,
            )
        except Exception:
            narrative = None
        narrative_text = " ".join(str(narrative or "").strip().split())[:1200]
        if not narrative_text:
            return base
        if narrative_text in base:
            return base
        return f"### Narrative\n{narrative_text}\n\n### Conversation\n{base}"

    def _attach_valid_foresight(
        self,
        rows: list[dict[str, Any]],
        user_id: str | None,
        group_id: str | None,
        as_of_ts: int | None = None,
        start_ts: int | None = None,
        end_ts: int | None = None,
    ) -> list[dict[str, Any]]:
        ids = [str(r.get("id", "")) for r in rows if r.get("id")]
        foresight_map = self.repo.get_valid_foresights_for_episodes(
            episode_ids=ids,
            user_id=user_id,
            group_id=group_id,
            as_of_ts=as_of_ts,
            start_ts=start_ts,
            end_ts=end_ts,
        )
        for row in rows:
            rid = str(row.get("id", ""))
            valid = foresight_map.get(rid, [])
            row["valid_foresights"] = valid
            row["has_valid_foresight"] = bool(valid)
        return rows

    def _merge_graph_hits(
        self,
        query: str,
        user_id: str | None,
        group_id: str | None,
        base_hits: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        if not self.graph_store.enabled:
            return base_hits[: max(1, top_k)]
        try:
            graph_rows = self.graph_store.search(
                query=query,
                user_id=user_id,
                group_id=group_id,
                limit=max(1, min(self.graph_top_k, top_k)),
            )
        except Exception:
            graph_rows = []
        merged = list(base_hits)
        existing_keys = {
            f"{str(row.get('subject', ''))}|{str(row.get('summary', ''))}"
            for row in merged
        }
        for g in graph_rows:
            summary = f"{g['subject']} -[{g['relation']}]-> {g['obj']}"
            dedup_key = f"{str(g.get('subject', ''))}|{summary}"
            if dedup_key in existing_keys:
                continue
            existing_keys.add(dedup_key)
            confidence = float(g.get("confidence", 0.5))
            match_score = float(g.get("match_score", 0.0))
            graph_score = (
                0.38
                + 0.27 * confidence
                + 0.35 * match_score
            )
            graph_importance = max(0.36, min(0.92, 0.30 + 0.35 * confidence + 0.35 * match_score))
            merged.append(
                {
                    "id": f"graph:{uuid.uuid4().hex}",
                    "event_id": g.get("event_id"),
                    "timestamp": g.get("timestamp"),
                    "summary": summary,
                    "subject": g.get("subject", ""),
                    "episode": summary,
                    "score": graph_score,
                    "importance_score": graph_importance,
                    "source": "graph_kuzu",
                    "confidence": confidence,
                    "storage_tier": "graph_text",
                }
            )
        merged.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        return merged[: max(1, top_k)]

    @staticmethod
    def _annotate_scene_info(
        rows: list[dict[str, Any]],
        episode_scene_map: dict[str, str],
        scene_score_map: dict[str, float],
    ) -> None:
        for row in rows:
            eid = str(row.get("id", ""))
            sid = str(episode_scene_map.get(eid, "")) if eid else ""
            row["memscene_id"] = sid
            similarity = float(scene_score_map.get(sid, 0.0)) if sid else 0.0
            row["similarity"] = similarity
            row["memscene_similarity"] = similarity
            row["memscene_score"] = similarity

    @staticmethod
    def _tier_supports_vector(tier: str) -> bool:
        name = str(tier or "").strip().lower()
        return name in {"vector_graph", "vector_only", "hybrid"}

    @staticmethod
    def _tier_supports_graph(tier: str) -> bool:
        name = str(tier or "").strip().lower()
        return name in {"vector_graph", "graph_text", "hybrid"}

    def _adjust_hybrid_score(self, row: dict[str, Any], raw_score: float) -> float:
        tier = str(row.get("storage_tier") or "text_only")
        importance = float(row.get("importance_score", 0.0))
        score = max(0.0, float(raw_score))
        if self._tier_supports_vector(tier):
            score *= 1.08
        if self._tier_supports_graph(tier):
            score *= 1.05
        if importance >= self.key_memory_importance_threshold:
            score *= 1.06
        return score

    def _profile_hint_hits(
        self, *, query: str, user_id: str | None, group_id: str | None
    ) -> list[dict[str, Any]]:
        profile = self.repo.get_latest_profile_snapshot(user_id=user_id, group_id=group_id)
        if not profile:
            return []
        data = profile.get("profile")
        if not isinstance(data, dict):
            return []
        explicit = data.get("explicit_facts")
        if not isinstance(explicit, dict):
            return []
        is_identity = self._is_identity_query(query)
        rows: list[dict[str, Any]] = []
        query_tokens = self._tokenize_search_text(query)
        for field, value_obj in explicit.items():
            if not isinstance(value_obj, dict):
                continue
            value = str(value_obj.get("value", "")).strip()
            if not value:
                continue
            text = f"{field} {value}"
            overlap = self._token_overlap(query_tokens, self._tokenize_search_text(text))
            if field == "name" and is_identity:
                overlap = max(overlap, 0.99)
            if overlap <= 0.02 and not (is_identity and field == "name"):
                continue
            rows.append(
                {
                    "id": f"profile:{field}",
                    "event_id": str(profile.get("event_id") or ""),
                    "timestamp": int(value_obj.get("timestamp") or profile.get("timestamp") or time.time()),
                    "summary": f"{field}: {value}",
                    "subject": str(profile.get("user_id") or ""),
                    "episode": f"用户{field}为{value}",
                    "importance_score": 0.95 if field == "name" else 0.82,
                    "storage_tier": "text_only",
                    "score": 0.72 + overlap,
                    "source": "profile_snapshot",
                }
            )
        rows.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        return rows[:3]

    def _merge_profile_hits(
        self, base_hits: list[dict[str, Any]], profile_hits: list[dict[str, Any]], top_k: int
    ) -> list[dict[str, Any]]:
        if not profile_hits:
            return base_hits[: max(1, top_k)]
        merged = list(base_hits)
        seen = {str(row.get("id", "")) for row in merged if row.get("id")}
        for row in profile_hits:
            rid = str(row.get("id", ""))
            if rid in seen:
                continue
            seen.add(rid)
            merged.append(row)
        merged.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        return merged[: max(1, top_k)]

    def _is_identity_query(self, query: str) -> bool:
        q = str(query or "").strip().lower()
        if not q:
            return False
        patterns = (
            r"我叫.*什么",
            r"我的名字",
            r"我是谁",
            r"who am i",
            r"what(?:'s| is) my name",
        )
        return any(re.search(p, q, flags=re.IGNORECASE) for p in patterns)

    def _lexical_overlap_ratio(self, query: str, text: str) -> float:
        return self._token_overlap(
            self._tokenize_search_text(query),
            self._tokenize_search_text(text),
        )

    @staticmethod
    def _token_overlap(a: set[str], b: set[str]) -> float:
        if not a or not b:
            return 0.0
        return float(len(a & b)) / float(max(1, len(a | b)))

    @staticmethod
    def _tokenize_search_text(text: str) -> set[str]:
        raw = str(text or "").lower().strip()
        if not raw:
            return set()
        tokens = {t for t in re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]{1,4}", raw) if t}
        chars = re.findall(r"[\u4e00-\u9fff]", raw)
        for i in range(len(chars) - 1):
            tokens.add(chars[i] + chars[i + 1])
        if chars:
            tokens.add("".join(chars))
        stop = {"我", "你", "他", "她", "它", "吗", "呢", "啊", "呀", "的", "了"}
        return {t for t in tokens if t and t not in stop}


def _safe_vector(value) -> list[float]:
    if isinstance(value, list):
        return [float(x) for x in value if isinstance(x, (int, float))]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [float(x) for x in parsed if isinstance(x, (int, float))]
        except Exception:
            return []
    return []


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
