from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
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
from evermemos_lite.service.event_log_extractor import RuleEventLogExtractor
from evermemos_lite.service.extractor import (
    ExtractedMemory,
    MemoryExtractor,
    RuleMemoryExtractor,
)
from evermemos_lite.service.foresight_extractor import (
    ForesightExtractorProtocol,
    RuleForesightExtractor,
)
from evermemos_lite.service.formation_enhancer import (
    BoundaryDecision,
    FormationEnhancerProtocol,
)
from evermemos_lite.service.graph_extractor import extract_graph_triples
from evermemos_lite.service.memcell_extractor import RuleMemCellExtractor
from evermemos_lite.service.query_rewriter import (
    HyDEDecision,
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


class RerankProviderProtocol(Protocol):
    def rerank(self, *, query: str, documents: list[str]) -> list[float]:
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


@dataclass(frozen=True)
class MultiStageExtractResult:
    extracted: ExtractedMemory
    event_log_items: list[Any]
    foresights: list[dict[str, Any]]


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
        rerank_provider: RerankProviderProtocol | None = None,
        rerank_trigger_k: int = 20,
        rerank_top_n: int = 20,
        rerank_timeout_ms: int = 220,
        phase4_reasoning_enabled: bool = True,
        temporal_rerank_weight: float = 0.35,
        multi_hop_max_queries: int = 6,
        graph_top_k: int = 3,
        graph_write_min_importance: float = 0.6,
        key_memory_importance_threshold: float = 0.5,
        vector_write_min_importance: float = 0.1,
        vector_embed_chunk_chars: int = 600,
        vector_embed_max_chunks: int = 8,
        search_budget_factor: int = 8,
        search_min_probe_k: int = 24,
        keyword_confident_best_score: float = 9.0,
        keyword_confident_kth_score: float = 2.8,
        semantic_vector_budget_cap: int = 64,
        semantic_keyword_budget_cap: int = 32,
        query_embed_cache_size: int = 256,
        query_embed_cache_ttl_sec: int = 900,
        search_trace_enabled: bool = False,
        search_trace_slow_ms: int = 0,
        event_log_vector_store: LanceVectorStore | None = None,
        foresight_extractor: ForesightExtractorProtocol | None = None,
        extract_max_retries: int = 3,
        recall_mode: bool = False,
        agentic_round_min_k: int = 24,
        agentic_round_max_k: int = 120,
        agentic_force_second_round: bool = False,
    ) -> None:
        self.engine = engine
        self.repo = MemoryRepository(engine)
        self.vector_store = vector_store
        self.event_log_vector_store = event_log_vector_store
        self.embedding_provider = embedding_provider
        self.extractor = extractor
        self._retrieval_compactor = RuleMemoryExtractor()
        self.memcell_extractor = RuleMemCellExtractor()
        self.event_log_extractor = RuleEventLogExtractor()
        self.foresight_extractor = foresight_extractor or RuleForesightExtractor()
        self.extract_max_retries = max(1, int(extract_max_retries))
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
        self.rerank_provider = rerank_provider
        self.rerank_trigger_k = max(2, int(rerank_trigger_k))
        self.rerank_top_n = max(1, int(rerank_top_n))
        self.rerank_timeout_ms = max(60, int(rerank_timeout_ms))
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
        self.recall_mode = bool(recall_mode)
        self.agentic_round_min_k = max(8, int(agentic_round_min_k))
        self.agentic_round_max_k = max(self.agentic_round_min_k, int(agentic_round_max_k))
        self.agentic_force_second_round = bool(agentic_force_second_round)
        self._query_embed_cache: OrderedDict[str, tuple[int, list[float]]] = OrderedDict()
        self._query_embed_lock = Lock()
        self._vector_index_stats: dict[str, int] = {}
        self._vector_index_lock = Lock()

    @staticmethod
    def estimate_importance(content: str) -> float:
        score = min(1.0, len(content.strip()) / 400.0 + 0.2)
        return round(float(score), 3)

    def _run_multi_stage_extract(self, *, payload: MemorizeInput) -> MultiStageExtractResult:
        extracted = self._extract_episode_with_retry(
            content=payload.content,
            sender=payload.sender,
            group_id=payload.group_id,
        )
        gated = self._apply_extract_quality_gate(
            extracted=extracted,
            fallback_episode=str(payload.content or "").strip(),
            sender=payload.sender,
            group_id=payload.group_id,
        )
        episode_text = str(gated.episode or "").strip() or str(payload.content or "").strip()
        event_log_items = self._extract_event_logs_with_retry(
            atomic_facts=list(gated.atomic_facts),
            episode=episode_text,
        )
        foresights = self._extract_foresights_with_retry(
            episode=episode_text,
            atomic_facts=list(gated.atomic_facts),
            existing=list(gated.foresights),
        )
        enriched = ExtractedMemory(
            episode=episode_text,
            summary=str(gated.summary or "").strip(),
            subject=str(gated.subject or payload.sender or "user").strip()[:60] or "user",
            importance_score=float(gated.importance_score),
            atomic_facts=list(gated.atomic_facts),
            foresights=list(foresights),
            profile_patch=dict(gated.profile_patch),
        )
        return MultiStageExtractResult(
            extracted=enriched,
            event_log_items=event_log_items,
            foresights=foresights,
        )

    def _extract_episode_with_retry(
        self, *, content: str, sender: str, group_id: str | None
    ) -> ExtractedMemory:
        last_exc: Exception | None = None
        for _ in range(self.extract_max_retries):
            try:
                out = self.extractor.extract(
                    content=content,
                    sender=sender,
                    group_id=group_id,
                )
                if isinstance(out, ExtractedMemory):
                    return out
            except Exception as exc:
                last_exc = exc
                continue
        fallback = self._retrieval_compactor.extract(
            content=content, sender=sender, group_id=group_id
        )
        if isinstance(fallback, ExtractedMemory):
            return fallback
        if last_exc is not None:
            raise last_exc
        return ExtractedMemory(
            episode=str(content or "").strip(),
            summary=str(content or "").strip()[:220],
            subject=str(sender or "user").strip()[:60] or "user",
            importance_score=self.estimate_importance(str(content or "").strip()),
            atomic_facts=[],
            foresights=[],
            profile_patch={},
        )

    def _extract_event_logs_with_retry(
        self, *, atomic_facts: list[str], episode: str
    ) -> list[Any]:
        for _ in range(self.extract_max_retries):
            try:
                rows = self.event_log_extractor.extract(
                    atomic_facts=list(atomic_facts),
                    episode=episode,
                )
            except Exception:
                rows = []
            if rows:
                return list(rows)
        try:
            return list(
                RuleEventLogExtractor().extract(
                    atomic_facts=list(atomic_facts), episode=episode
                )
            )
        except Exception:
            return []

    def _extract_foresights_with_retry(
        self,
        *,
        episode: str,
        atomic_facts: list[str],
        existing: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        for _ in range(self.extract_max_retries):
            try:
                rows = self.foresight_extractor.extract(
                    episode=episode,
                    atomic_facts=list(atomic_facts),
                    existing=list(existing),
                )
            except Exception:
                rows = []
            if rows:
                return list(rows)
        try:
            return list(
                RuleForesightExtractor().extract(
                    episode=episode,
                    atomic_facts=list(atomic_facts),
                    existing=list(existing),
                )
            )
        except Exception:
            return []

    def _extract_memcells_with_retry(self, *, episode: str) -> list[Any]:
        for _ in range(self.extract_max_retries):
            try:
                rows = self.memcell_extractor.split(episode)
            except Exception:
                rows = []
            if rows:
                return list(rows)
        try:
            return list(RuleMemCellExtractor().split(episode))
        except Exception:
            return []

    def _apply_extract_quality_gate(
        self,
        *,
        extracted: ExtractedMemory,
        fallback_episode: str,
        sender: str,
        group_id: str | None,
    ) -> ExtractedMemory:
        episode = str(extracted.episode or "").strip() or str(fallback_episode or "").strip()
        summary = str(extracted.summary or "").strip()
        subject = str(extracted.subject or sender or "user").strip()[:60] or "user"
        atomic_facts = [
            str(x).strip()[:180]
            for x in list(extracted.atomic_facts or [])
            if str(x).strip()
        ]
        profile_patch = dict(extracted.profile_patch or {})
        min_atomic_needed = 3 if len(episode) >= 240 else 1
        if len(atomic_facts) < min_atomic_needed:
            try:
                rule_rows = self._retrieval_compactor.extract(
                    content=episode,
                    sender=sender,
                    group_id=group_id,
                )
                merged: list[str] = []
                seen: set[str] = set()
                for item in [*atomic_facts, *list(rule_rows.atomic_facts or [])]:
                    fact = str(item or "").strip()
                    if not fact:
                        continue
                    key = fact.lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    merged.append(fact[:180])
                    if len(merged) >= 28:
                        break
                atomic_facts = merged
                if not summary:
                    summary = str(rule_rows.summary or "").strip()
                if not profile_patch:
                    profile_patch = dict(rule_rows.profile_patch or {})
            except Exception:
                pass
        if not summary:
            if atomic_facts:
                summary = "；".join(atomic_facts[:8])[:260]
            else:
                summary = episode[:220]
        importance = float(extracted.importance_score)
        if importance <= 0.0 or importance > 1.0:
            importance = self.estimate_importance(episode)
        return ExtractedMemory(
            episode=episode,
            summary=summary[:360],
            subject=subject,
            importance_score=max(0.0, min(1.0, importance)),
            atomic_facts=atomic_facts[:32],
            foresights=list(extracted.foresights or []),
            profile_patch=profile_patch,
        )

    def memorize(self, payload: MemorizeInput, request_id: str) -> dict[str, Any]:
        staged = self._run_multi_stage_extract(payload=payload)
        extracted = staged.extracted
        raw_episode = str(payload.content or "").strip()
        extracted_episode = str(extracted.episode or "").strip()
        if self._should_preserve_raw_episode(
            raw_episode=raw_episode,
            extracted_episode=extracted_episode,
        ):
            episode_text = raw_episode
        elif (
            self._looks_encoding_broken(extracted_episode)
            and not self._looks_encoding_broken(raw_episode)
            and raw_episode
        ):
            episode_text = raw_episode
        else:
            episode_text = extracted_episode or raw_episode
        extracted_importance = float(extracted.importance_score)
        if extracted_importance <= 0.01:
            extracted_importance = self.estimate_importance(episode_text)
        triples = extract_graph_triples(
            facts=list(extracted.atomic_facts) or [episode_text],
            user_id=payload.sender,
        )
        has_entity_relation = bool(triples)
        storage_tier = self._decide_storage_tier(
            importance=float(extracted_importance),
            has_entity_relation=has_entity_relation,
        )
        memory_category = self._decide_memory_category(
            episode=episode_text,
            summary=extracted.summary,
            subject=extracted.subject,
            atomic_facts=list(extracted.atomic_facts),
            profile_patch=extracted.profile_patch,
            foresights=list(extracted.foresights),
            has_entity_relation=has_entity_relation,
        )
        episode = self.repo.save_message_as_memory(
            message_id=payload.message_id,
            create_time=int(payload.create_time),
            sender=payload.sender,
            content=episode_text,
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
            foresights=staged.foresights,
            profile_patch=extracted.profile_patch,
            memory_category=memory_category,
            event_id=request_id,
        )
        memcells = self._extract_memcells_with_retry(episode=episode_text)
        memcell_count = self.repo.save_memcells(
            memory_id=str(episode.get("id", "")),
            event_id=str(episode.get("event_id", "")),
            user_id=payload.sender,
            group_id=payload.group_id,
            memcells=[cell.content for cell in memcells],
            created_at=int(payload.create_time),
        )
        event_log_count = self.repo.save_event_logs(
            memory_id=str(episode.get("id", "")),
            event_id=str(episode.get("event_id", "")),
            user_id=payload.sender,
            group_id=payload.group_id,
            event_logs=[
                {
                    "fact_order": int(item.fact_order),
                    "fact": item.fact,
                    "fact_norm": item.fact_norm,
                }
                for item in staged.event_log_items
            ],
            created_at=int(payload.create_time),
        )

        consolidate_result = self.semantic_consolidator.consolidate(
            ConsolidateInput(
                user_id=payload.sender,
                group_id=payload.group_id,
                event_id=episode["event_id"],
                memory_id=episode["id"],
                timestamp=int(payload.create_time),
                episode=episode_text,
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
            "memory_category": memory_category,
            "scene_id": episode.get("scene_id"),
            "memcell_count": int(memcell_count),
            "event_log_count": int(event_log_count),
        }

    @staticmethod
    def _looks_encoding_broken(text: str) -> bool:
        raw = str(text or "").strip()
        if not raw:
            return False
        q_count = raw.count("?")
        if len(raw) >= 4 and q_count >= max(2, len(raw) // 3):
            return True
        markers = ("Ã", "Â", "ä", "å", "æ", "ç", "è", "é", "ï", "ð", "椋", "炰", "锟")
        return any(m in raw for m in markers)

    @classmethod
    def _should_preserve_raw_episode(
        cls,
        *,
        raw_episode: str,
        extracted_episode: str,
    ) -> bool:
        raw_meta = cls._extract_metadata_dict(raw_episode)
        if not raw_meta:
            return False
        extracted_meta = cls._extract_metadata_dict(extracted_episode)
        if not extracted_meta:
            return True
        raw_keys = {str(k).strip() for k in raw_meta.keys() if str(k).strip()}
        extracted_keys = {str(k).strip() for k in extracted_meta.keys() if str(k).strip()}
        return bool(raw_keys - extracted_keys)

    @staticmethod
    def _extract_metadata_dict(text: str) -> dict[str, Any]:
        for line in str(text or "").splitlines():
            stripped = line.strip()
            if not stripped.lower().startswith("[metadata]"):
                continue
            raw_json = stripped[len("[metadata]") :].strip()
            if not raw_json:
                return {}
            try:
                parsed = json.loads(raw_json)
            except Exception:
                return {}
            return parsed if isinstance(parsed, dict) else {}
        return {}

    def maybe_index_vector(self, policy: EffectivePolicy, episode: dict[str, Any]) -> dict[str, Any]:
        event_log_index_result = self._index_event_log_vectors_for_memory(episode=episode)
        if not policy.vector_enabled:
            self._record_vector_index("skip.policy_disabled")
            return {
                "status": "skipped",
                "reason": "policy_disabled",
                "event_log_index": event_log_index_result,
            }
        tier = str(episode.get("storage_tier") or "text_only")
        if not self._tier_supports_vector(tier):
            self._record_vector_index("skip.tier_not_vector")
            return {
                "status": "skipped",
                "reason": "tier_not_vector",
                "tier": tier,
                "event_log_index": event_log_index_result,
            }
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
                "event_log_index": event_log_index_result,
            }
        try:
            vector = self._embed_episode_for_index(str(episode.get("episode", "")))
        except Exception as exc:
            self._record_vector_index("error.embed_failed")
            return {
                "status": "error",
                "reason": "embed_failed",
                "detail": str(exc)[:260],
                "event_log_index": event_log_index_result,
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
                "event_log_index": event_log_index_result,
            }
        self._record_vector_index("indexed.ok")
        return {
            "status": "indexed",
            "reason": "ok",
            "event_log_index": event_log_index_result,
        }

    def get_vector_index_stats(self) -> dict[str, int]:
        with self._vector_index_lock:
            return dict(self._vector_index_stats)

    def _record_vector_index(self, key: str) -> None:
        with self._vector_index_lock:
            self._vector_index_stats[key] = int(self._vector_index_stats.get(key, 0)) + 1

    def _fit_vector_dim(self, vector: list[float]) -> list[float]:
        return self._fit_vector_dim_for_dim(vector=vector, target_dim=self.vector_store.vector_dim)

    @staticmethod
    def _fit_vector_dim_for_dim(*, vector: list[float], target_dim: int) -> list[float]:
        dim = max(1, int(target_dim))
        if len(vector) == dim:
            return vector
        if len(vector) > dim:
            return vector[:dim]
        return vector + [0.0] * (dim - len(vector))

    def _index_event_log_vectors_for_memory(self, *, episode: dict[str, Any]) -> dict[str, Any]:
        store = self.event_log_vector_store
        if store is None or not bool(getattr(store, "enabled", False)):
            return {"status": "skipped", "reason": "event_log_vector_disabled", "count": 0}
        memory_id = str(episode.get("id", "")).strip()
        if not memory_id:
            return {"status": "skipped", "reason": "missing_memory_id", "count": 0}
        event_logs = self.repo.get_event_logs_by_memory_id(memory_id)
        if not event_logs:
            return {"status": "skipped", "reason": "empty_event_logs", "count": 0}
        upserted = 0
        errors: list[str] = []
        timestamp = int(episode.get("timestamp", 0) or time.time())
        importance_score = float(episode.get("importance_score", 0.0))
        user_id = episode.get("user_id")
        group_id = episode.get("group_id")
        for row in event_logs[:64]:
            fact = " ".join(str(row.get("fact", "")).split()).strip()
            if not fact:
                continue
            try:
                vec = self.embedding_provider.embed(fact)
                safe_vec = [float(x) for x in vec if isinstance(x, (int, float))]
                if not safe_vec:
                    continue
                fitted = self._fit_vector_dim_for_dim(
                    vector=safe_vec,
                    target_dim=int(getattr(store, "vector_dim", len(safe_vec))),
                )
                store.upsert(
                    uuid.uuid4().hex,
                    memory_id,
                    fitted,
                    {
                        "id": memory_id,
                        "user_id": user_id,
                        "group_id": group_id,
                        "timestamp": timestamp,
                        "importance_score": importance_score,
                        "event_id": row.get("event_id"),
                        "fact_order": int(row.get("fact_order", 0) or 0),
                        "fact": fact[:240],
                        "channel": "event_log",
                    },
                )
                upserted += 1
            except Exception as exc:
                errors.append(str(exc)[:120])
                continue
        if upserted > 0:
            return {"status": "indexed", "reason": "ok", "count": int(upserted)}
        if errors:
            return {
                "status": "error",
                "reason": "event_log_embed_or_upsert_failed",
                "count": 0,
                "detail": str(errors[-1])[:180],
            }
        return {"status": "skipped", "reason": "empty_after_filter", "count": 0}

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
        rows = self.repo.fetch_episodes(
            user_id=user_id,
            group_id=group_id,
            limit=max(1, limit),
            as_of_ts=as_of_ts,
            start_ts=start_ts,
            end_ts=end_ts,
        )
        rows = self._filter_rows_by_time_bounds(
            rows=rows,
            as_of_ts=as_of_ts,
            start_ts=start_ts,
            end_ts=end_ts,
        )
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
        self,
        query: str,
        user_id: str | None,
        group_id: str | None,
        limit: int,
        as_of_ts: int | None = None,
        start_ts: int | None = None,
        end_ts: int | None = None,
    ) -> list[dict[str, Any]]:
        if not self.graph_store.enabled:
            return []
        try:
            rows = self.graph_store.search(  # type: ignore[call-arg]
                query=query,
                user_id=user_id,
                group_id=group_id,
                limit=max(1, limit),
                as_of_ts=as_of_ts,
                start_ts=start_ts,
                end_ts=end_ts,
            )
        except TypeError:
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
        effective_as_of_ts, effective_start_ts, effective_end_ts, temporal_reason = (
            self._resolve_query_time_constraints(
                query=query,
                as_of_ts=as_of_ts,
                start_ts=start_ts,
                end_ts=end_ts,
            )
        )
        if policy.agentic_enabled:
            base = self._agentic_search(
                policy,
                query,
                user_id,
                group_id,
                top_k,
                as_of_ts=effective_as_of_ts,
                start_ts=effective_start_ts,
                end_ts=effective_end_ts,
            )
        else:
            if (
                effective_as_of_ts is None
                and effective_start_ts is None
                and effective_end_ts is None
            ):
                base = self._basic_search(policy, query, user_id, group_id, top_k, None)
            else:
                base = self._basic_search(
                    policy,
                    query,
                    user_id,
                    group_id,
                    top_k,
                    None,
                    as_of_ts=effective_as_of_ts,
                    start_ts=effective_start_ts,
                    end_ts=effective_end_ts,
                )
        after_base = time.perf_counter()
        profile_hints = self._profile_hint_hits(
            query=query,
            user_id=user_id,
            group_id=group_id,
            as_of_ts=effective_as_of_ts,
            start_ts=effective_start_ts,
            end_ts=effective_end_ts,
        )
        after_profile = time.perf_counter()
        merged_base = self._merge_profile_hits(base, profile_hints, top_k)
        graph_eligible = self._is_graph_query_eligible(query)
        if graph_eligible:
            merged = self._merge_graph_hits(
                query,
                user_id,
                group_id,
                merged_base,
                top_k,
                rrf_k=int(policy.rrf_k),
                as_of_ts=effective_as_of_ts,
                start_ts=effective_start_ts,
                end_ts=effective_end_ts,
            )
        else:
            merged = merged_base[: max(1, int(top_k))]
        merged = self._filter_rows_by_time_bounds(
            rows=merged,
            as_of_ts=effective_as_of_ts,
            start_ts=effective_start_ts,
            end_ts=effective_end_ts,
        )
        compacted = self._compress_retrieval_rows(query=query, rows=merged)
        final_rows = self._dedup_and_density_merge_rows(
            query=query,
            rows=compacted,
            top_k=max(1, int(top_k)),
        )
        end = time.perf_counter()
        self._emit_search_trace(
            "search_total",
            query_len=len(str(query or "")),
            top_k=max(1, int(top_k)),
            profile=str(policy.profile),
            agentic=bool(policy.agentic_enabled),
            temporal_applied=bool(
                effective_as_of_ts is not None
                or effective_start_ts is not None
                or effective_end_ts is not None
            ),
            temporal_reason=temporal_reason,
            graph_eligible=bool(graph_eligible),
            output_count=len(final_rows),
            ms_total=round((end - start) * 1000.0, 3),
            ms_base=round((after_base - start) * 1000.0, 3),
            ms_profile=round((after_profile - after_base) * 1000.0, 3),
            ms_graph=round((end - after_profile) * 1000.0, 3),
        )
        return self._strip_internal_search_fields(final_rows)

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
        event_kw_ms = 0.0
        event_vec_ms = 0.0
        foresight_kw_ms = 0.0
        hydrate_ms = 0.0
        foresight_ms = 0.0
        vector_probe_error = False
        vector_probe_error_reason = ""
        keyword_hits: list[dict[str, Any]] = []
        vector_hits: list[dict[str, Any]] = []
        event_log_keyword_hits: list[dict[str, Any]] = []
        event_log_vector_hits: list[dict[str, Any]] = []
        foresight_keyword_hits: list[dict[str, Any]] = []
        top = max(1, int(top_k))
        pool_top = max(top, min(200, max(top * 3, top + 6)))
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
                as_of_ts=as_of_ts,
                start_ts=start_ts,
                end_ts=end_ts,
            )
            kw_ms = (time.perf_counter() - kw_started) * 1000.0
            for row in keyword_hits:
                row["source"] = "keyword"
                row["in_keyword_hits"] = True
            event_kw_started = time.perf_counter()
            event_log_keyword_hits = self.repo.search_event_log_keyword(
                query=query,
                user_id=user_id,
                group_id=group_id,
                top_k=keyword_budget,
                candidate_episode_ids=candidate_episode_ids,
                as_of_ts=as_of_ts,
                start_ts=start_ts,
                end_ts=end_ts,
            )
            event_kw_ms = (time.perf_counter() - event_kw_started) * 1000.0
            for row in event_log_keyword_hits:
                row["source"] = "event_log_keyword"
                row["in_event_log_keyword_hits"] = True
            foresight_kw_started = time.perf_counter()
            foresight_keyword_hits = self.repo.search_foresight_keyword(
                query=query,
                user_id=user_id,
                group_id=group_id,
                top_k=keyword_budget,
                candidate_episode_ids=candidate_episode_ids,
                as_of_ts=as_of_ts,
                start_ts=start_ts,
                end_ts=end_ts,
            )
            foresight_kw_ms = (time.perf_counter() - foresight_kw_started) * 1000.0
            for row in foresight_keyword_hits:
                row["source"] = "foresight_keyword"
                row["in_foresight_keyword_hits"] = True

        keyword_hits = self._dedupe_hit_rows_by_memory(keyword_hits)
        event_log_keyword_hits = self._dedupe_hit_rows_by_memory(event_log_keyword_hits)
        foresight_keyword_hits = self._dedupe_hit_rows_by_memory(foresight_keyword_hits)
        all_keyword_hits = self._dedupe_hit_rows_by_memory(
            [*keyword_hits, *event_log_keyword_hits, *foresight_keyword_hits]
        )
        keyword_mid_set = {
            str(x.get("memory_id", "")).strip() for x in keyword_hits if x.get("memory_id")
        }
        event_keyword_mid_set = {
            str(x.get("memory_id", "")).strip()
            for x in event_log_keyword_hits
            if x.get("memory_id")
        }
        foresight_keyword_mid_set = {
            str(x.get("memory_id", "")).strip()
            for x in foresight_keyword_hits
            if x.get("memory_id")
        }
        for row in all_keyword_hits:
            mid = str(row.get("memory_id", "")).strip()
            row["in_keyword_hits"] = mid in keyword_mid_set
            row["in_event_log_keyword_hits"] = mid in event_keyword_mid_set
            row["in_foresight_keyword_hits"] = mid in foresight_keyword_mid_set

        vector_channels_enabled = bool(policy.vector_enabled) and (
            bool(self.vector_store.enabled)
            or bool(
                self.event_log_vector_store
                and bool(getattr(self.event_log_vector_store, "enabled", False))
            )
        )
        run_vector_probe = (
            vector_channels_enabled
            and not (
                id_like_query
                and self._is_keyword_confident(all_keyword_hits, top)
            )
        )
        if run_vector_probe:
            vector: list[float] = []
            vector_errs: list[str] = []
            try:
                vector = self._fit_vector_dim(self._embed_query(query))
            except Exception as exc:
                vector_probe_error = True
                vector_errs.append(str(exc)[:180])
            if vector:
                if bool(self.vector_store.enabled):
                    try:
                        vec_started = time.perf_counter()
                        vector_hits = self._search_vector_with_time_constraints(
                            vector=vector,
                            top_k=vector_budget,
                            user_id=user_id,
                            group_id=group_id,
                            candidate_episode_ids=candidate_episode_ids,
                            as_of_ts=as_of_ts,
                            start_ts=start_ts,
                            end_ts=end_ts,
                        )
                        vec_ms = (time.perf_counter() - vec_started) * 1000.0
                        for row in vector_hits:
                            row["source"] = "vector"
                            row["in_vector_hits"] = True
                    except Exception as exc:
                        vector_probe_error = True
                        vector_errs.append(str(exc)[:180])
                        vector_hits = []
                if self.event_log_vector_store and bool(
                    getattr(self.event_log_vector_store, "enabled", False)
                ):
                    try:
                        event_vec_started = time.perf_counter()
                        event_log_vector_hits = self._search_event_log_vector_with_time_constraints(
                            vector=vector,
                            top_k=vector_budget,
                            user_id=user_id,
                            group_id=group_id,
                            candidate_episode_ids=candidate_episode_ids,
                            as_of_ts=as_of_ts,
                            start_ts=start_ts,
                            end_ts=end_ts,
                        )
                        event_vec_ms = (time.perf_counter() - event_vec_started) * 1000.0
                        for row in event_log_vector_hits:
                            row["source"] = "event_log_vector"
                            row["in_event_log_vector_hits"] = True
                    except Exception as exc:
                        vector_probe_error = True
                        vector_errs.append(str(exc)[:180])
                        event_log_vector_hits = []
            if vector_errs:
                vector_probe_error_reason = " | ".join(x for x in vector_errs if x)[:180]

        vector_hits = self._dedupe_hit_rows_by_memory(vector_hits)
        event_log_vector_hits = self._dedupe_hit_rows_by_memory(event_log_vector_hits)
        all_vector_hits = self._dedupe_hit_rows_by_memory([*vector_hits, *event_log_vector_hits])
        vector_mid_set = {
            str(x.get("memory_id", "")).strip() for x in vector_hits if x.get("memory_id")
        }
        event_vector_mid_set = {
            str(x.get("memory_id", "")).strip()
            for x in event_log_vector_hits
            if x.get("memory_id")
        }
        for row in all_vector_hits:
            mid = str(row.get("memory_id", "")).strip()
            row["in_vector_hits"] = mid in vector_mid_set
            row["in_event_log_vector_hits"] = mid in event_vector_mid_set

        merge_strategy = "none"
        rrf_channels: list[list[dict[str, Any]]] = []
        if keyword_hits:
            rrf_channels.append(keyword_hits)
        if vector_hits:
            rrf_channels.append(vector_hits)
        if event_log_keyword_hits:
            rrf_channels.append(event_log_keyword_hits)
        if foresight_keyword_hits:
            rrf_channels.append(foresight_keyword_hits)
        if event_log_vector_hits:
            rrf_channels.append(event_log_vector_hits)

        if all_keyword_hits and all_vector_hits:
            prefer_vector = self._should_prefer_vector_for_semantic(
                query=query,
                id_like_query=id_like_query,
                keyword_hits=all_keyword_hits,
                vector_hits=all_vector_hits,
                top_k=top,
            )
            if prefer_vector:
                base_candidates = self._vector_priority_merge(
                    vector_hits=all_vector_hits,
                    keyword_hits=all_keyword_hits,
                    top_k=pool_top,
                )
                merge_strategy = "semantic_vector_priority"
            else:
                fused = reciprocal_rank_fusion(
                    rrf_channels, key="memory_id", rrf_k=policy.rrf_k
                )
                for row in fused:
                    mid = str(row.get("memory_id", "")).strip()
                    row["source"] = "hybrid_rrf"
                    row.setdefault(
                        "in_keyword_hits",
                        mid in keyword_mid_set,
                    )
                    row.setdefault(
                        "in_vector_hits",
                        mid in vector_mid_set,
                    )
                    row.setdefault(
                        "in_event_log_keyword_hits",
                        mid in event_keyword_mid_set,
                    )
                    row.setdefault(
                        "in_foresight_keyword_hits",
                        mid in foresight_keyword_mid_set,
                    )
                    row.setdefault(
                        "in_event_log_vector_hits",
                        mid in event_vector_mid_set,
                    )
                base_candidates = fused[:pool_top]
                merge_strategy = f"hybrid_rrf_{len(rrf_channels)}ch"
            hit_rows = self._rerank_basic_hit_candidates(
                candidates=base_candidates,
                keyword_hits=all_keyword_hits,
                vector_hits=all_vector_hits,
                top_k=pool_top,
                prefer_vector=prefer_vector,
            )
            merge_strategy = f"{merge_strategy}_rerank"
        elif all_keyword_hits:
            hit_rows = self._rerank_basic_hit_candidates(
                candidates=all_keyword_hits[:pool_top],
                keyword_hits=all_keyword_hits,
                vector_hits=[],
                top_k=pool_top,
                prefer_vector=False,
            )
            merge_strategy = "keyword_only_rerank"
        elif all_vector_hits:
            hit_rows = self._rerank_basic_hit_candidates(
                candidates=all_vector_hits[:pool_top],
                keyword_hits=[],
                vector_hits=all_vector_hits,
                top_k=pool_top,
                prefer_vector=True,
            )
            merge_strategy = "vector_only_rerank"
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
            base["in_event_log_keyword_hits"] = bool(
                hit.get("in_event_log_keyword_hits", False)
            )
            base["in_foresight_keyword_hits"] = bool(
                hit.get("in_foresight_keyword_hits", False)
            )
            base["in_event_log_vector_hits"] = bool(
                hit.get("in_event_log_vector_hits", False)
            )
            if hit.get("event_log_fact_hint"):
                base["event_log_fact_hint"] = str(hit.get("event_log_fact_hint"))[:240]
            if hit.get("foresight_text_hint"):
                base["foresight_text_hint"] = str(hit.get("foresight_text_hint"))[:240]
            rows.append(base)
        if rows:
            rows = self._rerank_basic_result_rows(query=query, rows=rows)
            rows = self._filter_rows_by_time_bounds(
                rows=rows,
                as_of_ts=as_of_ts,
                start_ts=start_ts,
                end_ts=end_ts,
            )

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
        self._attach_vector_probe_diag(
            rows=limited,
            has_error=vector_probe_error,
            error_reason=vector_probe_error_reason,
        )
        if not attach_foresight:
            self._emit_search_trace(
                "basic_search",
                query_len=len(str(query or "")),
                top_k=top,
                id_like=bool(id_like_query),
                candidate_count=0 if candidate_episode_ids is None else len(candidate_episode_ids),
                keyword_budget=keyword_budget,
                vector_budget=vector_budget,
                keyword_hits=len(all_keyword_hits),
                vector_hits=len(all_vector_hits),
                event_keyword_hits=len(event_log_keyword_hits),
                foresight_keyword_hits=len(foresight_keyword_hits),
                event_vector_hits=len(event_log_vector_hits),
                vector_probe_error=bool(vector_probe_error),
                vector_probe_error_reason=vector_probe_error_reason,
                merge_strategy=merge_strategy,
                output_count=len(limited),
                foresight_attached=False,
                ms_total=round((time.perf_counter() - started_at) * 1000.0, 3),
                ms_keyword=round(kw_ms, 3),
                ms_vector=round(vec_ms, 3),
                ms_event_keyword=round(event_kw_ms, 3),
                ms_event_vector=round(event_vec_ms, 3),
                ms_foresight_keyword=round(foresight_kw_ms, 3),
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
        self._attach_vector_probe_diag(
            rows=final_rows,
            has_error=vector_probe_error,
            error_reason=vector_probe_error_reason,
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
            keyword_hits=len(all_keyword_hits),
            vector_hits=len(all_vector_hits),
            event_keyword_hits=len(event_log_keyword_hits),
            foresight_keyword_hits=len(foresight_keyword_hits),
            event_vector_hits=len(event_log_vector_hits),
            vector_probe_error=bool(vector_probe_error),
            vector_probe_error_reason=vector_probe_error_reason,
            merge_strategy=merge_strategy,
            output_count=len(final_rows),
            foresight_attached=True,
            ms_total=round((time.perf_counter() - started_at) * 1000.0, 3),
            ms_keyword=round(kw_ms, 3),
            ms_vector=round(vec_ms, 3),
            ms_event_keyword=round(event_kw_ms, 3),
            ms_event_vector=round(event_vec_ms, 3),
            ms_foresight_keyword=round(foresight_kw_ms, 3),
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
        round_top_k = self._expand_agentic_round_top_k(top_k)
        # Round-1: full-scope retrieval with low overhead.
        if as_of_ts is None and start_ts is None and end_ts is None:
            hits = self._basic_search(
                policy,
                query,
                user_id,
                group_id,
                round_top_k,
                None,
                attach_foresight=False,
            )
        else:
            hits = self._basic_search(
                policy,
                query,
                user_id,
                group_id,
                round_top_k,
                None,
                attach_foresight=False,
                as_of_ts=as_of_ts,
                start_ts=start_ts,
                end_ts=end_ts,
            )
        vector_probe_error, vector_probe_error_reason = (
            self._extract_vector_probe_diag(hits)
        )
        sufficiency = self._assess_retrieval_sufficiency(query=query, hits=hits, top_k=top_k)
        if self._should_skip_agentic_second_round(
            query=query,
            hits=hits,
            top_k=top_k,
            sufficiency=sufficiency,
            vector_probe_error=vector_probe_error,
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
                vector_probe_error=bool(vector_probe_error),
                vector_probe_error_reason=vector_probe_error_reason,
                ms_total=round((time.perf_counter() - started_at) * 1000.0, 3),
            )
            return final_rows
        # Round-2: scene-guided candidate narrowing + rewrite only when needed.
        scene_started = time.perf_counter()
        scene = self._scene_guided_candidate_ids(query, user_id, group_id, top_scene_n=6)
        scene_ms = (time.perf_counter() - scene_started) * 1000.0
        candidate_ids = set(scene.get("episode_ids", []))
        graph_candidate_ids = self._graph_guided_candidate_ids(
            query=query,
            user_id=user_id,
            group_id=group_id,
            top_k=top_k,
            as_of_ts=as_of_ts,
            start_ts=start_ts,
            end_ts=end_ts,
        )
        if graph_candidate_ids:
            candidate_ids.update(graph_candidate_ids)
        rewrite_plan = self._build_agentic_rewrite_plan(
            query=query, hits=hits, sufficiency=sufficiency
        )
        rewritten = rewrite_plan.query
        llm_sufficient = bool(
            sufficiency.sufficient and str(sufficiency.source) == "llm_verifier"
        )
        if (
            not vector_probe_error
            and not candidate_ids
            and rewritten.strip() == query.strip()
            and llm_sufficient
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
                graph_candidate_count=len(graph_candidate_ids),
                output_count=len(final_rows),
                vector_probe_error=bool(vector_probe_error),
                vector_probe_error_reason=vector_probe_error_reason,
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
            row["agentic_graph_count"] = len(graph_candidate_ids)
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
                graph_candidate_count=len(graph_candidate_ids),
                output_count=len(final_rows),
                vector_probe_error=bool(vector_probe_error),
                vector_probe_error_reason=vector_probe_error_reason,
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
            graph_candidate_count=len(graph_candidate_ids),
            output_count=len(final_rows),
            rewrite_source=rewrite_plan.source,
            rewrite_confidence=round(float(rewrite_plan.confidence), 3),
            vector_probe_error=bool(vector_probe_error),
            vector_probe_error_reason=vector_probe_error_reason,
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
        if self._is_temporal_query(query) or self._is_precision_sensitive_query(query):
            return False
        if len(str(query or "").strip()) < 16:
            return False
        if not vector_hits:
            return False
        if self._is_keyword_confident(keyword_hits, min(max(1, int(top_k)), 2)):
            return False
        top_vector_score = max(float(x.get("score", 0.0)) for x in vector_hits)
        if top_vector_score < 0.22:
            return False
        probe_width = max(4, min(len(vector_hits), max(2, int(top_k)) * 2))
        keyword_probe = {
            str(x.get("memory_id"))
            for x in keyword_hits[:probe_width]
            if str(x.get("memory_id", "")).strip()
        }
        vector_probe = {
            str(x.get("memory_id"))
            for x in vector_hits[:probe_width]
            if str(x.get("memory_id", "")).strip()
        }
        has_channel_overlap = bool(keyword_probe & vector_probe)
        if not has_channel_overlap:
            return False
        min_keyword_hits = min(max(1, int(top_k)), 2)
        if len(keyword_hits) < min_keyword_hits:
            return has_channel_overlap
        top_keyword_score = max(float(x.get("score", 0.0)) for x in keyword_hits)
        # Prefer vector-only priority when lexical retrieval is weak.
        return top_keyword_score < 0.25

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

    @staticmethod
    def _dedupe_hit_rows_by_memory(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            memory_id = str(row.get("memory_id", "")).strip()
            if not memory_id:
                continue
            item = dict(row)
            item["memory_id"] = memory_id
            existing = merged.get(memory_id)
            if existing is None:
                merged[memory_id] = item
                continue
            score = float(item.get("score", 0.0))
            prev_score = float(existing.get("score", 0.0))
            if score > prev_score:
                preserved_flags = {
                    k: v
                    for k, v in existing.items()
                    if str(k).startswith("in_") and isinstance(v, bool) and bool(v)
                }
                item.update(preserved_flags)
                merged[memory_id] = item
                existing = item
            for key, value in item.items():
                if str(key).startswith("in_") and isinstance(value, bool) and value:
                    existing[key] = True
            support = int(existing.get("event_log_support_count", 0))
            existing["event_log_support_count"] = support + int(
                item.get("event_log_support_count", 0)
            )
            if not existing.get("event_log_fact_hint") and item.get("event_log_fact_hint"):
                existing["event_log_fact_hint"] = item.get("event_log_fact_hint")
            foresight_support = int(existing.get("foresight_support_count", 0))
            existing["foresight_support_count"] = foresight_support + int(
                item.get("foresight_support_count", 0)
            )
            if not existing.get("foresight_text_hint") and item.get("foresight_text_hint"):
                existing["foresight_text_hint"] = item.get("foresight_text_hint")
        out = list(merged.values())
        out.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        return out

    @staticmethod
    def _rank_to_unit_score(rank: int | None, total: int) -> float:
        if rank is None or rank <= 0 or total <= 0:
            return 0.0
        return float(total - rank + 1) / float(total)

    def _rerank_basic_hit_candidates(
        self,
        *,
        candidates: list[dict[str, Any]],
        keyword_hits: list[dict[str, Any]],
        vector_hits: list[dict[str, Any]],
        top_k: int,
        prefer_vector: bool,
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []
        kw_rank = {
            str(row.get("memory_id")): idx
            for idx, row in enumerate(keyword_hits, start=1)
            if row.get("memory_id")
        }
        vec_rank = {
            str(row.get("memory_id")): idx
            for idx, row in enumerate(vector_hits, start=1)
            if row.get("memory_id")
        }
        total_kw = max(1, len(keyword_hits))
        total_vec = max(1, len(vector_hits))
        total_candidates = max(1, len(candidates))
        weighted: list[dict[str, Any]] = []
        for idx, row in enumerate(candidates, start=1):
            mid = str(row.get("memory_id", "")).strip()
            if not mid:
                continue
            item = dict(row)
            kw_score = self._rank_to_unit_score(kw_rank.get(mid), total_kw)
            vec_score = self._rank_to_unit_score(vec_rank.get(mid), total_vec)
            base_score = self._rank_to_unit_score(idx, total_candidates)
            dual_score = 1.0 if (mid in kw_rank and mid in vec_rank) else 0.0
            if prefer_vector:
                channel_score = (
                    0.56 * vec_score
                    + 0.24 * kw_score
                    + 0.12 * base_score
                    + 0.08 * dual_score
                )
            else:
                channel_score = (
                    0.44 * vec_score
                    + 0.34 * kw_score
                    + 0.14 * base_score
                    + 0.08 * dual_score
                )
            item["retrieval_channel_score"] = round(float(channel_score), 4)
            item["score"] = float(channel_score)
            item["in_keyword_hits"] = bool(item.get("in_keyword_hits", mid in kw_rank))
            item["in_vector_hits"] = bool(item.get("in_vector_hits", mid in vec_rank))
            weighted.append(item)
        weighted.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        return weighted[: max(1, int(top_k))]

    def _rerank_basic_result_rows(self, *, query: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not rows:
            return rows
        weighted: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            text = " ".join(
                [
                    str(item.get("summary", "")),
                    str(item.get("episode", "")),
                    str(item.get("subject", "")),
                    str(item.get("event_log_fact_hint", "")),
                    str(item.get("foresight_text_hint", "")),
                ]
            ).strip()
            overlap = self._lexical_overlap_ratio(query, text) if text else 0.0
            has_keyword = bool(item.get("in_keyword_hits", False)) or bool(
                item.get("in_event_log_keyword_hits", False)
            ) or bool(item.get("in_foresight_keyword_hits", False))
            has_vector = bool(item.get("in_vector_hits", False)) or bool(
                item.get("in_event_log_vector_hits", False)
            )
            dual_bonus = (
                1.0
                if has_keyword and has_vector
                else 0.0
            )
            base = float(item.get("score", 0.0))
            item["query_overlap_bonus"] = round(float(overlap), 4)
            item["score"] = (
                float(base)
                + 0.16 * float(overlap)
                + 0.04 * float(dual_bonus)
            )
            weighted.append(item)
        weighted.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        weighted = self._apply_model_rerank(query=query, rows=weighted)
        if self.phase4_reasoning_enabled and self._is_temporal_query(query):
            return self._rerank_temporal_rows(query=query, rows=weighted)
        return weighted

    def _apply_model_rerank(self, *, query: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        provider = self.rerank_provider
        if provider is None:
            return rows
        if len(rows) < self.rerank_trigger_k:
            return rows
        top_n = min(len(rows), self.rerank_top_n)
        subset = [dict(x) for x in rows[:top_n]]
        docs = [
            " ".join(
                [
                    str(item.get("summary", "")),
                    str(item.get("episode", "")),
                    str(item.get("subject", "")),
                ]
            ).strip()
            for item in subset
        ]
        start = time.perf_counter()
        try:
            scores = provider.rerank(query=query, documents=docs)
        except Exception:
            return rows
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        if elapsed_ms > self.rerank_timeout_ms:
            return rows
        if not isinstance(scores, list):
            return rows
        raw_scores: list[float] = []
        for idx in range(len(subset)):
            try:
                raw_scores.append(float(scores[idx]) if idx < len(scores) else 0.0)
            except Exception:
                raw_scores.append(0.0)
        min_score = min(raw_scores) if raw_scores else 0.0
        max_score = max(raw_scores) if raw_scores else 0.0
        span = max_score - min_score
        for idx, item in enumerate(subset):
            rerank_score = raw_scores[idx] if idx < len(raw_scores) else 0.0
            if span > 1e-9:
                rerank_norm = (rerank_score - min_score) / span
            else:
                rerank_norm = 0.5
            base_rank_norm = float(top_n - idx) / float(max(1, top_n))
            item["rerank_model_score"] = round(rerank_score, 4)
            item["rerank_model_score_norm"] = round(rerank_norm, 4)
            item["rerank_applied"] = True
            item["rerank_latency_ms"] = elapsed_ms
            item["score"] = 0.20 * float(base_rank_norm) + 0.80 * float(rerank_norm)
            subset[idx] = item
        subset.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        return subset + rows[top_n:]

    def _should_skip_agentic_second_round(
        self,
        *,
        query: str,
        hits: list[dict[str, Any]],
        top_k: int,
        sufficiency: RetrievalSufficiency | None = None,
        vector_probe_error: bool = False,
    ) -> bool:
        if vector_probe_error:
            return False
        if self.agentic_force_second_round and not self._is_id_like_query(query):
            return False
        if self._is_id_like_query(query):
            if sufficiency is not None:
                return bool(sufficiency.sufficient)
            return self._is_sufficient(hits, top_k)
        if not hits:
            return False
        usable_hits = [
            h for h in hits if str(h.get("source", "")).strip().lower() != "fallback_recent"
        ]
        if not usable_hits:
            return False
        ranked = sorted((float(h.get("score", 0.0)) for h in usable_hits), reverse=True)
        best = ranked[0] if ranked else 0.0
        second = ranked[1] if len(ranked) > 1 else 0.0
        complex_query = (
            self._is_multi_hop_query(query)
            or self._is_temporal_query(query)
            or self._is_precision_sensitive_query(query)
            or self._is_cross_event_query(query)
        )
        unique_sources = {
            str(h.get("source", "")).strip().lower() for h in usable_hits if h.get("source")
        }
        has_dual_hit = any(
            bool(h.get("in_keyword_hits", False)) and bool(h.get("in_vector_hits", False))
            for h in usable_hits
        )
        if sufficiency is not None and bool(sufficiency.sufficient):
            if sufficiency.source == "llm_verifier":
                if complex_query:
                    return (
                        best >= 0.72
                        and second >= 0.54
                        and len(usable_hits) >= max(3, min(5, top_k))
                        and (len(unique_sources) >= 2 or has_dual_hit)
                    )
                return best >= 0.56 and second >= 0.38
            if complex_query:
                return (
                    best >= 0.64
                    and second >= 0.46
                    and len(usable_hits) >= max(3, min(4, top_k))
                    and (len(unique_sources) >= 2 or has_dual_hit)
                )
            return best >= 0.50 and second >= 0.34
        if complex_query:
            return (
                best >= 0.70
                and second >= 0.50
                and len(usable_hits) >= max(3, min(4, top_k))
                and (len(unique_sources) >= 2 or has_dual_hit)
            )
        if len(usable_hits) == 1:
            only_src = str(usable_hits[0].get("source", "")).strip().lower()
            if only_src == "vector" and best >= 0.62:
                return True
            return best >= 0.64
        if best >= 0.45 and second >= 0.30:
            return True
        if best >= 0.38 and len(usable_hits) >= max(2, min(3, top_k)):
            if len(unique_sources) >= 2 or has_dual_hit:
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
        round_top_k = self._expand_agentic_round_top_k(top_k)

        rounds_by_idx: dict[int, list[dict[str, Any]]] = {}
        if len(queries) <= 1:
            for idx, q in enumerate(queries):
                rows = self._call_basic_search_with_optional_as_of(
                    policy=policy,
                    query=q,
                    user_id=user_id,
                    group_id=group_id,
                    top_k=round_top_k,
                    candidate_episode_ids=candidate_episode_ids,
                    as_of_ts=as_of_ts,
                    start_ts=start_ts,
                    end_ts=end_ts,
                )
                rounds_by_idx[idx] = rows
        else:
            max_workers = max(1, min(4, len(queries)))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        self._call_basic_search_with_optional_as_of,
                        policy=policy,
                        query=q,
                        user_id=user_id,
                        group_id=group_id,
                        top_k=round_top_k,
                        candidate_episode_ids=candidate_episode_ids,
                        as_of_ts=as_of_ts,
                        start_ts=start_ts,
                        end_ts=end_ts,
                    ): idx
                    for idx, q in enumerate(queries)
                }
                for future, idx in [(f, futures[f]) for f in futures]:
                    try:
                        rounds_by_idx[idx] = future.result()
                    except Exception:
                        rounds_by_idx[idx] = []
        rounds = [rounds_by_idx.get(i, []) for i in range(len(queries))]
        if not rounds:
            return []
        if len(rounds) == 1:
            second = rounds[0]
        else:
            second = reciprocal_rank_fusion(rounds, key="id", rrf_k=policy.rrf_k)
            second = second[: max(1, int(round_top_k))]
        second = self._rerank_query_overlap_rows(rows=second, query_variants=queries)
        if self.phase4_reasoning_enabled and self._is_temporal_query(original_query):
            second = self._rerank_temporal_rows(query=original_query, rows=second)
        return second

    def _expand_agentic_round_top_k(self, top_k: int) -> int:
        base = max(1, int(top_k))
        if self.recall_mode:
            widened = max(base * 4, base + 24, self.agentic_round_min_k)
        else:
            widened = max(base * 3, base + 12, self.agentic_round_min_k)
        return max(base, min(self.agentic_round_max_k, widened))

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
        if len(queries) < max_queries:
            hyde_queries = self._expand_hyde_queries(
                query=seed or original_query,
                first_round_hits=first_round_hits,
                insufficiency_reason=insufficiency_reason,
                max_queries=max_queries - len(queries),
            )
            seen = {q.lower() for q in queries}
            for item in hyde_queries:
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
        if len(queries) < max_queries:
            extras = self._expand_temporal_boundary_queries(
                seed=seed,
                original_query=original_query,
                insufficiency_reason=insufficiency_reason,
            )
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

    def _expand_hyde_queries(
        self,
        *,
        query: str,
        first_round_hits: list[dict[str, Any]],
        insufficiency_reason: str,
        max_queries: int,
    ) -> list[str]:
        cap = max(0, min(2, int(max_queries)))
        if cap <= 0:
            return []
        rewriter = self.query_rewriter
        if rewriter is None or not hasattr(rewriter, "generate_hyde_documents"):
            return []
        try:
            decision = rewriter.generate_hyde_documents(  # type: ignore[attr-defined]
                query=query,
                hits=first_round_hits,
                insufficiency_reason=insufficiency_reason,
                max_docs=cap,
            )
        except Exception:
            decision = None
        if not isinstance(decision, HyDEDecision):
            return []
        if float(decision.confidence) < self.query_rewriter_min_confidence:
            return []
        out: list[str] = []
        seen: set[str] = set()
        for item in decision.documents:
            doc = " ".join(str(item or "").strip().split())[:280]
            if not doc:
                continue
            qx = f"{query} {doc}".strip()
            key = qx.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(qx)
            if len(out) >= cap:
                break
        return out

    def _expand_temporal_boundary_queries(
        self,
        *,
        seed: str,
        original_query: str,
        insufficiency_reason: str,
    ) -> list[str]:
        base = " ".join(str(seed or "").strip().split())
        if not base:
            return []
        q = str(original_query or "").strip().lower()
        reason = str(insufficiency_reason or "").strip().lower()
        extras: list[str] = []
        temporal_needed = self._is_temporal_query(original_query) or any(
            term in reason
            for term in (
                "time",
                "timeline",
                "date",
                "when",
                "时序",
                "时间",
                "先后",
            )
        )
        if temporal_needed:
            target_year = self._extract_target_year(original_query)
            if target_year is not None:
                extras.append(f"{base} {target_year} 时间线 关键事件")
            extras.append(f"{base} 时间 顺序 先后")
            if any(term in q for term in ("latest", "recent", "最近", "刚刚", "newest")):
                extras.append(f"{base} 最近 最新 进展")
            if any(term in q for term in ("earliest", "first", "最早", "起初")):
                extras.append(f"{base} 最早 起因 背景")
        if any(term in reason for term in ("who", "person", "entity", "人物", "角色", "参与")):
            extras.append(f"{base} 人物 参与者 关系")
        if any(term in reason for term in ("where", "location", "place", "地点", "场景")):
            extras.append(f"{base} 地点 场景 背景")
        if not extras and self._is_multi_hop_query(original_query):
            extras.append(f"{base} 关键细节 关联线索")
        dedup: list[str] = []
        seen: set[str] = set()
        for item in extras:
            qx = " ".join(str(item or "").strip().split())
            key = qx.lower()
            if not qx or key in seen:
                continue
            seen.add(key)
            dedup.append(qx)
        return dedup[:3]

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

    def _search_vector_with_time_constraints(
        self,
        *,
        vector: list[float],
        top_k: int,
        user_id: str | None,
        group_id: str | None,
        candidate_episode_ids: set[str] | None,
        as_of_ts: int | None,
        start_ts: int | None,
        end_ts: int | None,
    ) -> list[dict[str, Any]]:
        try:
            return self.vector_store.search(
                vector=vector,
                top_k=top_k,
                user_id=user_id,
                group_id=group_id,
                candidate_episode_ids=candidate_episode_ids,
                as_of_ts=as_of_ts,
                start_ts=start_ts,
                end_ts=end_ts,
            )
        except TypeError as exc:
            # Compatibility path for lightweight stubs that still use the old signature.
            msg = str(exc)
            if "unexpected keyword argument" not in msg:
                raise
            return self.vector_store.search(
                vector=vector,
                top_k=top_k,
                user_id=user_id,
                group_id=group_id,
                candidate_episode_ids=candidate_episode_ids,
            )

    def _search_event_log_vector_with_time_constraints(
        self,
        *,
        vector: list[float],
        top_k: int,
        user_id: str | None,
        group_id: str | None,
        candidate_episode_ids: set[str] | None,
        as_of_ts: int | None,
        start_ts: int | None,
        end_ts: int | None,
    ) -> list[dict[str, Any]]:
        store = self.event_log_vector_store
        if store is None:
            return []
        try:
            return store.search(
                vector=vector,
                top_k=top_k,
                user_id=user_id,
                group_id=group_id,
                candidate_episode_ids=candidate_episode_ids,
                as_of_ts=as_of_ts,
                start_ts=start_ts,
                end_ts=end_ts,
            )
        except TypeError as exc:
            msg = str(exc)
            if "unexpected keyword argument" not in msg:
                raise
            return store.search(
                vector=vector,
                top_k=top_k,
                user_id=user_id,
                group_id=group_id,
                candidate_episode_ids=candidate_episode_ids,
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

    def _resolve_query_time_constraints(
        self,
        *,
        query: str,
        as_of_ts: int | None,
        start_ts: int | None,
        end_ts: int | None,
    ) -> tuple[int | None, int | None, int | None, str]:
        explicit_as_of = int(as_of_ts) if as_of_ts is not None else None
        explicit_start = int(start_ts) if start_ts is not None else None
        explicit_end = int(end_ts) if end_ts is not None else None

        if explicit_as_of is not None and explicit_as_of > 0:
            return explicit_as_of, None, None, "explicit_as_of"
        if explicit_start is not None and explicit_start <= 0:
            explicit_start = None
        if explicit_end is not None and explicit_end <= 0:
            explicit_end = None
        if explicit_start is not None or explicit_end is not None:
            if (
                explicit_start is not None
                and explicit_end is not None
                and explicit_start > explicit_end
            ):
                explicit_start, explicit_end = explicit_end, explicit_start
            return None, explicit_start, explicit_end, "explicit_range"

        derived_as_of, derived_start, derived_end, reason = (
            self._derive_query_time_bounds(query=query)
        )
        return derived_as_of, derived_start, derived_end, reason

    def _derive_query_time_bounds(
        self, *, query: str
    ) -> tuple[int | None, int | None, int | None, str]:
        q = str(query or "").strip()
        if not q:
            return None, None, None, "none"
        ql = q.lower()

        year_range_patterns = (
            r"((?:19|20)\d{2})\s*年?\s*(?:-|~|～|到|至|to)\s*((?:19|20)\d{2})\s*年?",
            r"\bbetween\s+((?:19|20)\d{2})\s+(?:and|to)\s+((?:19|20)\d{2})\b",
            r"\bfrom\s+((?:19|20)\d{2})\s+(?:to|until)\s+((?:19|20)\d{2})\b",
        )
        for pattern in year_range_patterns:
            match = re.search(pattern, ql)
            if not match:
                continue
            y1 = int(match.group(1))
            y2 = int(match.group(2))
            lo, hi = (y1, y2) if y1 <= y2 else (y2, y1)
            start, _ = self._year_bounds(lo)
            _, end = self._year_bounds(hi)
            return None, start, end, "query_year_range"

        match = re.search(
            r"(?:before|until|till)\s+((?:19|20)\d{2})\b|((?:19|20)\d{2})\s*年?\s*(?:前|以前|之前)",
            ql,
        )
        if match:
            year = int(match.group(1) or match.group(2))
            _, end = self._year_bounds(year)
            return None, None, end, "query_before_year"

        match = re.search(
            r"(?:after|since|from)\s+((?:19|20)\d{2})\b|((?:19|20)\d{2})\s*年?\s*(?:后|以后|之后)",
            ql,
        )
        if match:
            year = int(match.group(1) or match.group(2))
            start, _ = self._year_bounds(year)
            return None, start, None, "query_after_year"

        now_local = time.localtime()
        this_year = int(now_local.tm_year)
        this_month = int(now_local.tm_mon)

        if any(term in ql for term in ("去年", "last year", "previous year")):
            start, end = self._year_bounds(this_year - 1)
            return None, start, end, "query_relative_year"
        if any(term in ql for term in ("今年", "this year")):
            start, end = self._year_bounds(this_year)
            return None, start, end, "query_relative_year"
        if any(term in ql for term in ("明年", "next year")):
            start, end = self._year_bounds(this_year + 1)
            return None, start, end, "query_relative_year"

        if any(term in ql for term in ("上个月", "last month", "previous month")):
            if this_month == 1:
                year = this_year - 1
                month = 12
            else:
                year = this_year
                month = this_month - 1
            start, end = self._month_bounds(year, month)
            return None, start, end, "query_relative_month"
        if any(term in ql for term in ("这个月", "本月", "this month")):
            start, end = self._month_bounds(this_year, this_month)
            return None, start, end, "query_relative_month"
        if any(term in ql for term in ("下个月", "next month")):
            if this_month == 12:
                year = this_year + 1
                month = 1
            else:
                year = this_year
                month = this_month + 1
            start, end = self._month_bounds(year, month)
            return None, start, end, "query_relative_month"

        years = re.findall(r"(?<!\d)((?:19|20)\d{2})(?!\d)", ql)
        if len(years) == 1:
            year = int(years[0])
            start, end = self._year_bounds(year)
            return None, start, end, "query_single_year"
        return None, None, None, "none"

    @staticmethod
    def _year_bounds(year: int) -> tuple[int, int]:
        y = int(year)
        start = int(time.mktime((y, 1, 1, 0, 0, 0, 0, 0, -1)))
        end = int(time.mktime((y, 12, 31, 23, 59, 59, 0, 0, -1)))
        return start, end

    @staticmethod
    def _month_bounds(year: int, month: int) -> tuple[int, int]:
        y = int(year)
        m = int(month)
        if m < 1:
            y -= (abs(m) // 12) + 1
            m = 12 - (abs(m) % 12)
        elif m > 12:
            y += (m - 1) // 12
            m = ((m - 1) % 12) + 1
        start = int(time.mktime((y, m, 1, 0, 0, 0, 0, 0, -1)))
        if m == 12:
            next_y, next_m = y + 1, 1
        else:
            next_y, next_m = y, m + 1
        next_month_start = int(time.mktime((next_y, next_m, 1, 0, 0, 0, 0, 0, -1)))
        return start, next_month_start - 1

    @staticmethod
    def _filter_rows_by_time_bounds(
        *,
        rows: list[dict[str, Any]],
        as_of_ts: int | None = None,
        start_ts: int | None = None,
        end_ts: int | None = None,
    ) -> list[dict[str, Any]]:
        if not rows:
            return []
        as_of = int(as_of_ts) if as_of_ts is not None else None
        start = int(start_ts) if start_ts is not None else None
        end = int(end_ts) if end_ts is not None else None
        if as_of is not None and as_of <= 0:
            as_of = None
        if start is not None and start <= 0:
            start = None
        if end is not None and end <= 0:
            end = None
        if as_of is not None:
            start = None
            end = None
        elif start is not None and end is not None and start > end:
            start, end = end, start
        if as_of is None and start is None and end is None:
            return rows

        filtered: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            try:
                ts = int(row.get("timestamp") or 0)
            except Exception:
                ts = 0
            if ts <= 0:
                continue
            if as_of is not None and ts > as_of:
                continue
            if start is not None and ts < start:
                continue
            if end is not None and ts > end:
                continue
            filtered.append(row)
        return filtered

    @staticmethod
    def _is_precision_sensitive_query(query: str) -> bool:
        q = str(query or "").strip().lower()
        if not q:
            return False
        english_patterns = (
            r"\bwhen\b",
            r"\bwhat\s+time\b",
            r"\bhow\s+many\b",
            r"\bhow\s+long\b",
            r"\bwhy\b",
            r"\breason\b",
            r"\bcause\b",
            r"\bduration\b",
            r"\bcount\b",
            r"\bnumber\s+of\b",
            r"\badvice\b",
            r"\bsuggest(?:ion|ed)?\b",
            r"\bshould\b",
        )
        if any(re.search(pattern, q) for pattern in english_patterns):
            return True
        chinese_terms = (
            "什么时候",
            "几点",
            "何时",
            "多久",
            "多长时间",
            "多少次",
            "多少",
            "原因",
            "为什么",
            "建议",
            "应该",
            "时长",
            "持续",
            "频率",
        )
        return any(term in q for term in chinese_terms)

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
        fetch_limit = max(1, int(top_k))
        fetch_limit = max(fetch_limit, min(200, max(fetch_limit * 4, fetch_limit + 6)))
        rows = self.repo.fetch_episodes(
            user_id=user_id,
            group_id=group_id,
            limit=fetch_limit,
            candidate_episode_ids=candidate_episode_ids,
            as_of_ts=as_of_ts,
            start_ts=start_ts,
            end_ts=end_ts,
        )
        rows = self._filter_rows_by_time_bounds(
            rows=rows,
            as_of_ts=as_of_ts,
            start_ts=start_ts,
            end_ts=end_ts,
        )
        rows = rows[: max(1, int(top_k))]
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
            q_vec = _safe_vector(self._fit_vector_dim(self._embed_query(query)))
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

    def _graph_guided_candidate_ids(
        self,
        *,
        query: str,
        user_id: str | None,
        group_id: str | None,
        top_k: int,
        as_of_ts: int | None,
        start_ts: int | None,
        end_ts: int | None,
    ) -> set[str]:
        if not self.graph_store.enabled:
            return set()
        if not self._is_graph_query_eligible(query):
            return set()
        graph_probe_limit = max(
            12,
            min(180, max(int(top_k) * 10, int(self.graph_top_k) * 8)),
        )
        try:
            graph_rows = self.graph_store.search(  # type: ignore[call-arg]
                query=query,
                user_id=user_id,
                group_id=group_id,
                limit=graph_probe_limit,
                as_of_ts=as_of_ts,
                start_ts=start_ts,
                end_ts=end_ts,
            )
        except TypeError as exc:
            if "unexpected keyword argument" not in str(exc):
                return set()
            try:
                graph_rows = self.graph_store.search(
                    query=query,
                    user_id=user_id,
                    group_id=group_id,
                    limit=graph_probe_limit,
                )
            except Exception:
                return set()
        except Exception:
            return set()
        if not graph_rows:
            return set()

        loose_threshold = (
            self._is_temporal_query(query)
            or self._is_multi_hop_query(query)
            or self._is_cross_event_query(query)
        )
        score_threshold = 0.34 if loose_threshold else 0.46
        event_ids: list[str] = []
        for row in graph_rows:
            event_id = str(row.get("event_id", "")).strip()
            if not event_id:
                continue
            confidence = max(0.0, min(1.0, float(row.get("confidence", 0.5))))
            match_score = max(0.0, float(row.get("match_score", 0.0)))
            blended = 0.62 * match_score + 0.38 * confidence
            if blended < score_threshold:
                continue
            event_ids.append(event_id)
            if len(event_ids) >= 120:
                break
        if not event_ids:
            return set()

        episode_ids = self.repo.get_episode_ids_by_event_ids(
            event_ids=event_ids,
            user_id=user_id,
            group_id=group_id,
            limit=max(80, min(500, max(int(top_k) * 30, len(event_ids) * 6))),
            as_of_ts=as_of_ts,
            start_ts=start_ts,
            end_ts=end_ts,
        )
        return {str(eid).strip() for eid in episode_ids if str(eid).strip()}

    def _is_graph_query_eligible(self, query: str) -> bool:
        q = str(query or "").strip()
        if len(q) < 4:
            return False
        ql = q.lower()
        relation_query = self._is_person_relation_query(q)
        score = 0.0
        if self._is_temporal_query(q):
            score += 1.2
        if self._is_multi_hop_query(q):
            score += 1.2
        if self._is_cross_event_query(q):
            score += 1.2
        if relation_query:
            score += 1.0
            if re.search(r"[?？]|谁|关系|between|relation|relationship", ql):
                score += 0.3
        if self._is_precision_sensitive_query(q):
            score += 0.35
        if len(q) >= 16:
            score += 0.15
        return score >= 1.15

    @staticmethod
    def _is_person_relation_query(query: str) -> bool:
        q = str(query or "").strip().lower()
        if not q:
            return False
        relation_terms = (
            "关系",
            "人物",
            "参与者",
            "谁",
            "儿子",
            "女儿",
            "父亲",
            "母亲",
            "家人",
            "同事",
            "老板",
            "朋友",
            "团队",
            "合作",
            "who",
            "whose",
            "relation",
            "relationship",
            "related",
            "parent",
            "child",
            "son",
            "daughter",
            "spouse",
            "wife",
            "husband",
            "colleague",
            "teammate",
        )
        if any(term in q for term in relation_terms):
            return True
        patterns = (
            r"[\u4e00-\u9fff]{1,16}(和|与)[\u4e00-\u9fff]{1,16}.*关系",
            r"\bbetween\b.{1,48}\band\b.{1,48}\b(?:relation|relationship)\b",
        )
        return any(re.search(p, q, flags=re.IGNORECASE) for p in patterns)

    @staticmethod
    def _is_cross_event_query(query: str) -> bool:
        q = str(query or "").strip().lower()
        if not q:
            return False
        terms = (
            "跨事件",
            "前后",
            "之前",
            "之后",
            "演进",
            "变化",
            "先后",
            "时间线",
            "对比",
            "关联事件",
            "across events",
            "cross event",
            "before and after",
            "timeline",
            "progression",
            "evolution",
            "compare",
            "correlation",
        )
        if any(term in q for term in terms):
            return True
        patterns = (
            r"从.{1,24}到.{1,24}",
            r"\bfrom\b.{1,48}\bto\b.{1,48}",
            r"\bbefore\b.{1,48}\bafter\b",
        )
        return any(re.search(p, q, flags=re.IGNORECASE) for p in patterns)

    @staticmethod
    def _attach_vector_probe_diag(
        *,
        rows: list[dict[str, Any]],
        has_error: bool,
        error_reason: str,
    ) -> None:
        if not rows:
            return
        reason = str(error_reason or "").strip()[:180]
        for row in rows:
            row["_vector_probe_error"] = bool(has_error)
            if has_error and reason:
                row["_vector_probe_error_reason"] = reason

    @staticmethod
    def _extract_vector_probe_diag(rows: list[dict[str, Any]]) -> tuple[bool, str]:
        for row in rows:
            if not isinstance(row, dict):
                continue
            if bool(row.get("_vector_probe_error", False)):
                reason = str(row.get("_vector_probe_error_reason", "")).strip()[:180]
                return True, reason
        return False, ""

    @staticmethod
    def _strip_internal_search_fields(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for row in rows:
            if not isinstance(row, dict):
                continue
            internal = [k for k in row.keys() if str(k).startswith("_")]
            for key in internal:
                row.pop(key, None)
        return rows

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
    def _decide_memory_category(
        *,
        episode: str,
        summary: str,
        subject: str,
        atomic_facts: list[str],
        profile_patch: dict[str, Any],
        foresights: list[dict[str, Any]],
        has_entity_relation: bool,
    ) -> str:
        if isinstance(profile_patch, dict) and any(str(k).strip() for k in profile_patch.keys()):
            return "profile"
        if any(str(item.get("content", "")).strip() for item in foresights if isinstance(item, dict)):
            return "plan"
        text = " ".join(
            [
                str(episode or ""),
                str(summary or ""),
                str(subject or ""),
                " ".join(str(x) for x in atomic_facts),
            ]
        ).lower()
        if re.search(r"(待办|任务|todo|deadline|截止|提醒|完成|进度)", text):
            return "task"
        if has_entity_relation or len([x for x in atomic_facts if str(x).strip()]) >= 2:
            return "knowledge"
        if re.search(r"(今天|昨天|上周|明天|会议|旅行|发生|event)", text):
            return "event"
        return "general"

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
        rrf_k: int = 60,
        as_of_ts: int | None = None,
        start_ts: int | None = None,
        end_ts: int | None = None,
    ) -> list[dict[str, Any]]:
        if not self.graph_store.enabled or not self._is_graph_query_eligible(query):
            return base_hits[: max(1, top_k)]
        try:
            graph_rows = self.graph_store.search(  # type: ignore[call-arg]
                query=query,
                user_id=user_id,
                group_id=group_id,
                limit=max(1, min(self.graph_top_k, top_k)),
                as_of_ts=as_of_ts,
                start_ts=start_ts,
                end_ts=end_ts,
            )
        except TypeError as exc:
            if "unexpected keyword argument" not in str(exc):
                raise
            try:
                graph_rows = self.graph_store.search(
                    query=query,
                    user_id=user_id,
                    group_id=group_id,
                    limit=max(1, min(self.graph_top_k, top_k)),
                )
            except Exception:
                graph_rows = []
        except Exception:
            graph_rows = []
        base_ranked = [dict(row) for row in base_hits[: max(1, int(top_k))]]
        existing_keys = {
            f"{str(row.get('subject', ''))}|{str(row.get('summary', ''))}"
            for row in base_ranked
        }
        loose_threshold = (
            self._is_temporal_query(query)
            or self._is_multi_hop_query(query)
            or self._is_cross_event_query(query)
        )
        score_threshold = 0.30 if loose_threshold else 0.40
        graph_ranked: list[dict[str, Any]] = []
        for g in graph_rows:
            summary = f"{g['subject']} -[{g['relation']}]-> {g['obj']}"
            dedup_key = f"{str(g.get('subject', ''))}|{summary}"
            if dedup_key in existing_keys:
                continue
            existing_keys.add(dedup_key)
            confidence = float(g.get("confidence", 0.5))
            match_score = float(g.get("match_score", 0.0))
            blended = 0.62 * match_score + 0.38 * confidence
            if blended < score_threshold:
                continue
            graph_score = (
                0.38
                + 0.27 * confidence
                + 0.35 * match_score
            )
            graph_importance = max(0.36, min(0.92, 0.30 + 0.35 * confidence + 0.35 * match_score))
            graph_ranked.append(
                {
                    "id": self._graph_memory_id(g),
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
                    "memory_category": "knowledge",
                }
            )
        if not graph_ranked:
            return base_ranked[: max(1, int(top_k))]

        for row in base_ranked:
            fusion_key = str(row.get("event_id") or row.get("id") or "")
            row["_fusion_key"] = fusion_key
        for row in graph_ranked:
            fusion_key = str(row.get("event_id") or row.get("id") or "")
            row["_fusion_key"] = fusion_key
        fused = reciprocal_rank_fusion(
            [graph_ranked, base_ranked],
            key="_fusion_key",
            rrf_k=max(10, int(rrf_k)),
        )
        for row in fused:
            row.pop("_fusion_key", None)
            if str(row.get("source", "")).strip():
                row["source"] = str(row["source"])
        return fused[: max(1, int(top_k))]

    @staticmethod
    def _graph_memory_id(graph_row: dict[str, Any]) -> str:
        seed = "|".join(
            [
                str(graph_row.get("event_id", "")),
                str(graph_row.get("subject", "")),
                str(graph_row.get("relation", "")),
                str(graph_row.get("obj", "")),
            ]
        )
        digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:20]
        return f"graph:{digest}"

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
        self,
        *,
        query: str,
        user_id: str | None,
        group_id: str | None,
        as_of_ts: int | None = None,
        start_ts: int | None = None,
        end_ts: int | None = None,
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
                    "memory_category": "profile",
                    "score": 0.72 + overlap,
                    "source": "profile_snapshot",
                }
            )
        rows = self._filter_rows_by_time_bounds(
            rows=rows,
            as_of_ts=as_of_ts,
            start_ts=start_ts,
            end_ts=end_ts,
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

    def _compress_retrieval_rows(
        self, *, query: str, rows: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        if not rows:
            return []
        out: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            item = dict(row)
            raw_episode = str(item.get("episode", "") or "").strip()
            raw_summary = str(item.get("summary", "") or "").strip()
            if raw_episode:
                try:
                    extracted = self._retrieval_compactor.extract(
                        content=raw_episode,
                        sender=str(item.get("subject", "") or "user"),
                        group_id=(
                            str(item.get("group_id"))
                            if item.get("group_id") is not None
                            else None
                        ),
                    )
                except Exception:
                    extracted = None
                if extracted is not None:
                    compact_summary = str(extracted.summary or "").strip()
                    compact_episode = str(extracted.episode or "").strip()
                    if compact_summary:
                        item["summary"] = compact_summary
                    elif raw_summary:
                        item["summary"] = raw_summary
                    if compact_episode:
                        item["episode"] = self._compact_episode_for_query(
                            query=query,
                            episode=compact_episode,
                        )
                    elif raw_episode:
                        item["episode"] = self._compact_episode_for_query(
                            query=query,
                            episode=raw_episode,
                        )
                    if (
                        not str(item.get("atomic_fact_text", "") or "").strip()
                        and extracted.atomic_facts
                    ):
                        item["atomic_fact_text"] = " ".join(
                            str(x).strip() for x in extracted.atomic_facts[:6] if str(x).strip()
                        )
                else:
                    item["episode"] = self._compact_episode_for_query(
                        query=query,
                        episode=raw_episode,
                    )
                    if raw_summary:
                        item["summary"] = raw_summary
            elif raw_summary:
                item["summary"] = raw_summary
                item["episode"] = raw_summary[:220]
            out.append(item)
        return out

    def _dedup_and_density_merge_rows(
        self, *, query: str, rows: list[dict[str, Any]], top_k: int
    ) -> list[dict[str, Any]]:
        if not rows:
            return []
        ranked = sorted(
            [dict(r) for r in rows if isinstance(r, dict)],
            key=lambda x: float(x.get("score", 0.0)),
            reverse=True,
        )
        merged: list[dict[str, Any]] = []
        for row in ranked:
            hit_idx = -1
            for idx, existing in enumerate(merged):
                if self._is_same_memory_cluster(existing, row):
                    hit_idx = idx
                    break
            if hit_idx < 0:
                merged.append(row)
            else:
                merged[hit_idx] = self._merge_memory_rows_for_density(
                    query=query,
                    primary=merged[hit_idx],
                    secondary=row,
                )
        merged.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        return merged[: max(1, int(top_k))]

    def _is_same_memory_cluster(self, left: dict[str, Any], right: dict[str, Any]) -> bool:
        left_id = str(left.get("id", "")).strip()
        right_id = str(right.get("id", "")).strip()
        if left_id and right_id and left_id == right_id:
            return True
        left_event = str(left.get("event_id", "")).strip()
        right_event = str(right.get("event_id", "")).strip()
        if left_event and right_event and left_event == right_event:
            return True
        left_text = " ".join(
            [
                str(left.get("summary", "")),
                str(left.get("episode", "")),
            ]
        ).strip()
        right_text = " ".join(
            [
                str(right.get("summary", "")),
                str(right.get("episode", "")),
            ]
        ).strip()
        if not left_text or not right_text:
            return False
        overlap = self._token_overlap(
            self._tokenize_search_text(left_text),
            self._tokenize_search_text(right_text),
        )
        try:
            left_ts = int(left.get("timestamp") or 0)
        except Exception:
            left_ts = 0
        try:
            right_ts = int(right.get("timestamp") or 0)
        except Exception:
            right_ts = 0
        close_in_time = (
            left_ts > 0 and right_ts > 0 and abs(left_ts - right_ts) <= 300
        )
        return bool(overlap >= 0.86 or (overlap >= 0.76 and close_in_time))

    def _merge_memory_rows_for_density(
        self,
        *,
        query: str,
        primary: dict[str, Any],
        secondary: dict[str, Any],
    ) -> dict[str, Any]:
        p_score = float(primary.get("score", 0.0))
        s_score = float(secondary.get("score", 0.0))
        base = dict(primary if p_score >= s_score else secondary)
        peer = dict(secondary if p_score >= s_score else primary)
        p_summary = str(base.get("summary", "")).strip()
        s_summary = str(peer.get("summary", "")).strip()
        if p_summary and s_summary and s_summary not in p_summary:
            base["summary"] = f"{p_summary} | {s_summary}"[:240]
        elif not p_summary and s_summary:
            base["summary"] = s_summary[:240]

        base_episode = str(base.get("episode", "")).strip()
        peer_episode = str(peer.get("episode", "")).strip()
        dense_episode = self._combine_dense_episode_text(
            query=query,
            first=base_episode,
            second=peer_episode,
            max_chars=560,
        )
        if dense_episode:
            base["episode"] = dense_episode

        base["score"] = max(p_score, s_score) + 0.01
        try:
            base_ts = int(base.get("timestamp") or 0)
        except Exception:
            base_ts = 0
        try:
            peer_ts = int(peer.get("timestamp") or 0)
        except Exception:
            peer_ts = 0
        if base_ts > 0 and peer_ts > 0:
            base["timestamp"] = max(base_ts, peer_ts)

        sources = []
        for src in [
            str(base.get("source", "")).strip(),
            str(peer.get("source", "")).strip(),
        ]:
            if not src:
                continue
            sources.extend([x.strip() for x in src.split(",") if x.strip()])
        if sources:
            base["source"] = ",".join(sorted(dict.fromkeys(sources)))
        return base

    def _combine_dense_episode_text(
        self, *, query: str, first: str, second: str, max_chars: int
    ) -> str:
        chunks: list[str] = []
        for raw in [first, second]:
            text = " ".join(str(raw or "").replace("\r\n", "\n").split())
            if not text:
                continue
            for part in re.split(r"[。！？!?；;\n]+", text):
                seg = str(part or "").strip(" ，,。;；")
                if not seg:
                    continue
                if any(
                    self._token_overlap(
                        self._tokenize_search_text(seg),
                        self._tokenize_search_text(existing),
                    )
                    >= 0.92
                    for existing in chunks
                ):
                    continue
                chunks.append(seg)
        if not chunks:
            return ""
        joined = "；".join(chunks)
        return self._compact_episode_for_query(
            query=query,
            episode=joined,
            max_chars=max_chars,
        )

    def _compact_episode_for_query(self, *, query: str, episode: str, max_chars: int = 420) -> str:
        text = " ".join(str(episode or "").replace("\r\n", "\n").split())
        if not text:
            return ""
        limit = max(120, int(max_chars))
        if len(text) <= limit:
            return text
        q = str(query or "").strip()
        if q:
            idx = text.lower().find(q.lower())
            if idx >= 0:
                half = max(48, limit // 2)
                start = max(0, idx - half)
                end = min(len(text), start + limit)
                return text[start:end]
        chunks = [text[i : i + limit] for i in range(0, len(text), limit)]
        if not chunks:
            return text[:limit]
        q_tokens = self._tokenize_search_text(q)
        if not q_tokens:
            return chunks[0]
        best = chunks[0]
        best_score = -1.0
        for chunk in chunks:
            score = self._token_overlap(q_tokens, self._tokenize_search_text(chunk))
            if score > best_score:
                best = chunk
                best_score = score
        return best


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
