from __future__ import annotations

import json
import math
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Protocol

from evermemos_lite.domain.policy import EffectivePolicy
from evermemos_lite.domain.retrieval.fusion import reciprocal_rank_fusion
from evermemos_lite.infra.graph.kuzu_store import KuzuGraphStore
from evermemos_lite.infra.sqlite.db import SQLiteEngine
from evermemos_lite.infra.sqlite.memory_repository import MemoryRepository
from evermemos_lite.infra.vector.lancedb_store import LanceVectorStore
from evermemos_lite.service.extractor import MemoryExtractor
from evermemos_lite.service.graph_extractor import extract_graph_triples
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


class MemoryService:
    def __init__(
        self,
        engine: SQLiteEngine,
        vector_store: LanceVectorStore,
        embedding_provider: EmbeddingProviderProtocol,
        extractor: MemoryExtractor,
        graph_store: KuzuGraphStore,
        graph_top_k: int = 3,
        graph_write_min_importance: float = 0.6,
        key_memory_importance_threshold: float = 0.5,
    ) -> None:
        self.engine = engine
        self.repo = MemoryRepository(engine)
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.extractor = extractor
        self.semantic_consolidator = SemanticConsolidator(
            self.repo, embedding_provider=embedding_provider
        )
        self.graph_store = graph_store
        self.graph_top_k = max(1, graph_top_k)
        self.graph_write_min_importance = max(0.0, min(1.0, graph_write_min_importance))
        self.key_memory_importance_threshold = max(
            0.0, min(1.0, float(key_memory_importance_threshold))
        )

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
        triples = extract_graph_triples(
            facts=list(extracted.atomic_facts) or [extracted.episode],
            user_id=payload.sender,
        )
        storage_tier = self._decide_storage_tier(
            importance=float(extracted.importance_score),
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
            importance_score=float(extracted.importance_score),
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
                importance_score=float(extracted.importance_score),
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
            "importance_score": extracted.importance_score,
            "storage_tier": storage_tier,
            "scene_id": episode.get("scene_id"),
        }

    def maybe_index_vector(self, policy: EffectivePolicy, episode: dict[str, Any]) -> None:
        if not policy.vector_enabled:
            return
        tier = str(episode.get("storage_tier") or "text_only")
        if not self._tier_supports_vector(tier):
            return
        importance = float(episode.get("importance_score", 0.0))
        min_key_threshold = max(
            float(policy.importance_threshold), self.key_memory_importance_threshold
        )
        if importance < min_key_threshold:
            return
        vector = self.embedding_provider.embed(str(episode.get("episode", "")))
        if len(vector) != self.vector_store.vector_dim:
            if len(vector) > self.vector_store.vector_dim:
                vector = vector[: self.vector_store.vector_dim]
            else:
                vector = vector + [0.0] * (self.vector_store.vector_dim - len(vector))
        vector_id = uuid.uuid4().hex
        now = int(time.time())
        self.engine.execute(
            """
            INSERT OR REPLACE INTO memory_vector
              (id, memory_type, memory_id, user_id, group_id, timestamp, importance_score, vector_dim, vector_dtype, model_name, created_at, updated_at)
            VALUES (?, 'episodic_memory', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                vector_id,
                episode["id"],
                episode.get("user_id"),
                episode.get("group_id"),
                int(episode.get("timestamp", now)),
                float(episode.get("importance_score", 0.0)),
                len(vector),
                "float32",
                str(getattr(self.embedding_provider, "model", "embedding-model")),
                now,
                now,
            ),
        )
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

    def fetch(self, user_id: str | None, group_id: str | None, limit: int) -> list[dict[str, Any]]:
        rows = self.repo.fetch_episodes(user_id=user_id, group_id=group_id, limit=max(1, limit))
        return self._attach_valid_foresight(rows=rows, user_id=user_id, group_id=group_id)

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
    ) -> list[dict[str, Any]]:
        if policy.agentic_enabled:
            base = self._agentic_search(policy, query, user_id, group_id, top_k)
        else:
            base = self._basic_search(policy, query, user_id, group_id, top_k, None)
        profile_hints = self._profile_hint_hits(query=query, user_id=user_id, group_id=group_id)
        merged_base = self._merge_profile_hits(base, profile_hints, top_k)
        return self._merge_graph_hits(query, user_id, group_id, merged_base, top_k)

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
    ) -> list[dict[str, Any]]:
        keyword_hits: list[dict[str, Any]] = []
        vector_hits: list[dict[str, Any]] = []

        if policy.keyword_enabled:
            keyword_hits = self.repo.search_keyword(
                query=query,
                user_id=user_id,
                group_id=group_id,
                top_k=max(1, min(200, policy.keyword_top_k)),
                candidate_episode_ids=candidate_episode_ids,
            )
            for row in keyword_hits:
                row["source"] = "keyword"
                row["in_keyword_hits"] = True

        if policy.vector_enabled and self.vector_store.enabled:
            try:
                vector = self.embedding_provider.embed(query)
                vector_hits = self.vector_store.search(
                    vector=vector,
                    top_k=max(1, min(200, policy.vector_top_k)),
                    user_id=user_id,
                    group_id=group_id,
                    candidate_episode_ids=candidate_episode_ids,
                )
                for row in vector_hits:
                    row["source"] = "vector"
                    row["in_vector_hits"] = True
            except Exception:
                vector_hits = []

        if keyword_hits and vector_hits:
            fused = reciprocal_rank_fusion([keyword_hits, vector_hits], key="memory_id", rrf_k=policy.rrf_k)
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
            hit_rows = fused[: max(1, top_k)]
        elif keyword_hits:
            hit_rows = keyword_hits[: max(1, top_k)]
        elif vector_hits:
            hit_rows = vector_hits[: max(1, top_k)]
        else:
            hit_rows = []

        episode_ids = [str(r.get("memory_id")) for r in hit_rows if r.get("memory_id")]
        hydrated = self._hydrate_vector_results(episode_ids)
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
            rows = self._fallback_recent(user_id, group_id, top_k, candidate_episode_ids)
        return self._attach_valid_foresight(rows=rows[: max(1, top_k)], user_id=user_id, group_id=group_id)

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
        auto_memory_saved = False
        auto_memory_error: str | None = None

        if segment and self._should_cut_segment(segment=segment, query=payload.user_text, now_ts=now_ts):
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
            "segment_turn_count": turn_count,
        }

    def _agentic_search(
        self,
        policy: EffectivePolicy,
        query: str,
        user_id: str | None,
        group_id: str | None,
        top_k: int,
    ) -> list[dict[str, Any]]:
        scene = self._scene_guided_candidate_ids(query, user_id, group_id, top_scene_n=8)
        candidate_ids = set(scene.get("episode_ids", []))
        hits = self._basic_search(policy, query, user_id, group_id, top_k, candidate_ids or None)
        self._annotate_scene_info(
            rows=hits,
            episode_scene_map=scene.get("episode_scene_map", {}),
            scene_score_map=scene.get("scene_score_map", {}),
        )
        if self._is_sufficient(hits, top_k):
            return hits

        rewritten = self._rewrite_query(query, hits)
        second = self._basic_search(policy, rewritten, user_id, group_id, top_k, candidate_ids or None)
        merged = reciprocal_rank_fusion([hits, second], key="id", rrf_k=policy.rrf_k)
        for row in merged:
            row.setdefault("source", "agentic_two_round")
            row["agentic_round"] = 2
            row["agentic_rewrite"] = rewritten
            row["agentic_scene_count"] = len(candidate_ids)
        if not merged:
            return self._fallback_recent(user_id, group_id, top_k, candidate_ids)
        return self._attach_valid_foresight(
            rows=merged[: max(1, top_k)], user_id=user_id, group_id=group_id
        )

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
        rows = self.repo.fetch_episodes(
            user_id=None,
            group_id=None,
            limit=max(1, len(unique_ids) * 2),
            candidate_episode_ids=set(unique_ids),
        )
        position = {eid: idx for idx, eid in enumerate(unique_ids)}
        rows.sort(key=lambda x: position.get(str(x.get("id")), 1_000_000))
        return rows

    def _fallback_recent(
        self,
        user_id: str | None,
        group_id: str | None,
        top_k: int,
        candidate_episode_ids: set[str] | None,
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
        return self._attach_valid_foresight(rows=rows, user_id=user_id, group_id=group_id)

    def _scene_guided_candidate_ids(
        self, query: str, user_id: str | None, group_id: str | None, top_scene_n: int
    ) -> dict[str, Any]:
        candidates = self.repo.get_memscene_candidates(
            user_id=user_id, group_id=group_id, limit=max(20, top_scene_n * 4)
        )
        try:
            q_vec = _safe_vector(self.embedding_provider.embed(query))
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
        is_key = float(importance) >= self.key_memory_importance_threshold
        if has_entity_relation and is_key:
            return "vector_graph"
        if has_entity_relation:
            return "graph_text"
        if is_key:
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
        seq = int(segment.get("segment_seq") or 1)
        now_ts = int(segment.get("last_time") or time.time())
        result = self.memorize(
            MemorizeInput(
                message_id=f"{conversation_id}-seg-{seq}",
                create_time=now_ts,
                sender=user_id,
                content=markdown,
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

    def _attach_valid_foresight(
        self, rows: list[dict[str, Any]], user_id: str | None, group_id: str | None
    ) -> list[dict[str, Any]]:
        ids = [str(r.get("id", "")) for r in rows if r.get("id")]
        foresight_map = self.repo.get_valid_foresights_for_episodes(
            episode_ids=ids, user_id=user_id, group_id=group_id
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
