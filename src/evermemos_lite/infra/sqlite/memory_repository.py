from __future__ import annotations

import json
import re
import sqlite3
import time
import uuid
from collections import defaultdict
from typing import Any

from evermemos_lite.infra.sqlite.db import SQLiteEngine

_CATEGORY_SET = {"general", "profile", "plan", "task", "event", "knowledge"}


class MemoryRepository:
    def __init__(self, engine: SQLiteEngine) -> None:
        self.engine = engine

    def save_message_as_memory(
        self,
        *,
        message_id: str,
        create_time: int,
        sender: str,
        content: str,
        user_id: str,
        group_id: str | None,
        group_name: str | None,
        sender_name: str | None,
        role: str,
        importance_score: float,
        storage_tier: str,
        summary: str,
        subject: str,
        atomic_facts: list[str],
        foresights: list[dict[str, Any]],
        profile_patch: dict[str, Any],
        memory_category: str = "general",
        event_id: str | None = None,
    ) -> dict[str, Any]:
        eid = event_id or uuid.uuid4().hex
        mid = uuid.uuid4().hex
        now = int(time.time())
        category = self._normalize_category(memory_category)
        tier = self._normalize_token_tag(storage_tier, fallback="text_only")

        self.engine.execute(
            """
            INSERT INTO episodic_memory(
                id,event_id,source_message_id,user_id,group_id,timestamp,role,sender,sender_name,
                group_name,episode,summary,subject,importance_score,scene_id,storage_tier,memory_category,created_at,updated_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                mid,
                eid,
                message_id,
                user_id,
                group_id,
                int(create_time),
                role,
                sender,
                sender_name,
                group_name,
                content,
                summary,
                subject,
                float(importance_score),
                None,
                tier,
                category,
                now,
                now,
            ),
        )

        self.engine.executemany(
            "INSERT INTO memory_fact(memory_id,fact) VALUES(?,?)",
            [(mid, fact.strip()) for fact in atomic_facts if fact.strip()],
        )
        self.engine.executemany(
            """
            INSERT INTO memory_foresight(memory_id,content,start_time,end_time,confidence)
            VALUES(?,?,?,?,?)
            """,
            [
                (
                    mid,
                    str(item.get("content", "")).strip(),
                    item.get("start_time"),
                    item.get("end_time"),
                    float(item.get("confidence", 0.5)),
                )
                for item in foresights
                if str(item.get("content", "")).strip()
            ],
        )
        self._sync_keyword_index(
            memory_id=mid,
            user_id=user_id,
            group_id=group_id,
            episode=content,
            summary=summary,
            subject=subject,
            atomic_facts=atomic_facts,
            memory_category=category,
            storage_tier=tier,
        )
        self._sync_foresight_keyword_index(memory_id=mid)
        if profile_patch:
            self.upsert_profile_snapshot(
                event_id=eid,
                user_id=user_id,
                group_id=group_id,
                profile_patch=profile_patch,
                timestamp=int(create_time),
            )

        return {
            "id": mid,
            "event_id": eid,
            "user_id": user_id,
            "group_id": group_id,
            "timestamp": int(create_time),
            "episode": content,
            "summary": summary,
            "subject": subject,
            "importance_score": float(importance_score),
            "scene_id": None,
            "storage_tier": tier,
            "memory_category": category,
        }

    def save_memcells(
        self,
        *,
        memory_id: str,
        event_id: str,
        user_id: str,
        group_id: str | None,
        memcells: list[str],
        created_at: int,
    ) -> int:
        rows: list[tuple[Any, ...]] = []
        for idx, item in enumerate(list(memcells or []), start=1):
            content = " ".join(str(item or "").split()).strip()
            if not content:
                continue
            rows.append(
                (
                    memory_id,
                    event_id,
                    user_id,
                    group_id,
                    int(idx),
                    content[:2000],
                    int(created_at),
                )
            )
            if len(rows) >= 48:
                break
        if not rows:
            return 0
        self.engine.executemany(
            """
            INSERT INTO memory_memcell(
              memory_id,event_id,user_id,group_id,cell_order,content,created_at
            ) VALUES(?,?,?,?,?,?,?)
            """,
            rows,
        )
        return len(rows)

    def save_event_logs(
        self,
        *,
        memory_id: str,
        event_id: str,
        user_id: str,
        group_id: str | None,
        event_logs: list[dict[str, Any]],
        created_at: int,
    ) -> int:
        rows: list[tuple[Any, ...]] = []
        seen: set[str] = set()
        for idx, item in enumerate(list(event_logs or []), start=1):
            if not isinstance(item, dict):
                continue
            fact = " ".join(str(item.get("fact", "")).split()).strip()
            fact_norm = str(item.get("fact_norm", "")).strip().lower()
            if not fact:
                continue
            if not fact_norm:
                fact_norm = re.sub(r"[\s\W_]+", "", fact.lower(), flags=re.UNICODE)
            if not fact_norm or fact_norm in seen:
                continue
            seen.add(fact_norm)
            fact_order = int(item.get("fact_order", idx) or idx)
            rows.append(
                (
                    memory_id,
                    event_id,
                    user_id,
                    group_id,
                    fact_order,
                    fact[:240],
                    fact_norm[:128],
                    int(created_at),
                )
            )
            if len(rows) >= 64:
                break
        if not rows:
            return 0
        self.engine.executemany(
            """
            INSERT INTO memory_event_log(
              memory_id,event_id,user_id,group_id,fact_order,fact,fact_norm,created_at
            ) VALUES(?,?,?,?,?,?,?,?)
            """,
            rows,
        )
        self._sync_event_log_keyword_index(memory_id=memory_id)
        return len(rows)

    def get_memcells_by_memory_id(self, memory_id: str) -> list[dict[str, Any]]:
        return self.engine.query_all(
            """
            SELECT memory_id,event_id,user_id,group_id,cell_order,content,created_at
            FROM memory_memcell
            WHERE memory_id=?
            ORDER BY cell_order ASC
            """,
            (str(memory_id or "").strip(),),
        )

    def get_event_logs_by_memory_id(self, memory_id: str) -> list[dict[str, Any]]:
        return self.engine.query_all(
            """
            SELECT memory_id,event_id,user_id,group_id,fact_order,fact,fact_norm,created_at
            FROM memory_event_log
            WHERE memory_id=?
            ORDER BY fact_order ASC, id ASC
            """,
            (str(memory_id or "").strip(),),
        )

    def fetch_episodes(
        self,
        *,
        user_id: str | None,
        group_id: str | None,
        limit: int = 40,
        candidate_episode_ids: set[str] | None = None,
        as_of_ts: int | None = None,
        start_ts: int | None = None,
        end_ts: int | None = None,
    ) -> list[dict[str, Any]]:
        sql = """
            SELECT
              m.id,
              m.event_id,
              m.user_id,
              m.sender,
              m.group_id,
              m.timestamp,
              m.episode,
              m.summary,
              m.subject,
              m.importance_score,
              m.scene_id,
              m.storage_tier,
              m.memory_category,
              COALESCE(f.fact_text,'') AS atomic_fact_text
            FROM episodic_memory m
            LEFT JOIN (
              SELECT memory_id, group_concat(fact, ' ') AS fact_text
              FROM memory_fact
              GROUP BY memory_id
            ) f ON f.memory_id = m.id
            WHERE m.is_deleted=0
        """
        params: list[Any] = []
        if user_id:
            sql += " AND m.user_id=?"
            params.append(user_id)
        if group_id:
            sql += " AND m.group_id=?"
            params.append(group_id)
        if candidate_episode_ids is not None:
            if not candidate_episode_ids:
                return []
            placeholders = ",".join("?" for _ in candidate_episode_ids)
            sql += f" AND m.id IN ({placeholders})"
            params.extend(list(candidate_episode_ids))
        as_of, start, end = self._normalize_time_window(
            as_of_ts=as_of_ts,
            start_ts=start_ts,
            end_ts=end_ts,
        )
        if as_of is not None or start is not None or end is not None:
            sql += " AND m.timestamp>0"
        if as_of is not None:
            sql += " AND m.timestamp<=?"
            params.append(as_of)
        else:
            if start is not None:
                sql += " AND m.timestamp>=?"
                params.append(start)
            if end is not None:
                sql += " AND m.timestamp<=?"
                params.append(end)
        sql += " ORDER BY m.timestamp DESC LIMIT ?"
        params.append(max(1, int(limit)))
        return self.engine.query_all(sql, params)

    def fetch_episodes_by_ids(self, episode_ids: list[str]) -> list[dict[str, Any]]:
        unique_ids = [x for x in dict.fromkeys(str(v).strip() for v in episode_ids) if x]
        if not unique_ids:
            return []
        placeholders = ",".join("?" for _ in unique_ids)
        rows = self.engine.query_all(
            f"""
            SELECT
              m.id,
              m.event_id,
              m.user_id,
              m.sender,
              m.group_id,
              m.timestamp,
              m.episode,
              m.summary,
              m.subject,
              m.importance_score,
              m.scene_id,
              m.storage_tier,
              m.memory_category
            FROM episodic_memory m
            WHERE m.is_deleted=0
              AND m.id IN ({placeholders})
            """,
            unique_ids,
        )
        order = {eid: idx for idx, eid in enumerate(unique_ids)}
        rows.sort(key=lambda x: order.get(str(x.get("id")), 1_000_000))
        return rows

    def get_episode_ids_by_event_ids(
        self,
        *,
        event_ids: list[str],
        user_id: str | None,
        group_id: str | None,
        limit: int = 500,
        as_of_ts: int | None = None,
        start_ts: int | None = None,
        end_ts: int | None = None,
    ) -> list[str]:
        normalized_event_ids = [
            x for x in dict.fromkeys(str(v).strip() for v in event_ids) if x
        ]
        if not normalized_event_ids:
            return []
        placeholders = ",".join("?" for _ in normalized_event_ids)
        sql = f"""
            SELECT id,event_id,timestamp
            FROM episodic_memory
            WHERE is_deleted=0
              AND event_id IN ({placeholders})
        """
        params: list[Any] = list(normalized_event_ids)
        if user_id:
            sql += " AND user_id=?"
            params.append(user_id)
        if group_id:
            sql += " AND group_id=?"
            params.append(group_id)
        as_of, start, end = self._normalize_time_window(
            as_of_ts=as_of_ts,
            start_ts=start_ts,
            end_ts=end_ts,
        )
        if as_of is not None or start is not None or end is not None:
            sql += " AND timestamp>0"
        if as_of is not None:
            sql += " AND timestamp<=?"
            params.append(as_of)
        else:
            if start is not None:
                sql += " AND timestamp>=?"
                params.append(start)
            if end is not None:
                sql += " AND timestamp<=?"
                params.append(end)
        sql += " LIMIT ?"
        params.append(max(1, int(limit)))
        rows = self.engine.query_all(sql, params)
        order = {
            event_id: idx for idx, event_id in enumerate(normalized_event_ids, start=1)
        }
        rows.sort(
            key=lambda row: (
                order.get(str(row.get("event_id", "")), 1_000_000),
                -int(row.get("timestamp", 0) or 0),
            )
        )
        episode_ids = [str(row.get("id", "")).strip() for row in rows]
        return [x for x in dict.fromkeys(episode_ids) if x]

    def list_groups(self, *, user_id: str | None, limit: int = 100) -> list[dict[str, Any]]:
        sql = """
            SELECT
              m.group_id,
              COUNT(1) AS memory_count,
              MAX(m.timestamp) AS last_timestamp
            FROM episodic_memory m
            WHERE m.is_deleted=0
              AND COALESCE(TRIM(m.group_id), '')<>''
        """
        params: list[Any] = []
        if user_id:
            sql += " AND m.user_id=?"
            params.append(user_id)
        sql += " GROUP BY m.group_id ORDER BY last_timestamp DESC LIMIT ?"
        params.append(max(1, int(limit)))
        return self.engine.query_all(sql, params)

    def search_keyword(
        self,
        *,
        query: str,
        user_id: str | None,
        group_id: str | None,
        top_k: int,
        candidate_episode_ids: set[str] | None = None,
        as_of_ts: int | None = None,
        start_ts: int | None = None,
        end_ts: int | None = None,
    ) -> list[dict[str, Any]]:
        fts_hits = self._search_keyword_fts(
            query=query,
            user_id=user_id,
            group_id=group_id,
            top_k=top_k,
            candidate_episode_ids=candidate_episode_ids,
            as_of_ts=as_of_ts,
            start_ts=start_ts,
            end_ts=end_ts,
        )
        try:
            scan_hits = self._search_keyword_scan(
                query=query,
                user_id=user_id,
                group_id=group_id,
                top_k=max(top_k, int(top_k) * 3),
                candidate_episode_ids=candidate_episode_ids,
                as_of_ts=as_of_ts,
                start_ts=start_ts,
                end_ts=end_ts,
            )
        except sqlite3.OperationalError as exc:
            if "no such table: memory_fact" not in str(exc).lower():
                raise
            scan_hits = []
        return self._merge_ranked_hit_lists(
            primary=fts_hits,
            secondary=scan_hits,
            top_k=top_k,
            primary_name="keyword_fts",
            secondary_name="keyword_scan",
        )

    def search_event_log_keyword(
        self,
        *,
        query: str,
        user_id: str | None,
        group_id: str | None,
        top_k: int,
        candidate_episode_ids: set[str] | None = None,
        as_of_ts: int | None = None,
        start_ts: int | None = None,
        end_ts: int | None = None,
    ) -> list[dict[str, Any]]:
        fts_hits = self._search_event_log_keyword_fts(
            query=query,
            user_id=user_id,
            group_id=group_id,
            top_k=top_k,
            candidate_episode_ids=candidate_episode_ids,
            as_of_ts=as_of_ts,
            start_ts=start_ts,
            end_ts=end_ts,
        )
        scan_hits = self._search_event_log_keyword_scan(
            query=query,
            user_id=user_id,
            group_id=group_id,
            top_k=max(top_k, int(top_k) * 3),
            candidate_episode_ids=candidate_episode_ids,
            as_of_ts=as_of_ts,
            start_ts=start_ts,
            end_ts=end_ts,
        )
        return self._merge_ranked_hit_lists(
            primary=fts_hits,
            secondary=scan_hits,
            top_k=top_k,
            primary_name="event_log_keyword_fts",
            secondary_name="event_log_keyword_scan",
        )

    def search_foresight_keyword(
        self,
        *,
        query: str,
        user_id: str | None,
        group_id: str | None,
        top_k: int,
        candidate_episode_ids: set[str] | None = None,
        as_of_ts: int | None = None,
        start_ts: int | None = None,
        end_ts: int | None = None,
    ) -> list[dict[str, Any]]:
        fts_hits = self._search_foresight_keyword_fts(
            query=query,
            user_id=user_id,
            group_id=group_id,
            top_k=top_k,
            candidate_episode_ids=candidate_episode_ids,
            as_of_ts=as_of_ts,
            start_ts=start_ts,
            end_ts=end_ts,
        )
        scan_hits = self._search_foresight_keyword_scan(
            query=query,
            user_id=user_id,
            group_id=group_id,
            top_k=max(top_k, int(top_k) * 3),
            candidate_episode_ids=candidate_episode_ids,
            as_of_ts=as_of_ts,
            start_ts=start_ts,
            end_ts=end_ts,
        )
        return self._merge_ranked_hit_lists(
            primary=fts_hits,
            secondary=scan_hits,
            top_k=top_k,
            primary_name="foresight_keyword_fts",
            secondary_name="foresight_keyword_scan",
        )

    def _search_keyword_scan(
        self,
        *,
        query: str,
        user_id: str | None,
        group_id: str | None,
        top_k: int,
        candidate_episode_ids: set[str] | None = None,
        as_of_ts: int | None = None,
        start_ts: int | None = None,
        end_ts: int | None = None,
    ) -> list[dict[str, Any]]:
        rows = self.fetch_episodes(
            user_id=user_id,
            group_id=group_id,
            limit=max(800, top_k * 40),
            candidate_episode_ids=candidate_episode_ids,
            as_of_ts=as_of_ts,
            start_ts=start_ts,
            end_ts=end_ts,
        )
        terms = self._tokenize_text(query)
        if not terms:
            terms = [query.lower()]
        hinted_categories = set(self._infer_query_categories(query))
        scored: list[dict[str, Any]] = []
        for row in rows:
            raw_text = " ".join(
                [
                    str(row.get("episode", "")),
                    str(row.get("summary", "")),
                    str(row.get("subject", "")),
                    str(row.get("atomic_fact_text", "")),
                    f"cat_{self._normalize_category(str(row.get('memory_category') or 'general'))}",
                    f"tier_{self._normalize_token_tag(str(row.get('storage_tier') or 'text_only'), fallback='text_only')}",
                ]
            ).lower()
            text_tokens = self._tokenize_text(raw_text)
            score = 0.0
            for term in terms:
                if term in text_tokens:
                    score += 1.0
                score += float(raw_text.count(term))
            if score <= 0 and query.lower() not in raw_text:
                continue
            memory_category = self._normalize_category(str(row.get("memory_category") or "general"))
            if hinted_categories and memory_category in hinted_categories:
                score += 1.8
            if score <= 0:
                score = 1.0
            scored.append(
                {
                    "memory_id": row["id"],
                    "bm25_score": float(score),
                    "score": float(score),
                    "source": "keyword",
                }
            )
        scored.sort(key=lambda x: float(x["score"]), reverse=True)
        return scored[: max(1, int(top_k))]

    def _search_keyword_fts(
        self,
        *,
        query: str,
        user_id: str | None,
        group_id: str | None,
        top_k: int,
        candidate_episode_ids: set[str] | None = None,
        as_of_ts: int | None = None,
        start_ts: int | None = None,
        end_ts: int | None = None,
    ) -> list[dict[str, Any]]:
        terms = self._tokenize_text(query)
        if not terms:
            terms = [query.lower()]
        dedup_terms = list(dict.fromkeys(t for t in terms if t.strip()))
        if not dedup_terms:
            return []
        hinted_categories = self._infer_query_categories(query)
        for cat in hinted_categories:
            dedup_terms.append(f"cat_{cat}")
        dedup_terms = list(dict.fromkeys(dedup_terms))
        # Keep query plan bounded on device-side workloads.
        selected = dedup_terms[:24]
        core_terms = [t for t in selected if not str(t).startswith("cat_")]
        if not core_terms:
            core_terms = list(selected)
        match_query = " OR ".join(
            f'"{self._escape_fts_term(t)}"'
            for t in selected
        )
        sql = """
            SELECT
              m.id AS memory_id,
              memory_keyword_fts.search_text AS search_text,
              bm25(memory_keyword_fts) AS bm25_score
            FROM memory_keyword_fts
            JOIN episodic_memory m ON m.id = memory_keyword_fts.memory_id
            WHERE memory_keyword_fts MATCH ?
              AND m.is_deleted=0
        """
        params: list[Any] = [match_query]
        if user_id:
            sql += " AND m.user_id=?"
            params.append(user_id)
        if group_id:
            sql += " AND m.group_id=?"
            params.append(group_id)
        if candidate_episode_ids is not None:
            if not candidate_episode_ids:
                return []
            placeholders = ",".join("?" for _ in candidate_episode_ids)
            sql += f" AND m.id IN ({placeholders})"
            params.extend(list(candidate_episode_ids))
        as_of, start, end = self._normalize_time_window(
            as_of_ts=as_of_ts,
            start_ts=start_ts,
            end_ts=end_ts,
        )
        if as_of is not None or start is not None or end is not None:
            sql += " AND m.timestamp>0"
        if as_of is not None:
            sql += " AND m.timestamp<=?"
            params.append(as_of)
        else:
            if start is not None:
                sql += " AND m.timestamp>=?"
                params.append(start)
            if end is not None:
                sql += " AND m.timestamp<=?"
                params.append(end)
        sql += " ORDER BY bm25(memory_keyword_fts) ASC, m.timestamp DESC LIMIT ?"
        params.append(max(100, int(top_k) * 16))
        try:
            rows = self.engine.query_all(sql, params)
        except Exception:
            return []
        scored: list[dict[str, Any]] = []
        for row in rows:
            raw_text = str(row.get("search_text", "")).lower()
            lexical = 0.0
            for term in core_terms:
                lexical += float(raw_text.count(term))
            term_hits = 0
            for term in core_terms:
                if term and term in raw_text:
                    term_hits += 1
            term_coverage = float(term_hits) / float(max(1, len(core_terms)))
            lexical_norm = min(1.0, float(lexical) / float(max(1, len(core_terms) * 2)))
            category_bonus = 0.0
            for cat in hinted_categories:
                if f"cat_{cat}" in raw_text:
                    category_bonus += 0.12
            bm25_raw = row.get("bm25_score")
            try:
                bm25_val = float(bm25_raw if bm25_raw is not None else 0.0)
            except Exception:
                bm25_val = 0.0
            fts_score = 1.0 / (1.0 + max(0.0, bm25_val))
            score = (
                1.95 * float(term_coverage)
                + 0.72 * float(lexical_norm)
                + 0.90 * float(fts_score)
                + float(category_bonus)
            )
            if len(core_terms) >= 2 and term_hits <= 1:
                score *= 0.72
            scored.append(
                {
                    "memory_id": row["memory_id"],
                    "bm25_score": bm25_val,
                    "score": float(score),
                    "source": "keyword_fts",
                }
            )
        scored.sort(
            key=lambda x: (
                float(x.get("score", 0.0)),
                -float(x.get("bm25_score", 0.0)),
            ),
            reverse=True,
        )
        return scored[: max(1, int(top_k))]

    def _search_event_log_keyword_fts(
        self,
        *,
        query: str,
        user_id: str | None,
        group_id: str | None,
        top_k: int,
        candidate_episode_ids: set[str] | None = None,
        as_of_ts: int | None = None,
        start_ts: int | None = None,
        end_ts: int | None = None,
    ) -> list[dict[str, Any]]:
        terms = self._tokenize_text(query)
        if not terms:
            terms = [query.lower()]
        dedup_terms = list(dict.fromkeys(t for t in terms if t.strip()))
        if not dedup_terms:
            return []
        selected = dedup_terms[:24]
        core_terms = list(selected)
        match_query = " OR ".join(f'"{self._escape_fts_term(t)}"' for t in selected)
        sql = """
            SELECT
              idx.memory_id AS memory_id,
              idx.fact_text AS fact_text,
              bm25(event_log_keyword_fts) AS bm25_score
            FROM event_log_keyword_fts idx
            JOIN episodic_memory m ON m.id = idx.memory_id
            WHERE event_log_keyword_fts MATCH ?
              AND m.is_deleted=0
        """
        params: list[Any] = [match_query]
        if user_id:
            sql += " AND m.user_id=?"
            params.append(user_id)
        if group_id:
            sql += " AND m.group_id=?"
            params.append(group_id)
        if candidate_episode_ids is not None:
            if not candidate_episode_ids:
                return []
            placeholders = ",".join("?" for _ in candidate_episode_ids)
            sql += f" AND m.id IN ({placeholders})"
            params.extend(list(candidate_episode_ids))
        as_of, start, end = self._normalize_time_window(
            as_of_ts=as_of_ts,
            start_ts=start_ts,
            end_ts=end_ts,
        )
        if as_of is not None or start is not None or end is not None:
            sql += " AND m.timestamp>0"
        if as_of is not None:
            sql += " AND m.timestamp<=?"
            params.append(as_of)
        else:
            if start is not None:
                sql += " AND m.timestamp>=?"
                params.append(start)
            if end is not None:
                sql += " AND m.timestamp<=?"
                params.append(end)
        sql += " ORDER BY bm25(event_log_keyword_fts) ASC, m.timestamp DESC LIMIT ?"
        params.append(max(120, int(top_k) * 20))
        try:
            rows = self.engine.query_all(sql, params)
        except Exception:
            return []
        scored: dict[str, dict[str, Any]] = {}
        for row in rows:
            memory_id = str(row.get("memory_id", "")).strip()
            if not memory_id:
                continue
            raw_text = str(row.get("fact_text", "")).lower()
            term_hits = 0
            lexical = 0.0
            for term in core_terms:
                count = float(raw_text.count(term))
                lexical += count
                if count > 0:
                    term_hits += 1
            if term_hits <= 0 and str(query or "").strip().lower() not in raw_text:
                continue
            coverage = float(term_hits) / float(max(1, len(core_terms)))
            lexical_norm = min(1.0, float(lexical) / float(max(1, len(core_terms) * 2)))
            bm25_raw = row.get("bm25_score")
            try:
                bm25_val = float(bm25_raw if bm25_raw is not None else 0.0)
            except Exception:
                bm25_val = 0.0
            fts_score = 1.0 / (1.0 + max(0.0, bm25_val))
            row_score = (
                1.82 * float(coverage) + 0.70 * float(lexical_norm) + 0.95 * float(fts_score)
            )
            prev = scored.get(memory_id)
            if prev is None:
                scored[memory_id] = {
                    "memory_id": memory_id,
                    "bm25_score": bm25_val,
                    "score": float(row_score),
                    "source": "event_log_keyword_fts",
                    "event_log_support_count": 1,
                    "event_log_fact_hint": str(row.get("fact_text", ""))[:240],
                }
                continue
            prev["event_log_support_count"] = int(prev.get("event_log_support_count", 1)) + 1
            prev["score"] = max(float(prev.get("score", 0.0)), float(row_score)) + 0.03
            prev["bm25_score"] = min(
                float(prev.get("bm25_score", bm25_val)),
                float(bm25_val),
            )
        out = list(scored.values())
        out.sort(
            key=lambda x: (
                float(x.get("score", 0.0)),
                float(x.get("event_log_support_count", 0)),
                -float(x.get("bm25_score", 0.0)),
            ),
            reverse=True,
        )
        return out[: max(1, int(top_k))]

    def _search_event_log_keyword_scan(
        self,
        *,
        query: str,
        user_id: str | None,
        group_id: str | None,
        top_k: int,
        candidate_episode_ids: set[str] | None = None,
        as_of_ts: int | None = None,
        start_ts: int | None = None,
        end_ts: int | None = None,
    ) -> list[dict[str, Any]]:
        terms = self._tokenize_text(query)
        if not terms:
            terms = [query.lower()]
        core_terms = list(dict.fromkeys(t for t in terms if t.strip()))
        if not core_terms:
            return []
        sql = """
            SELECT
              e.memory_id,
              e.fact,
              m.timestamp
            FROM memory_event_log e
            JOIN episodic_memory m ON m.id = e.memory_id
            WHERE m.is_deleted=0
        """
        params: list[Any] = []
        if user_id:
            sql += " AND m.user_id=?"
            params.append(user_id)
        if group_id:
            sql += " AND m.group_id=?"
            params.append(group_id)
        if candidate_episode_ids is not None:
            if not candidate_episode_ids:
                return []
            placeholders = ",".join("?" for _ in candidate_episode_ids)
            sql += f" AND m.id IN ({placeholders})"
            params.extend(list(candidate_episode_ids))
        as_of, start, end = self._normalize_time_window(
            as_of_ts=as_of_ts,
            start_ts=start_ts,
            end_ts=end_ts,
        )
        if as_of is not None or start is not None or end is not None:
            sql += " AND m.timestamp>0"
        if as_of is not None:
            sql += " AND m.timestamp<=?"
            params.append(as_of)
        else:
            if start is not None:
                sql += " AND m.timestamp>=?"
                params.append(start)
            if end is not None:
                sql += " AND m.timestamp<=?"
                params.append(end)
        like_terms = core_terms[:8]
        if like_terms:
            clauses = []
            for term in like_terms:
                clauses.append("LOWER(e.fact) LIKE ?")
                params.append(f"%{term}%")
            sql += " AND (" + " OR ".join(clauses) + ")"
        sql += " ORDER BY m.timestamp DESC, e.fact_order ASC LIMIT ?"
        params.append(max(200, int(top_k) * 40))
        try:
            rows = self.engine.query_all(sql, params)
        except Exception:
            return []
        scored: dict[str, dict[str, Any]] = {}
        query_low = str(query or "").strip().lower()
        for row in rows:
            memory_id = str(row.get("memory_id", "")).strip()
            if not memory_id:
                continue
            raw_text = str(row.get("fact", "")).lower()
            term_hits = 0
            lexical = 0.0
            for term in core_terms:
                count = float(raw_text.count(term))
                lexical += count
                if count > 0:
                    term_hits += 1
            if term_hits <= 0 and query_low not in raw_text:
                continue
            coverage = float(term_hits) / float(max(1, len(core_terms)))
            lexical_norm = min(1.0, float(lexical) / float(max(1, len(core_terms) * 2)))
            row_score = 1.88 * float(coverage) + 0.78 * float(lexical_norm)
            prev = scored.get(memory_id)
            if prev is None:
                scored[memory_id] = {
                    "memory_id": memory_id,
                    "bm25_score": float(row_score),
                    "score": float(row_score),
                    "source": "event_log_keyword_scan",
                    "event_log_support_count": 1,
                    "event_log_fact_hint": str(row.get("fact", ""))[:240],
                }
                continue
            prev["event_log_support_count"] = int(prev.get("event_log_support_count", 1)) + 1
            prev["score"] = max(float(prev.get("score", 0.0)), float(row_score)) + 0.02
        out = list(scored.values())
        out.sort(
            key=lambda x: (
                float(x.get("score", 0.0)),
                float(x.get("event_log_support_count", 0)),
            ),
            reverse=True,
        )
        return out[: max(1, int(top_k))]

    def _search_foresight_keyword_fts(
        self,
        *,
        query: str,
        user_id: str | None,
        group_id: str | None,
        top_k: int,
        candidate_episode_ids: set[str] | None = None,
        as_of_ts: int | None = None,
        start_ts: int | None = None,
        end_ts: int | None = None,
    ) -> list[dict[str, Any]]:
        terms = self._tokenize_text(query)
        if not terms:
            terms = [query.lower()]
        dedup_terms = list(dict.fromkeys(t for t in terms if t.strip()))
        if not dedup_terms:
            return []
        selected = dedup_terms[:24]
        core_terms = list(selected)
        match_query = " OR ".join(f'"{self._escape_fts_term(t)}"' for t in selected)
        sql = """
            SELECT
              idx.memory_id AS memory_id,
              idx.foresight_text AS foresight_text,
              bm25(foresight_keyword_fts) AS bm25_score
            FROM foresight_keyword_fts idx
            JOIN episodic_memory m ON m.id = idx.memory_id
            WHERE foresight_keyword_fts MATCH ?
              AND m.is_deleted=0
        """
        params: list[Any] = [match_query]
        if user_id:
            sql += " AND m.user_id=?"
            params.append(user_id)
        if group_id:
            sql += " AND m.group_id=?"
            params.append(group_id)
        if candidate_episode_ids is not None:
            if not candidate_episode_ids:
                return []
            placeholders = ",".join("?" for _ in candidate_episode_ids)
            sql += f" AND m.id IN ({placeholders})"
            params.extend(list(candidate_episode_ids))
        as_of, start, end = self._normalize_time_window(
            as_of_ts=as_of_ts,
            start_ts=start_ts,
            end_ts=end_ts,
        )
        if as_of is not None or start is not None or end is not None:
            sql += " AND m.timestamp>0"
        if as_of is not None:
            sql += " AND m.timestamp<=?"
            params.append(as_of)
        else:
            if start is not None:
                sql += " AND m.timestamp>=?"
                params.append(start)
            if end is not None:
                sql += " AND m.timestamp<=?"
                params.append(end)
        sql += " ORDER BY bm25(foresight_keyword_fts) ASC, m.timestamp DESC LIMIT ?"
        params.append(max(120, int(top_k) * 20))
        try:
            rows = self.engine.query_all(sql, params)
        except Exception:
            return []
        scored: dict[str, dict[str, Any]] = {}
        for row in rows:
            memory_id = str(row.get("memory_id", "")).strip()
            if not memory_id:
                continue
            raw_text = str(row.get("foresight_text", "")).lower()
            term_hits = 0
            lexical = 0.0
            for term in core_terms:
                count = float(raw_text.count(term))
                lexical += count
                if count > 0:
                    term_hits += 1
            if term_hits <= 0 and str(query or "").strip().lower() not in raw_text:
                continue
            coverage = float(term_hits) / float(max(1, len(core_terms)))
            lexical_norm = min(1.0, float(lexical) / float(max(1, len(core_terms) * 2)))
            bm25_raw = row.get("bm25_score")
            try:
                bm25_val = float(bm25_raw if bm25_raw is not None else 0.0)
            except Exception:
                bm25_val = 0.0
            fts_score = 1.0 / (1.0 + max(0.0, bm25_val))
            row_score = 1.80 * float(coverage) + 0.68 * float(lexical_norm) + 0.90 * float(
                fts_score
            )
            prev = scored.get(memory_id)
            if prev is None:
                scored[memory_id] = {
                    "memory_id": memory_id,
                    "bm25_score": bm25_val,
                    "score": float(row_score),
                    "source": "foresight_keyword_fts",
                    "foresight_support_count": 1,
                    "foresight_text_hint": str(row.get("foresight_text", ""))[:240],
                }
                continue
            prev["foresight_support_count"] = int(prev.get("foresight_support_count", 1)) + 1
            prev["score"] = max(float(prev.get("score", 0.0)), float(row_score)) + 0.03
            prev["bm25_score"] = min(float(prev.get("bm25_score", bm25_val)), float(bm25_val))
        out = list(scored.values())
        out.sort(
            key=lambda x: (
                float(x.get("score", 0.0)),
                float(x.get("foresight_support_count", 0)),
                -float(x.get("bm25_score", 0.0)),
            ),
            reverse=True,
        )
        return out[: max(1, int(top_k))]

    def _search_foresight_keyword_scan(
        self,
        *,
        query: str,
        user_id: str | None,
        group_id: str | None,
        top_k: int,
        candidate_episode_ids: set[str] | None = None,
        as_of_ts: int | None = None,
        start_ts: int | None = None,
        end_ts: int | None = None,
    ) -> list[dict[str, Any]]:
        terms = self._tokenize_text(query)
        if not terms:
            terms = [query.lower()]
        core_terms = list(dict.fromkeys(t for t in terms if t.strip()))
        if not core_terms:
            return []
        sql = """
            SELECT
              f.memory_id,
              f.content AS foresight_text,
              m.timestamp
            FROM memory_foresight f
            JOIN episodic_memory m ON m.id = f.memory_id
            WHERE m.is_deleted=0
        """
        params: list[Any] = []
        if user_id:
            sql += " AND m.user_id=?"
            params.append(user_id)
        if group_id:
            sql += " AND m.group_id=?"
            params.append(group_id)
        if candidate_episode_ids is not None:
            if not candidate_episode_ids:
                return []
            placeholders = ",".join("?" for _ in candidate_episode_ids)
            sql += f" AND m.id IN ({placeholders})"
            params.extend(list(candidate_episode_ids))
        as_of, start, end = self._normalize_time_window(
            as_of_ts=as_of_ts,
            start_ts=start_ts,
            end_ts=end_ts,
        )
        if as_of is not None or start is not None or end is not None:
            sql += " AND m.timestamp>0"
        if as_of is not None:
            sql += " AND m.timestamp<=?"
            params.append(as_of)
        else:
            if start is not None:
                sql += " AND m.timestamp>=?"
                params.append(start)
            if end is not None:
                sql += " AND m.timestamp<=?"
                params.append(end)
        like_terms = core_terms[:8]
        if like_terms:
            clauses = []
            for term in like_terms:
                clauses.append("LOWER(f.content) LIKE ?")
                params.append(f"%{term}%")
            sql += " AND (" + " OR ".join(clauses) + ")"
        sql += " ORDER BY m.timestamp DESC LIMIT ?"
        params.append(max(200, int(top_k) * 40))
        try:
            rows = self.engine.query_all(sql, params)
        except Exception:
            return []
        scored: dict[str, dict[str, Any]] = {}
        query_low = str(query or "").strip().lower()
        for row in rows:
            memory_id = str(row.get("memory_id", "")).strip()
            if not memory_id:
                continue
            raw_text = str(row.get("foresight_text", "")).lower()
            term_hits = 0
            lexical = 0.0
            for term in core_terms:
                count = float(raw_text.count(term))
                lexical += count
                if count > 0:
                    term_hits += 1
            if term_hits <= 0 and query_low not in raw_text:
                continue
            coverage = float(term_hits) / float(max(1, len(core_terms)))
            lexical_norm = min(1.0, float(lexical) / float(max(1, len(core_terms) * 2)))
            row_score = 1.78 * float(coverage) + 0.74 * float(lexical_norm)
            prev = scored.get(memory_id)
            if prev is None:
                scored[memory_id] = {
                    "memory_id": memory_id,
                    "bm25_score": float(row_score),
                    "score": float(row_score),
                    "source": "foresight_keyword_scan",
                    "foresight_support_count": 1,
                    "foresight_text_hint": str(row.get("foresight_text", ""))[:240],
                }
                continue
            prev["foresight_support_count"] = int(prev.get("foresight_support_count", 1)) + 1
            prev["score"] = max(float(prev.get("score", 0.0)), float(row_score)) + 0.02
        out = list(scored.values())
        out.sort(
            key=lambda x: (
                float(x.get("score", 0.0)),
                float(x.get("foresight_support_count", 0)),
            ),
            reverse=True,
        )
        return out[: max(1, int(top_k))]

    @staticmethod
    def _merge_ranked_hit_lists(
        *,
        primary: list[dict[str, Any]],
        secondary: list[dict[str, Any]],
        top_k: int,
        primary_name: str,
        secondary_name: str,
    ) -> list[dict[str, Any]]:
        if not primary and not secondary:
            return []
        if primary and not secondary:
            return primary[: max(1, int(top_k))]
        if secondary and not primary:
            return secondary[: max(1, int(top_k))]
        score_map: dict[str, dict[str, Any]] = {}
        primary_rank: dict[str, int] = {}
        secondary_rank: dict[str, int] = {}
        for idx, row in enumerate(primary, start=1):
            memory_id = str(row.get("memory_id", "")).strip()
            if not memory_id:
                continue
            primary_rank[memory_id] = idx
            item = dict(row)
            item["merge_sources"] = [primary_name]
            score_map[memory_id] = item
        for idx, row in enumerate(secondary, start=1):
            memory_id = str(row.get("memory_id", "")).strip()
            if not memory_id:
                continue
            secondary_rank[memory_id] = idx
            existing = score_map.get(memory_id)
            if existing is None:
                item = dict(row)
                item["merge_sources"] = [secondary_name]
                score_map[memory_id] = item
                continue
            sources = list(existing.get("merge_sources", []))
            if secondary_name not in sources:
                sources.append(secondary_name)
            existing["merge_sources"] = sources
            if not existing.get("event_log_fact_hint") and row.get("event_log_fact_hint"):
                existing["event_log_fact_hint"] = row.get("event_log_fact_hint")
            support_count = int(existing.get("event_log_support_count", 0)) + int(
                row.get("event_log_support_count", 0)
            )
            if support_count > 0:
                existing["event_log_support_count"] = support_count
        rank_bias = 12.0
        for memory_id, item in score_map.items():
            p_rank = primary_rank.get(memory_id)
            s_rank = secondary_rank.get(memory_id)
            merged_score = 0.0
            if p_rank is not None:
                merged_score += 1.0 / float(rank_bias + p_rank)
            if s_rank is not None:
                merged_score += 0.58 / float(rank_bias + s_rank)
            if p_rank is not None and s_rank is not None:
                merged_score += 0.02
            item["score"] = float(merged_score)
        merged = list(score_map.values())
        merged.sort(
            key=lambda x: (
                float(x.get("score", 0.0)),
                1 if primary_name in list(x.get("merge_sources", [])) else 0,
                len(list(x.get("merge_sources", []))),
            ),
            reverse=True,
        )
        return merged[: max(1, int(top_k))]

    @staticmethod
    def _normalize_time_window(
        *,
        as_of_ts: int | None,
        start_ts: int | None,
        end_ts: int | None,
    ) -> tuple[int | None, int | None, int | None]:
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
            return as_of, None, None
        if start is not None and end is not None and start > end:
            start, end = end, start
        return None, start, end

    @staticmethod
    def _tokenize_text(text: str) -> list[str]:
        raw = str(text or "").lower().strip()
        if not raw:
            return []
        tokens = [
            t
            for t in re.findall(r"[a-z0-9][a-z0-9_-]{0,63}|[\u4e00-\u9fff]{1,4}", raw)
            if t
        ]
        chars = re.findall(r"[\u4e00-\u9fff]", raw)
        for i in range(len(chars) - 1):
            tokens.append(chars[i] + chars[i + 1])
        if chars:
            tokens.append("".join(chars))
        stop = {
            "我",
            "你",
            "他",
            "她",
            "它",
            "吗",
            "呢",
            "啊",
            "呀",
            "的",
            "了",
            "what",
            "when",
            "where",
            "which",
            "who",
            "whom",
            "whose",
            "why",
            "how",
            "is",
            "are",
            "am",
            "was",
            "were",
            "be",
            "been",
            "being",
            "do",
            "does",
            "did",
            "done",
            "can",
            "could",
            "would",
            "should",
            "will",
            "shall",
            "may",
            "might",
            "must",
            "to",
            "of",
            "in",
            "on",
            "at",
            "for",
            "by",
            "with",
            "about",
            "from",
            "as",
            "into",
            "through",
            "after",
            "before",
            "up",
            "down",
            "over",
            "under",
            "again",
            "further",
            "then",
            "once",
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "if",
            "because",
            "while",
            "this",
            "that",
            "these",
            "those",
            "it",
            "its",
            "i",
            "you",
            "he",
            "she",
            "they",
            "we",
            "me",
            "him",
            "her",
            "them",
            "my",
            "your",
            "his",
            "their",
            "our",
            "mine",
            "yours",
            "hers",
            "theirs",
            "ours",
        }
        return [t for t in tokens if t and t not in stop]

    @staticmethod
    def _escape_fts_term(term: str) -> str:
        return str(term or "").replace('"', '""')

    @staticmethod
    def _normalize_category(category: str, *, fallback: str = "general") -> str:
        name = re.sub(r"[^a-z0-9_]+", "_", str(category or "").strip().lower()).strip("_")
        if not name:
            name = fallback
        return name if name in _CATEGORY_SET else fallback

    @staticmethod
    def _normalize_token_tag(value: str, *, fallback: str) -> str:
        name = re.sub(r"[^a-z0-9_]+", "_", str(value or "").strip().lower()).strip("_")
        return name or fallback

    @staticmethod
    def _infer_query_categories(query: str) -> list[str]:
        text = str(query or "").strip().lower()
        if not text:
            return []
        mapping = (
            (
                "profile",
                ("我是谁", "我的名字", "叫什", "name", "生日", "联系方式", "住址", "profile"),
            ),
            (
                "task",
                ("待办", "任务", "todo", "deadline", "截止", "提醒", "完成", "进度"),
            ),
            (
                "plan",
                ("计划", "打算", "明天", "下周", "下个月", "行程", "安排", "future", "next"),
            ),
            (
                "event",
                ("发生", "什么时候", "昨天", "今天", "上周", "去年", "会议", "旅行", "event"),
            ),
            (
                "knowledge",
                ("是什么", "为什么", "关系", "定义", "知识", "项目", "ticket", "issue", "how", "what"),
            ),
        )
        matched: list[str] = []
        for category, terms in mapping:
            if any(term in text for term in terms):
                matched.append(category)
        return matched

    def get_latest_profile_snapshot(
        self, user_id: str | None, group_id: str | None
    ) -> dict[str, Any] | None:
        sql = """
            SELECT event_id,user_id,group_id,profile_json,timestamp
            FROM profile_snapshot
            WHERE 1=1
        """
        params: list[Any] = []
        if user_id:
            sql += " AND user_id=?"
            params.append(user_id)
        if group_id:
            sql += " AND group_id=?"
            params.append(group_id)
        sql += " ORDER BY timestamp DESC, id DESC LIMIT 1"
        row = self.engine.query_one(sql, params)
        if row is None:
            return None
        row["profile"] = json.loads(str(row["profile_json"]))
        return row

    def upsert_profile_snapshot(
        self,
        *,
        event_id: str,
        user_id: str,
        group_id: str | None,
        profile_patch: dict[str, Any],
        timestamp: int,
    ) -> None:
        prev = self.get_latest_profile_snapshot(user_id=user_id, group_id=group_id)
        merged = dict(prev["profile"]) if prev and isinstance(prev.get("profile"), dict) else {}
        merged.update({k: v for k, v in profile_patch.items() if str(k).strip()})
        self.engine.execute(
            """
            INSERT INTO profile_snapshot(event_id,user_id,group_id,profile_json,timestamp)
            VALUES(?,?,?,?,?)
            """,
            (event_id, user_id, group_id, json.dumps(merged, ensure_ascii=False), int(timestamp)),
        )

    def get_valid_foresights_for_episodes(
        self,
        *,
        episode_ids: list[str],
        user_id: str | None,
        group_id: str | None,
        as_of_ts: int | None = None,
        start_ts: int | None = None,
        end_ts: int | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        if not episode_ids:
            return {}
        placeholders = ",".join("?" for _ in episode_ids)
        rows = self.engine.query_all(
            f"""
            SELECT f.memory_id,f.content,f.start_time,f.end_time,f.confidence,m.user_id,m.group_id
            FROM memory_foresight f
            JOIN episodic_memory m ON m.id=f.memory_id
            WHERE m.is_deleted=0 AND f.memory_id IN ({placeholders})
            """,
            episode_ids,
        )
        query_start = int(start_ts) if start_ts is not None else None
        query_end = int(end_ts) if end_ts is not None else None
        use_interval = query_start is not None or query_end is not None
        ref_ts = int(as_of_ts) if as_of_ts is not None else int(time.time())
        out: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if user_id and row["user_id"] != user_id:
                continue
            if group_id and row["group_id"] != group_id:
                continue
            st_raw = row.get("start_time")
            et_raw = row.get("end_time")
            st = int(st_raw) if st_raw is not None else None
            et = int(et_raw) if et_raw is not None else None
            if st is not None and et is not None and st > et:
                continue
            if use_interval:
                # overlap([st, et], [query_start, query_end]) for open-ended bounds
                if query_start is not None and et is not None and et < query_start:
                    continue
                if query_end is not None and st is not None and st > query_end:
                    continue
            else:
                if st is not None and st > ref_ts:
                    continue
                if et is not None and et < ref_ts:
                    continue
            item = {
                "content": row["content"],
                "start_time": st,
                "end_time": et,
                "confidence": float(row.get("confidence", 0.5)),
            }
            if use_interval:
                item["query_start_time"] = query_start
                item["query_end_time"] = query_end
            else:
                item["as_of_time"] = ref_ts
            out[str(row["memory_id"])].append(item)
        return dict(out)

    def get_memscene_candidates(
        self, *, user_id: str | None, group_id: str | None, limit: int = 300
    ) -> list[dict[str, Any]]:
        sql = """
            SELECT id,user_id,group_id,summary,centroid_vector_json,memory_count,updated_at,last_memory_ts
            FROM memscene
            WHERE 1=1
        """
        params: list[Any] = []
        if user_id:
            sql += " AND user_id=?"
            params.append(user_id)
        if group_id:
            sql += " AND group_id=?"
            params.append(group_id)
        sql += " ORDER BY updated_at DESC LIMIT ?"
        params.append(max(1, int(limit)))
        return self.engine.query_all(sql, params)

    def get_episode_ids_by_memscene_ids(
        self, memscene_ids: list[str], *, limit: int = 500
    ) -> list[str]:
        if not memscene_ids:
            return []
        placeholders = ",".join("?" for _ in memscene_ids)
        rows = self.engine.query_all(
            f"""
            SELECT memory_id,scene_id,created_at
            FROM memscene_memory_link
            WHERE scene_id IN ({placeholders})
            ORDER BY created_at DESC
            LIMIT ?
            """,
            [*memscene_ids, max(1, int(limit))],
        )
        return [str(r.get("memory_id")) for r in rows if r.get("memory_id")]

    def get_episode_scene_map(self, episode_ids: list[str]) -> dict[str, str]:
        if not episode_ids:
            return {}
        placeholders = ",".join("?" for _ in episode_ids)
        rows = self.engine.query_all(
            f"""
            SELECT memory_id,scene_id
            FROM memscene_memory_link
            WHERE memory_id IN ({placeholders})
            """,
            episode_ids,
        )
        return {
            str(r.get("memory_id")): str(r.get("scene_id"))
            for r in rows
            if r.get("memory_id") and r.get("scene_id")
        }

    def update_memory_scene(self, *, memory_id: str, scene_id: str, storage_tier: str) -> None:
        now = int(time.time())
        self.engine.execute(
            """
            UPDATE episodic_memory
            SET scene_id=?, storage_tier=?, updated_at=?
            WHERE id=?
            """,
            (scene_id, storage_tier, now, memory_id),
        )
        self.engine.execute(
            """
            INSERT INTO memscene_memory_link(memory_id,scene_id,user_id,group_id,created_at)
            SELECT id,?,user_id,group_id,?
            FROM episodic_memory
            WHERE id=?
            ON CONFLICT(memory_id) DO UPDATE SET
              scene_id=excluded.scene_id,
              created_at=excluded.created_at
            """,
            (scene_id, now, memory_id),
        )

    def create_memscene(
        self,
        *,
        scene_id: str,
        user_id: str,
        group_id: str | None,
        summary: str,
        centroid_vector_json: str,
        last_memory_ts: int,
    ) -> dict[str, Any]:
        now = int(time.time())
        self.engine.execute(
            """
            INSERT INTO memscene(
              id,user_id,group_id,summary,centroid_vector_json,memory_count,created_at,updated_at,last_memory_ts
            ) VALUES(?,?,?,?,?,?,?,?,?)
            """,
            (
                scene_id,
                user_id,
                group_id,
                summary,
                centroid_vector_json,
                1,
                now,
                now,
                int(last_memory_ts),
            ),
        )
        return {
            "id": scene_id,
            "user_id": user_id,
            "group_id": group_id,
            "summary": summary,
            "centroid_vector_json": centroid_vector_json,
            "memory_count": 1,
            "created_at": now,
            "updated_at": now,
            "last_memory_ts": int(last_memory_ts),
        }

    def update_memscene(
        self,
        *,
        scene_id: str,
        summary: str,
        centroid_vector_json: str,
        memory_count_delta: int,
        last_memory_ts: int,
    ) -> None:
        now = int(time.time())
        self.engine.execute(
            """
            UPDATE memscene
            SET summary=?,
                centroid_vector_json=?,
                memory_count=MAX(0,memory_count+?),
                updated_at=?,
                last_memory_ts=?
            WHERE id=?
            """,
            (
                summary,
                centroid_vector_json,
                int(memory_count_delta),
                now,
                int(last_memory_ts),
                scene_id,
            ),
        )

    def get_memscene(self, scene_id: str) -> dict[str, Any] | None:
        return self.engine.query_one(
            """
            SELECT id,user_id,group_id,summary,centroid_vector_json,memory_count,created_at,updated_at,last_memory_ts
            FROM memscene WHERE id=?
            """,
            (scene_id,),
        )

    def list_recent_conflicts(
        self, *, user_id: str | None, group_id: str | None, limit: int = 20
    ) -> list[dict[str, Any]]:
        sql = """
            SELECT id,user_id,group_id,field_name,old_value,new_value,happened_at,evidence_event_id
            FROM profile_conflict_log
            WHERE 1=1
        """
        params: list[Any] = []
        if user_id:
            sql += " AND user_id=?"
            params.append(user_id)
        if group_id:
            sql += " AND group_id=?"
            params.append(group_id)
        sql += " ORDER BY happened_at DESC LIMIT ?"
        params.append(max(1, int(limit)))
        return self.engine.query_all(sql, params)

    def insert_profile_conflict(
        self,
        *,
        conflict_id: str,
        user_id: str,
        group_id: str | None,
        field_name: str,
        old_value: str,
        new_value: str,
        happened_at: int,
        evidence_event_id: str,
    ) -> None:
        self.engine.execute(
            """
            INSERT OR REPLACE INTO profile_conflict_log(
              id,user_id,group_id,field_name,old_value,new_value,happened_at,evidence_event_id
            ) VALUES(?,?,?,?,?,?,?,?)
            """,
            (
                conflict_id,
                user_id,
                group_id,
                field_name,
                old_value,
                new_value,
                int(happened_at),
                evidence_event_id,
            ),
        )

    def get_conversation_segment(self, conversation_id: str) -> dict[str, Any] | None:
        return self.engine.query_one(
            """
            SELECT conversation_id,user_id,group_id,segment_seq,turns_markdown,last_query,turn_count,start_time,last_time,updated_at
            FROM conversation_segment WHERE conversation_id=?
            """,
            (conversation_id,),
        )

    def upsert_conversation_segment(
        self,
        *,
        conversation_id: str,
        user_id: str,
        group_id: str | None,
        segment_seq: int,
        turns_markdown: str,
        last_query: str,
        turn_count: int,
        start_time: int,
        last_time: int,
    ) -> None:
        now = int(time.time())
        self.engine.execute(
            """
            INSERT INTO conversation_segment(
              conversation_id,user_id,group_id,segment_seq,turns_markdown,last_query,turn_count,start_time,last_time,updated_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(conversation_id) DO UPDATE SET
              user_id=excluded.user_id,
              group_id=excluded.group_id,
              segment_seq=excluded.segment_seq,
              turns_markdown=excluded.turns_markdown,
              last_query=excluded.last_query,
              turn_count=excluded.turn_count,
              start_time=excluded.start_time,
              last_time=excluded.last_time,
              updated_at=excluded.updated_at
            """,
            (
                conversation_id,
                user_id,
                group_id,
                int(segment_seq),
                turns_markdown,
                last_query,
                int(turn_count),
                int(start_time),
                int(last_time),
                now,
            ),
        )

    def delete_conversation_segment(self, conversation_id: str) -> None:
        self.engine.execute(
            "DELETE FROM conversation_segment WHERE conversation_id=?",
            (conversation_id,),
        )

    def delete_by_event_id(self, event_id: str) -> int:
        row = self.engine.query_one(
            "SELECT id FROM episodic_memory WHERE event_id=? AND is_deleted=0",
            (event_id,),
        )
        affected = self.engine.execute(
            "UPDATE episodic_memory SET is_deleted=1, updated_at=? WHERE event_id=? AND is_deleted=0",
            (int(time.time()), event_id),
        )
        memory_id = str(row.get("id", "")).strip() if row else ""
        if affected > 0 and memory_id:
            try:
                self.engine.execute(
                    "DELETE FROM memory_keyword_fts WHERE memory_id=?",
                    (memory_id,),
                )
            except Exception:
                pass
            try:
                self.engine.execute(
                    "DELETE FROM event_log_keyword_fts WHERE memory_id=?",
                    (memory_id,),
                )
            except Exception:
                pass
            try:
                self.engine.execute(
                    "DELETE FROM foresight_keyword_fts WHERE memory_id=?",
                    (memory_id,),
                )
            except Exception:
                pass
        return affected

    def _sync_keyword_index(
        self,
        *,
        memory_id: str,
        user_id: str,
        group_id: str | None,
        episode: str,
        summary: str,
        subject: str,
        atomic_facts: list[str],
        memory_category: str,
        storage_tier: str,
    ) -> None:
        category = self._normalize_category(memory_category)
        tier = self._normalize_token_tag(storage_tier, fallback="text_only")
        search_text = " ".join(
            [
                str(episode or "").strip(),
                str(summary or "").strip(),
                str(subject or "").strip(),
                " ".join(str(x).strip() for x in atomic_facts if str(x).strip()),
                f"cat_{category}",
                f"tier_{tier}",
            ]
        ).strip()
        if not search_text:
            return
        try:
            self.engine.execute(
                "DELETE FROM memory_keyword_fts WHERE memory_id=?",
                (memory_id,),
            )
            self.engine.execute(
                """
                INSERT INTO memory_keyword_fts(memory_id,user_id,group_id,search_text)
                VALUES(?,?,?,?)
                """,
                (memory_id, user_id, str(group_id or ""), search_text),
            )
        except Exception:
            return

    def _sync_event_log_keyword_index(self, *, memory_id: str) -> None:
        mid = str(memory_id or "").strip()
        if not mid:
            return
        try:
            self.engine.execute(
                "DELETE FROM event_log_keyword_fts WHERE memory_id=?",
                (mid,),
            )
            self.engine.execute(
                """
                INSERT INTO event_log_keyword_fts(event_log_id,memory_id,user_id,group_id,fact_text)
                SELECT
                  CAST(e.id AS TEXT),
                  e.memory_id,
                  e.user_id,
                  COALESCE(e.group_id, ''),
                  trim(COALESCE(e.fact, ''))
                FROM memory_event_log e
                WHERE e.memory_id=?
                  AND COALESCE(trim(e.fact), '')<>''
                """,
                (mid,),
            )
        except Exception:
            return

    def _sync_foresight_keyword_index(self, *, memory_id: str) -> None:
        mid = str(memory_id or "").strip()
        if not mid:
            return
        try:
            self.engine.execute(
                "DELETE FROM foresight_keyword_fts WHERE memory_id=?",
                (mid,),
            )
            self.engine.execute(
                """
                INSERT INTO foresight_keyword_fts(foresight_id,memory_id,user_id,group_id,foresight_text)
                SELECT
                  CAST(f.id AS TEXT),
                  f.memory_id,
                  m.user_id,
                  COALESCE(m.group_id, ''),
                  trim(COALESCE(f.content, ''))
                FROM memory_foresight f
                JOIN episodic_memory m ON m.id = f.memory_id
                WHERE f.memory_id=?
                  AND m.is_deleted=0
                  AND COALESCE(trim(f.content), '')<>''
                """,
                (mid,),
            )
        except Exception:
            return
