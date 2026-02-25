from __future__ import annotations

import json
import re
import time
import uuid
from collections import defaultdict
from typing import Any

from evermemos_lite.infra.sqlite.db import SQLiteEngine


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
        event_id: str | None = None,
    ) -> dict[str, Any]:
        eid = event_id or uuid.uuid4().hex
        mid = uuid.uuid4().hex
        now = int(time.time())

        self.engine.execute(
            """
            INSERT INTO episodic_memory(
                id,event_id,source_message_id,user_id,group_id,timestamp,role,sender,sender_name,
                group_name,episode,summary,subject,importance_score,scene_id,storage_tier,created_at,updated_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
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
                str(storage_tier or "text_only"),
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
        )
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
            "storage_tier": str(storage_tier or "text_only"),
        }

    def fetch_episodes(
        self,
        *,
        user_id: str | None,
        group_id: str | None,
        limit: int = 40,
        candidate_episode_ids: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        sql = """
            SELECT
              m.id,
              m.event_id,
              m.user_id,
              m.group_id,
              m.timestamp,
              m.episode,
              m.summary,
              m.subject,
              m.importance_score,
              m.scene_id,
              m.storage_tier,
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
              m.group_id,
              m.timestamp,
              m.episode,
              m.summary,
              m.subject,
              m.importance_score,
              m.scene_id,
              m.storage_tier
            FROM episodic_memory m
            WHERE m.is_deleted=0
              AND m.id IN ({placeholders})
            """,
            unique_ids,
        )
        order = {eid: idx for idx, eid in enumerate(unique_ids)}
        rows.sort(key=lambda x: order.get(str(x.get("id")), 1_000_000))
        return rows

    def search_keyword(
        self,
        *,
        query: str,
        user_id: str | None,
        group_id: str | None,
        top_k: int,
        candidate_episode_ids: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        fts_hits = self._search_keyword_fts(
            query=query,
            user_id=user_id,
            group_id=group_id,
            top_k=top_k,
            candidate_episode_ids=candidate_episode_ids,
        )
        if fts_hits:
            return fts_hits
        return self._search_keyword_scan(
            query=query,
            user_id=user_id,
            group_id=group_id,
            top_k=top_k,
            candidate_episode_ids=candidate_episode_ids,
        )

    def _search_keyword_scan(
        self,
        *,
        query: str,
        user_id: str | None,
        group_id: str | None,
        top_k: int,
        candidate_episode_ids: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        rows = self.fetch_episodes(
            user_id=user_id,
            group_id=group_id,
            limit=max(800, top_k * 40),
            candidate_episode_ids=candidate_episode_ids,
        )
        terms = self._tokenize_text(query)
        if not terms:
            terms = [query.lower()]
        scored: list[dict[str, Any]] = []
        for row in rows:
            raw_text = " ".join(
                [
                    str(row.get("episode", "")),
                    str(row.get("summary", "")),
                    str(row.get("subject", "")),
                    str(row.get("atomic_fact_text", "")),
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
    ) -> list[dict[str, Any]]:
        terms = self._tokenize_text(query)
        if not terms:
            terms = [query.lower()]
        dedup_terms = list(dict.fromkeys(t for t in terms if t.strip()))
        if not dedup_terms:
            return []
        # Keep query plan bounded on device-side workloads.
        selected = dedup_terms[:24]
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
            for term in dedup_terms:
                lexical += float(raw_text.count(term))
            bm25_raw = row.get("bm25_score")
            try:
                bm25_val = float(bm25_raw if bm25_raw is not None else 0.0)
            except Exception:
                bm25_val = 0.0
            fts_score = 1.0 / (1.0 + max(0.0, bm25_val))
            scored.append(
                {
                    "memory_id": row["memory_id"],
                    "bm25_score": bm25_val,
                    "score": float(lexical * 1.15 + fts_score),
                    "source": "keyword_fts",
                }
            )
        scored.sort(key=lambda x: float(x["score"]), reverse=True)
        return scored[: max(1, int(top_k))]

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
        stop = {"我", "你", "他", "她", "它", "吗", "呢", "啊", "呀", "的", "了"}
        return [t for t in tokens if t and t not in stop]

    @staticmethod
    def _escape_fts_term(term: str) -> str:
        return str(term or "").replace('"', '""')

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
    ) -> None:
        search_text = " ".join(
            [
                str(episode or "").strip(),
                str(summary or "").strip(),
                str(subject or "").strip(),
                " ".join(str(x).strip() for x in atomic_facts if str(x).strip()),
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
