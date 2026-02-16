from __future__ import annotations

import json
import math
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Protocol


class EmbeddingLike(Protocol):
    def embed(self, text: str) -> list[float]:
        ...


@dataclass(frozen=True)
class ConsolidateInput:
    user_id: str
    group_id: str | None
    event_id: str
    memory_id: str
    timestamp: int
    episode: str
    summary: str
    importance_score: float
    atomic_facts: list[str]
    profile_patch: dict[str, Any]
    storage_tier: str


class SemanticConsolidator:
    def __init__(
        self,
        repo,
        embedding_provider: EmbeddingLike,
        *,
        clustering_threshold: float = 0.70,
        max_time_gap_days: int = 7,
    ) -> None:
        self.repo = repo
        self.embedding_provider = embedding_provider
        self.clustering_threshold = max(0.0, min(1.0, float(clustering_threshold)))
        self.max_time_gap_sec = max(1, int(max_time_gap_days)) * 86400

    def consolidate(self, payload: ConsolidateInput) -> dict[str, Any]:
        scene_text = (payload.summary or payload.episode or "").strip()
        if not scene_text:
            return {"scene_id": None}
        vector = self._safe_embed(scene_text)
        scene = self._match_scene(
            user_id=payload.user_id,
            group_id=payload.group_id,
            timestamp=payload.timestamp,
            scene_text=scene_text,
            vector=vector,
        )
        if scene is None:
            scene_id = uuid.uuid4().hex
            scene_summary = self._merge_scene_summary("", scene_text)
            self.repo.create_memscene(
                scene_id=scene_id,
                user_id=payload.user_id,
                group_id=payload.group_id,
                summary=scene_summary,
                centroid_vector_json=json.dumps(vector or [], ensure_ascii=False),
                last_memory_ts=payload.timestamp,
            )
            memory_count_delta = 1
        else:
            scene_id = str(scene.get("id"))
            prev_vec = self._safe_vector(scene.get("centroid_vector_json"))
            old_count = int(scene.get("memory_count") or 0)
            merged_vec = self._incremental_centroid(prev_vec, vector, old_count)
            scene_summary = self._merge_scene_summary(
                str(scene.get("summary", "")), scene_text
            )
            self.repo.update_memscene(
                scene_id=scene_id,
                summary=scene_summary,
                centroid_vector_json=json.dumps(merged_vec, ensure_ascii=False),
                memory_count_delta=1,
                last_memory_ts=payload.timestamp,
            )
            memory_count_delta = 1

        self.repo.update_memory_scene(
            memory_id=payload.memory_id,
            scene_id=scene_id,
            storage_tier=payload.storage_tier,
        )

        self._evolve_profile(
            payload=payload,
            scene_id=scene_id,
            scene_summary=scene_summary,
            memory_count_delta=memory_count_delta,
        )
        return {"scene_id": scene_id}

    def _match_scene(
        self,
        *,
        user_id: str,
        group_id: str | None,
        timestamp: int,
        scene_text: str,
        vector: list[float],
    ) -> dict[str, Any] | None:
        candidates = self.repo.get_memscene_candidates(
            user_id=user_id, group_id=group_id, limit=120
        )
        best: dict[str, Any] | None = None
        best_score = -1.0
        for scene in candidates:
            last_ts = int(scene.get("last_memory_ts") or 0)
            if last_ts > 0 and abs(int(timestamp) - last_ts) > self.max_time_gap_sec:
                continue
            centroid = self._safe_vector(scene.get("centroid_vector_json"))
            vector_score = self._cosine(vector, centroid) if vector and centroid else 0.0
            lexical_score = self._lexical_overlap(scene_text, str(scene.get("summary", "")))
            score = max(vector_score, lexical_score * 0.9)
            if score > best_score:
                best_score = score
                best = scene
        if best is None:
            return None
        # Vector match is primary. Lexical fallback prevents complete scene fragmentation.
        if best_score < self.clustering_threshold and best_score < 0.25:
            return None
        return best

    def _evolve_profile(
        self,
        *,
        payload: ConsolidateInput,
        scene_id: str,
        scene_summary: str,
        memory_count_delta: int,
    ) -> None:
        prev = self.repo.get_latest_profile_snapshot(
            user_id=payload.user_id, group_id=payload.group_id
        )
        profile = (
            dict(prev.get("profile", {}))
            if prev and isinstance(prev.get("profile"), dict)
            else {}
        )
        explicit = profile.get("explicit_facts")
        if not isinstance(explicit, dict):
            explicit = {}
        implicit = profile.get("implicit_traits")
        if not isinstance(implicit, list):
            implicit = []
        conflicts = profile.get("conflicts")
        if not isinstance(conflicts, list):
            conflicts = []

        extracted_explicit = self._extract_explicit(scene_summary, payload.timestamp)
        extracted_implicit = self._extract_implicit(scene_summary)
        patch = payload.profile_patch if isinstance(payload.profile_patch, dict) else {}
        if patch:
            for k, v in patch.items():
                key = str(k).strip()
                val = str(v).strip()
                if key and val:
                    extracted_explicit.setdefault(key, {"value": val, "timestamp": payload.timestamp})

        for field, new_obj in extracted_explicit.items():
            old_obj = explicit.get(field)
            if old_obj and isinstance(old_obj, dict):
                old_value = str(old_obj.get("value", "")).strip()
                new_value = str(new_obj.get("value", "")).strip()
                if old_value and new_value and old_value != new_value:
                    if self._is_time_varying_field(field):
                        explicit[field] = {
                            "value": new_value,
                            "timestamp": int(new_obj.get("timestamp") or payload.timestamp),
                            "previous_value": old_value,
                            "trend": self._calc_trend(old_value, new_value),
                        }
                    else:
                        conflict = {
                            "field": field,
                            "old": old_value,
                            "new": new_value,
                            "timestamp": payload.timestamp,
                            "scene_id": scene_id,
                        }
                        conflicts.append(conflict)
                        self.repo.insert_profile_conflict(
                            conflict_id=uuid.uuid4().hex,
                            user_id=payload.user_id,
                            group_id=payload.group_id,
                            field_name=field,
                            old_value=old_value,
                            new_value=new_value,
                            happened_at=payload.timestamp,
                            evidence_event_id=payload.event_id,
                        )
                        if int(new_obj.get("timestamp") or payload.timestamp) >= int(
                            old_obj.get("timestamp") or 0
                        ):
                            explicit[field] = new_obj
            else:
                explicit[field] = new_obj

        for item in extracted_implicit:
            if item and item not in implicit:
                implicit.append(item)
        implicit = implicit[-12:]
        conflicts = conflicts[-24:]

        profile["explicit_facts"] = explicit
        profile["implicit_traits"] = implicit
        profile["conflicts"] = conflicts
        profile["scene_stats"] = {
            "last_scene_id": scene_id,
            "last_scene_summary": scene_summary[:300],
            "updated_at": payload.timestamp,
            "memory_count_delta": memory_count_delta,
        }
        self.repo.upsert_profile_snapshot(
            event_id=payload.event_id,
            user_id=payload.user_id,
            group_id=payload.group_id,
            profile_patch=profile,
            timestamp=payload.timestamp,
        )

    def _extract_explicit(self, text: str, timestamp: int) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        raw = text.strip()
        if not raw:
            return out

        patterns = [
            ("weight", r"(?:体重|weight)\D{0,6}(\d+(?:\.\d+)?)\s*(kg|公斤)?"),
            ("waist", r"(?:腰围|waist)\D{0,6}(\d+(?:\.\d+)?)\s*(cm|厘米)?"),
            ("blood_pressure", r"(?:血压|blood pressure)\D{0,6}(\d{2,3}/\d{2,3})"),
            ("age", r"(?:今年|年龄|age)\D{0,6}(\d{1,3})\s*(?:岁|years?)?"),
            ("city", r"(?:住在|在)([\u4e00-\u9fa5A-Za-z·\-\s]{2,24})(?:工作|上班|生活|读书)?"),
            ("company", r"(?:在|于)([\u4e00-\u9fa5A-Za-z0-9&\-\s]{2,36})(?:公司|团队|集团)(?:工作|任职)?"),
            ("profession", r"(?:我是|职业是|做)([\u4e00-\u9fa5A-Za-z0-9&\-\s]{2,24})(?:工程师|老师|医生|设计师|产品经理|销售|运营|顾问)?"),
        ]
        for key, pattern in patterns:
            m = re.search(pattern, raw, re.IGNORECASE)
            if not m:
                continue
            value = str(m.group(1)).strip()
            unit = str(m.group(2)).strip() if m.lastindex and m.lastindex >= 2 and m.group(2) else ""
            out[key] = {
                "value": f"{value}{(' ' + unit) if unit else ''}".strip(),
                "timestamp": timestamp,
            }

        m = re.search(r"我叫([^\s，。；;！？!?]{1,32})", raw)
        if m:
            out["name"] = {"value": m.group(1).strip(), "timestamp": timestamp}
        likes = re.findall(r"(?:喜欢|偏好)([\u4e00-\u9fa5A-Za-z0-9、,\s]{1,40})", raw)
        if likes:
            out["preferences"] = {"value": " / ".join([x.strip() for x in likes[:3] if x.strip()]), "timestamp": timestamp}
        return out

    def _extract_implicit(self, text: str) -> list[str]:
        raw = text.lower()
        traits: list[str] = []
        keyword_map = {
            "goal_oriented": ["计划", "目标", "坚持", "持续", "improve", "goal"],
            "health_conscious": ["健康", "体重", "腰围", "锻炼", "饮食", "diet", "exercise"],
            "actionable_preference": ["建议", "具体", "可执行", "方案", "步骤", "actionable"],
            "social_preference": ["朋友", "家人", "一起", "聚会"],
            "detail_oriented": ["细节", "具体", "精确", "详细", "度量"],
            "time_sensitive": ["马上", "尽快", "截止", "deadline", "今晚", "明天"],
            "learning_oriented": ["学习", "复盘", "总结", "提升", "课程", "读书"],
        }
        for trait, keys in keyword_map.items():
            if any(k in raw for k in keys):
                traits.append(trait)
        return traits

    @staticmethod
    def _merge_scene_summary(old: str, new: str) -> str:
        old = old.strip()
        new = new.strip()
        if not old:
            return new[:400]
        if not new:
            return old[:400]
        merged = f"{old}\n{new}"
        merged_lines: list[str] = []
        seen: set[str] = set()
        for line in merged.splitlines():
            t = line.strip()
            if not t:
                continue
            key = t.lower()
            if key in seen:
                continue
            seen.add(key)
            merged_lines.append(t)
            if len(merged_lines) >= 8:
                break
        return " | ".join(merged_lines)[:400]

    @staticmethod
    def _safe_vector(raw: Any) -> list[float]:
        if isinstance(raw, list):
            vals = raw
        elif isinstance(raw, str):
            try:
                vals = json.loads(raw)
            except Exception:
                vals = []
        else:
            vals = []
        out: list[float] = []
        for v in vals:
            try:
                out.append(float(v))
            except Exception:
                continue
        return out

    def _safe_embed(self, text: str) -> list[float]:
        try:
            vec = self.embedding_provider.embed(text)
        except Exception:
            return []
        return self._safe_vector(vec)

    @staticmethod
    def _incremental_centroid(
        old_vec: list[float], new_vec: list[float], old_count: int
    ) -> list[float]:
        if not old_vec:
            return new_vec
        size = min(len(old_vec), len(new_vec))
        if size <= 0:
            return new_vec or old_vec
        count = max(1, int(old_count))
        out = [
            (old_vec[i] * count + new_vec[i]) / (count + 1)
            for i in range(size)
        ]
        return out

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        if not a or not b:
            return 0.0
        size = min(len(a), len(b))
        if size <= 0:
            return 0.0
        dot = 0.0
        na = 0.0
        nb = 0.0
        for i in range(size):
            av = float(a[i])
            bv = float(b[i])
            dot += av * bv
            na += av * av
            nb += bv * bv
        if na <= 0 or nb <= 0:
            return 0.0
        return dot / (math.sqrt(na) * math.sqrt(nb))

    @staticmethod
    def _lexical_overlap(a: str, b: str) -> float:
        ta = SemanticConsolidator._tokenize_for_overlap(a)
        tb = SemanticConsolidator._tokenize_for_overlap(b)
        if not ta or not tb:
            return 0.0
        return float(len(ta & tb)) / float(max(1, len(ta | tb)))

    @staticmethod
    def _tokenize_for_overlap(text: str) -> set[str]:
        raw = text.lower().strip()
        if not raw:
            return set()
        tokens = {t for t in re.split(r"[\s,，。；;：:！!？?\-|_/]+", raw) if t}
        # Chinese bi-gram tokens improve scene similarity when no spaces are present.
        chars = re.findall(r"[\u4e00-\u9fff]", raw)
        for i in range(len(chars) - 1):
            tokens.add(chars[i] + chars[i + 1])
        return tokens

    @staticmethod
    def _is_time_varying_field(field: str) -> bool:
        return field in {"weight", "waist", "blood_pressure", "heart_rate", "sleep_hours"}

    @staticmethod
    def _calc_trend(old_value: str, new_value: str) -> str:
        old_num = _extract_first_number(old_value)
        new_num = _extract_first_number(new_value)
        if old_num is None or new_num is None:
            return "changed"
        if new_num > old_num:
            return "up"
        if new_num < old_num:
            return "down"
        return "stable"


def _extract_first_number(text: str) -> float | None:
    m = re.search(r"-?\d+(?:\.\d+)?", text)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None
