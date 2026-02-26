from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

_SESSION_KEY_RE = re.compile(r"^session_(\d+)$")
_SESSION_DT_RE = re.compile(r"^session_(\d+)_date_time$")
_DIA_ID_RE = re.compile(r"^[Dd](\d+):0*(\d+)$")
_DIA_ID_IN_TEXT_RE = re.compile(r"[Dd]\s*:?\s*\d+\s*:\s*\d+")


def _load_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8-sig") as f:
            for i, line in enumerate(f, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    row = json.loads(text)
                except Exception as exc:
                    raise ValueError(f"Invalid JSONL at line {i}: {exc}") from exc
                if isinstance(row, dict):
                    rows.append(row)
        return rows

    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]
    if isinstance(raw, dict):
        for key in ("data", "items", "records", "examples"):
            val = raw.get(key)
            if isinstance(val, list):
                return [x for x in val if isinstance(x, dict)]
        return [raw]
    raise ValueError("Input must be a JSON object, list, or JSONL records")


def _as_str(value: Any) -> str:
    text = str(value or "").strip()
    return text


def _as_str_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [_as_str(x) for x in value if _as_str(x)]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if "," in text:
            return [_as_str(x) for x in text.split(",") if _as_str(x)]
        return [text]
    return []


def _normalize_message_id(value: Any) -> str:
    token = _as_str(value)
    if not token:
        return ""
    token = re.sub(r"\s+", "", token)
    token = re.sub(r"^[Dd]:", "D", token)
    token = token.replace("\u3000", " ")
    m = _DIA_ID_RE.match(token)
    if m:
        return f"D{int(m.group(1))}:{int(m.group(2))}"
    return token


def _as_message_id_list(value: Any) -> list[str]:
    tokens: list[str] = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str):
                matched = _DIA_ID_IN_TEXT_RE.findall(item)
                if matched:
                    tokens.extend(matched)
                else:
                    parts = [x for x in re.split(r"[,;\s]+", item.strip()) if x]
                    tokens.extend(parts if parts else [item.strip()])
            else:
                text = _as_str(item)
                if text:
                    tokens.append(text)
    elif isinstance(value, str):
        text = value.strip()
        if text:
            matched = _DIA_ID_IN_TEXT_RE.findall(text)
            if matched:
                tokens.extend(matched)
            else:
                parts = [x for x in re.split(r"[,;\s]+", text) if x]
                tokens.extend(parts if parts else [text])
    else:
        return []

    out: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        norm = _normalize_message_id(token)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(norm)
    return out


def _pick_text(obj: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        if key in obj:
            text = _as_str(obj.get(key))
            if text:
                return text
    return ""


def _build_memories(
    *,
    turns: Any,
    user_id: str,
    base_ts: int,
    turn_id_field: str,
    turn_text_field: str,
    turn_speaker_field: str,
    turn_time_field: str,
) -> list[dict[str, Any]]:
    if not isinstance(turns, list):
        return []
    out: list[dict[str, Any]] = []
    for idx, turn in enumerate(turns, start=1):
        if not isinstance(turn, dict):
            continue
        turn_id = _pick_text(turn, [turn_id_field, "id", "message_id", "turn_id"])
        if not turn_id:
            turn_id = f"turn-{idx}"
        turn_id = _normalize_message_id(turn_id)
        content = _pick_text(
            turn, [turn_text_field, "content", "text", "utterance", "message", "episode"]
        )
        if not content:
            continue
        sender = _pick_text(turn, [turn_speaker_field, "sender", "speaker", "role"]) or user_id
        create_time: Any = turn.get(turn_time_field)
        if create_time is None or (isinstance(create_time, str) and not create_time.strip()):
            create_time = int(base_ts + idx)
        out.append(
            {
                "message_id": turn_id,
                "content": content,
                "sender": sender,
                "create_time": create_time,
            }
        )
    return out


def _parse_locomo_datetime(text: str) -> int | None:
    raw = _as_str(text)
    if not raw:
        return None
    formats = (
        "%I:%M %p on %d %B, %Y",
        "%I:%M %p on %d %b, %Y",
        "%H:%M on %d %B, %Y",
        "%H:%M on %d %b, %Y",
    )
    for fmt in formats:
        try:
            return int(datetime.strptime(raw, fmt).timestamp())
        except Exception:
            continue
    return None


def _build_memories_from_locomo(
    *,
    conversation: dict[str, Any],
    default_sender: str,
    base_ts: int,
) -> list[dict[str, Any]]:
    sessions: list[tuple[int, list[dict[str, Any]]]] = []
    session_ts: dict[int, int] = {}
    for key, value in conversation.items():
        if not isinstance(key, str):
            continue
        m_dt = _SESSION_DT_RE.match(key)
        if m_dt:
            sid = int(m_dt.group(1))
            parsed = _parse_locomo_datetime(_as_str(value))
            if parsed is not None:
                session_ts[sid] = parsed
            continue
        m_session = _SESSION_KEY_RE.match(key)
        if m_session and isinstance(value, list):
            sid = int(m_session.group(1))
            turns = [x for x in value if isinstance(x, dict)]
            sessions.append((sid, turns))

    sessions.sort(key=lambda x: x[0])
    out: list[dict[str, Any]] = []
    for sid, turns in sessions:
        base = session_ts.get(sid, int(base_ts + sid * 3600))
        for idx, turn in enumerate(turns, start=1):
            message_id = _pick_text(turn, ["dia_id", "turn_id", "message_id", "id"])
            if not message_id:
                message_id = f"D{sid}:{idx}"
            message_id = _normalize_message_id(message_id)
            content = _pick_text(turn, ["text", "content", "utterance", "message", "episode"])
            if not content:
                continue
            sender = _pick_text(turn, ["speaker", "sender", "role"]) or default_sender
            out.append(
                {
                    "message_id": message_id,
                    "content": content,
                    "sender": sender,
                    "create_time": int(base + idx),
                }
            )
    return out


def convert_records(
    *,
    rows: list[dict[str, Any]],
    case_id_field: str,
    user_id_field: str,
    group_id_field: str,
    conversation_field: str,
    qa_field: str,
    question_field: str,
    support_field: str,
    turn_id_field: str,
    turn_text_field: str,
    turn_speaker_field: str,
    turn_time_field: str,
    default_user_id: str,
    default_group_id: str,
    base_timestamp: int,
    keep_empty_support: bool,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    out: list[dict[str, Any]] = []
    stats = {"input_rows": len(rows), "output_rows": 0, "skipped_rows": 0}
    for row_idx, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            stats["skipped_rows"] += 1
            continue

        case_id_base = _pick_text(row, [case_id_field, "sample_id", "id", "case_id"]) or f"locomo-{row_idx}"
        conversation_obj = row.get(conversation_field) or row.get("conversation")
        speaker_a = (
            _pick_text(conversation_obj, ["speaker_a"])
            if isinstance(conversation_obj, dict)
            else ""
        )
        user_id = _pick_text(row, [user_id_field, "user_id", "uid"]) or speaker_a or default_user_id
        group_id = _pick_text(row, [group_id_field, "group_id", "gid"]) or case_id_base or default_group_id

        # Support pass-through records that are already in "one QA per row" format.
        if _pick_text(row, [question_field, "query", "question"]) and isinstance(row.get("memories"), list):
            expected_msg_ids = _as_str_list(
                row.get("expected_message_ids")
                or row.get("supporting_message_ids")
                or row.get("supporting_turn_ids")
            )
            expected_msg_ids = _as_message_id_list(expected_msg_ids)
            if expected_msg_ids or keep_empty_support:
                item = {
                    "case_id": case_id_base,
                    "user_id": user_id,
                    "group_id": group_id,
                    "query": _pick_text(row, [question_field, "query", "question"]),
                    "expected_message_ids": expected_msg_ids,
                    "memories": row.get("memories"),
                }
                out.append(item)
            else:
                stats["skipped_rows"] += 1
            continue

        turns = row.get(conversation_field) or row.get("memories") or row.get("conversation")
        if isinstance(turns, dict):
            memories = _build_memories_from_locomo(
                conversation=turns,
                default_sender=user_id,
                base_ts=base_timestamp,
            )
        else:
            memories = _build_memories(
                turns=turns,
                user_id=user_id,
                base_ts=base_timestamp,
                turn_id_field=turn_id_field,
                turn_text_field=turn_text_field,
                turn_speaker_field=turn_speaker_field,
                turn_time_field=turn_time_field,
            )
        if not memories:
            stats["skipped_rows"] += 1
            continue

        qa_items = row.get(qa_field)
        if isinstance(qa_items, list):
            for q_idx, qa in enumerate(qa_items, start=1):
                if not isinstance(qa, dict):
                    continue
                query = _pick_text(qa, [question_field, "question", "query"])
                if not query:
                    continue
                expected_msg_ids = _as_str_list(
                    qa.get(support_field)
                    or qa.get("evidence")
                    or qa.get("supporting_turn_ids")
                    or qa.get("supporting_message_ids")
                    or qa.get("expected_message_ids")
                )
                expected_msg_ids = _as_message_id_list(expected_msg_ids)
                if not expected_msg_ids and not keep_empty_support:
                    continue
                out.append(
                    {
                        "case_id": f"{case_id_base}-q{q_idx}",
                        "user_id": user_id,
                        "group_id": group_id,
                        "query": query,
                        "answer": qa.get("answer") if isinstance(qa.get("answer"), (str, int, float)) else qa.get("adversarial_answer"),
                        "category": qa.get("category"),
                        "expected_message_ids": expected_msg_ids,
                        "memories": memories,
                    }
                )
            continue

        # Single QA record with direct fields.
        query = _pick_text(row, [question_field, "question", "query"])
        expected_msg_ids = _as_str_list(
            row.get(support_field)
            or row.get("evidence")
            or row.get("supporting_turn_ids")
            or row.get("supporting_message_ids")
            or row.get("expected_message_ids")
        )
        expected_msg_ids = _as_message_id_list(expected_msg_ids)
        if query and (expected_msg_ids or keep_empty_support):
            out.append(
                {
                    "case_id": case_id_base,
                    "user_id": user_id,
                    "group_id": group_id,
                    "query": query,
                    "expected_message_ids": expected_msg_ids,
                    "memories": memories,
                }
            )
        else:
            stats["skipped_rows"] += 1

    stats["output_rows"] = len(out)
    return out, stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert LoCoMo-style raw data into MiniMem LoCoMo eval JSONL."
    )
    parser.add_argument("--input", required=True, help="Raw dataset path (.json or .jsonl)")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--case-id-field", default="case_id")
    parser.add_argument("--user-id-field", default="user_id")
    parser.add_argument("--group-id-field", default="group_id")
    parser.add_argument("--conversation-field", default="conversation")
    parser.add_argument("--qa-field", default="qa")
    parser.add_argument("--question-field", default="question")
    parser.add_argument("--support-field", default="supporting_turn_ids")
    parser.add_argument("--turn-id-field", default="turn_id")
    parser.add_argument("--turn-text-field", default="text")
    parser.add_argument("--turn-speaker-field", default="speaker")
    parser.add_argument("--turn-time-field", default="timestamp")
    parser.add_argument("--default-user-id", default="locomo_user")
    parser.add_argument("--default-group-id", default="locomo_group")
    parser.add_argument(
        "--base-timestamp",
        type=int,
        default=int(time.time()) - 86400,
        help="Fallback timestamp for turns missing time fields.",
    )
    parser.add_argument(
        "--keep-empty-support",
        action="store_true",
        help="Keep rows even if supporting ids are empty.",
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    if not input_path.exists():
        raise SystemExit(f"input not found: {input_path}")

    rows = _load_records(input_path)
    converted, stats = convert_records(
        rows=rows,
        case_id_field=args.case_id_field,
        user_id_field=args.user_id_field,
        group_id_field=args.group_id_field,
        conversation_field=args.conversation_field,
        qa_field=args.qa_field,
        question_field=args.question_field,
        support_field=args.support_field,
        turn_id_field=args.turn_id_field,
        turn_text_field=args.turn_text_field,
        turn_speaker_field=args.turn_speaker_field,
        turn_time_field=args.turn_time_field,
        default_user_id=args.default_user_id,
        default_group_id=args.default_group_id,
        base_timestamp=args.base_timestamp,
        keep_empty_support=bool(args.keep_empty_support),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for item in converted:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    report = {
        "status": "ok",
        "input": str(input_path),
        "output": str(output_path),
        **stats,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
