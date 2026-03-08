from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import statistics
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from flockmem.bootstrap.app_factory import create_app
from flockmem.config.settings import LiteSettings


DOCS: list[dict[str, Any]] = [
    {
        "id": "d1",
        "title": "Q3 launch note",
        "text": "项目周会记录。我们在上海确认了Q3发布时间，灰度窗口是9月18日，回滚开关命名为 ROLLBACK_SWITCH_ALPHA。上线前要完成压测和告警演练。",
        "query": "Q3灰度窗口是几号？",
        "expect": ["9月18日", "ROLLBACK_SWITCH_ALPHA"],
    },
    {
        "id": "d2",
        "title": "Data pipeline incident",
        "text": "故障复盘。根因是ETL任务重复消费，修复提交号是 fix/etl-dedup-4421，最终在22:40恢复。补偿脚本名 backfill_orders_202603。",
        "query": "这次ETL故障的修复分支是什么？",
        "expect": ["fix/etl-dedup-4421", "backfill_orders_202603"],
    },
    {
        "id": "d3",
        "title": "Security checklist",
        "text": "安全清单。生产密钥轮换周期改为45天，审计任务ID为 SEC-AUDIT-771。外部供应商访问采用最小权限和双人审批。",
        "query": "审计任务ID是什么？",
        "expect": ["SEC-AUDIT-771", "45天"],
    },
    {
        "id": "d4",
        "title": "Mobile release",
        "text": "移动端发布计划。iOS构建号 3.9.12(812)，Android构建号 3.9.12(9012)。发布负责人是Lina，冻结窗口周四18:00。",
        "query": "iOS构建号是多少？",
        "expect": ["3.9.12(812)", "Lina"],
    },
    {
        "id": "d5",
        "title": "Cost optimization",
        "text": "成本优化会议。对象存储分层后月成本下降17%，关键动作是归档策略 policy/archive-cold-30d。预计两周内覆盖全部项目。",
        "query": "归档策略名称是什么？",
        "expect": ["policy/archive-cold-30d", "下降17%"],
    },
    {
        "id": "d6",
        "title": "Oncall playbook",
        "text": "值班手册。夜间告警先执行 runbook#P1-redis-latency，10分钟内无改善再升级到SRE。升级联系人群组为 sre-escalation-cn。",
        "query": "夜间Redis延迟要先执行哪个runbook？",
        "expect": ["runbook#P1-redis-latency", "sre-escalation-cn"],
    },
    {
        "id": "d7",
        "title": "Contract memo",
        "text": "法务备忘。合同附录B第7条新增数据保留上限180天，例外审批单号 LEGAL-EX-902。客户侧确认邮件主题为 DataRetentionUpdate。",
        "query": "例外审批单号是多少？",
        "expect": ["LEGAL-EX-902", "180天"],
    },
    {
        "id": "d8",
        "title": "ML eval note",
        "text": "模型评估。A/B中B方案F1=0.843，高于A方案0.817。最终采用配置实验名 exp-rerank-b-20260305。后续关注冷启动样本。",
        "query": "最终采用的实验名是什么？",
        "expect": ["exp-rerank-b-20260305", "0.843"],
    },
]


@contextmanager
def _temporary_env(patch: dict[str, str]) -> Any:
    old_values: dict[str, str | None] = {k: os.environ.get(k) for k in patch}
    try:
        for key, value in patch.items():
            os.environ[key] = value
        yield
    finally:
        for key, old in old_values.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * p))
    idx = max(0, min(len(ordered) - 1, idx))
    return float(ordered[idx])


def _make_client(temp_root: Path) -> TestClient:
    env_patch = {
        "LITE_DATA_DIR": str(temp_root / "data"),
        "LITE_CONFIG_DIR": str(temp_root / "cfg"),
        "LITE_EXTRACTOR_PROVIDER": "rule",
        "LITE_EMBEDDING_PROVIDER": "local",
        "LITE_EMBEDDING_MODEL": "local-hash-384",
        "LITE_SKILL_ADAPTER_ENABLED": "true",
        "LITE_SKILL_ADAPTER_WHITELIST": "markitdown,pdf,pptx",
    }
    with _temporary_env(env_patch):
        app = create_app(LiteSettings.from_env())
    return TestClient(app)


def _ingest_plain(client: TestClient, docs: list[dict[str, Any]], group_id: str) -> None:
    for idx, doc in enumerate(docs, start=1):
        payload = {
            "message_id": f"plain-{doc['id']}-{idx}",
            "create_time": 1772677000 + idx,
            "sender": "eval-user",
            "content": str(doc["text"]),
            "group_id": group_id,
            "role": "user",
        }
        resp = client.post("/api/v1/memories", json=payload)
        if resp.status_code != 200:
            raise RuntimeError(f"plain ingest failed: {resp.status_code} {resp.text}")


def _ingest_skill(client: TestClient, docs: list[dict[str, Any]], group_id: str) -> None:
    for doc in docs:
        chunks = [x.strip() for x in str(doc["text"]).split("。") if x.strip()]
        payload = {
            "source_type": "pdf",
            "source_uri": f"file:///tmp/{doc['id']}.pdf",
            "summary": str(doc["title"]),
            "chunks": chunks,
            "skill_name": "pdf",
            "agent_id": "eval-agent",
            "sender": "eval-user",
            "group_id": group_id,
            "task_id": str(doc["id"]),
            "trace_id": f"trace-{doc['id']}",
        }
        resp = client.post("/api/v1/ingest/skill", json=payload)
        if resp.status_code != 200:
            raise RuntimeError(f"skill ingest failed: {resp.status_code} {resp.text}")
        accepted = (
            resp.json().get("result", {}).get("accepted")
            if isinstance(resp.json(), dict)
            else False
        )
        if accepted is not True:
            raise RuntimeError(f"skill ingest not accepted: {resp.text}")


def _is_hit(row: dict[str, Any], expected_tokens: list[str]) -> bool:
    text = " ".join(
        [
            str(row.get("content", "")),
            str(row.get("summary", "")),
            str(row.get("episode", "")),
        ]
    )
    return any(token in text for token in expected_tokens)


def _evaluate_method(
    client: TestClient,
    docs: list[dict[str, Any]],
    *,
    method: str,
    top_k: int,
    group_id: str,
) -> dict[str, Any]:
    latencies: list[float] = []
    hit1 = 0
    hitk = 0
    for doc in docs:
        started = time.perf_counter()
        resp = client.get(
            "/api/v1/memories/search",
            params={
                "query": str(doc["query"]),
                "user_id": "eval-user",
                "group_id": group_id,
                "retrieve_method": method,
                "decision_mode": "static",
                "top_k": top_k,
            },
        )
        latencies.append((time.perf_counter() - started) * 1000.0)
        if resp.status_code != 200:
            raise RuntimeError(f"search failed ({method}): {resp.status_code} {resp.text}")
        rows = resp.json().get("result", {}).get("memories", [])
        expected = [str(x) for x in doc.get("expect", [])]
        if rows and _is_hit(rows[0], expected):
            hit1 += 1
        if any(_is_hit(row, expected) for row in rows[:top_k]):
            hitk += 1
    return {
        "method": method,
        "cases": len(docs),
        "recall_at_1": round(hit1 / len(docs), 4) if docs else 0.0,
        f"recall_at_{top_k}": round(hitk / len(docs), 4) if docs else 0.0,
        "p50_ms": round(_percentile(latencies, 0.50), 3),
        "p95_ms": round(_percentile(latencies, 0.95), 3),
        "avg_ms": round(statistics.mean(latencies), 3) if latencies else 0.0,
    }


def _run_single_mode(
    *,
    mode: str,
    docs: list[dict[str, Any]],
    methods: list[str],
    top_k: int,
    group_id: str,
) -> dict[str, Any]:
    td = Path(tempfile.mkdtemp(prefix=f"skill-ablation-{mode}-"))
    client: TestClient | None = None
    try:
        client = _make_client(td)
        if mode == "plain":
            _ingest_plain(client, docs, group_id)
        elif mode == "skill":
            _ingest_skill(client, docs, group_id)
        else:
            raise ValueError(f"unknown mode: {mode}")
        return {
            method: _evaluate_method(
                client,
                docs,
                method=method,
                top_k=top_k,
                group_id=group_id,
            )
            for method in methods
        }
    finally:
        if client is not None:
            client.close()
        shutil.rmtree(td, ignore_errors=True)


def _summarize(rounds: list[dict[str, Any]], methods: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for method in methods:
        skill_r1 = [r["skill"][method]["recall_at_1"] for r in rounds]
        plain_r1 = [r["plain"][method]["recall_at_1"] for r in rounds]
        skill_p50 = [r["skill"][method]["p50_ms"] for r in rounds]
        plain_p50 = [r["plain"][method]["p50_ms"] for r in rounds]
        summary[method] = {
            "rounds": len(rounds),
            "plain_recall_at_1_avg": round(statistics.mean(plain_r1), 4) if plain_r1 else 0.0,
            "skill_recall_at_1_avg": round(statistics.mean(skill_r1), 4) if skill_r1 else 0.0,
            "recall_at_1_delta": round(
                (statistics.mean(skill_r1) - statistics.mean(plain_r1)),
                4,
            )
            if skill_r1 and plain_r1
            else 0.0,
            "plain_p50_ms_avg": round(statistics.mean(plain_p50), 3) if plain_p50 else 0.0,
            "skill_p50_ms_avg": round(statistics.mean(skill_p50), 3) if skill_p50 else 0.0,
            "p50_ms_delta": round(
                (statistics.mean(skill_p50) - statistics.mean(plain_p50)),
                3,
            )
            if skill_p50 and plain_p50
            else 0.0,
        }
    return summary


def run_eval(
    *,
    methods: list[str],
    rounds: int,
    top_k: int,
    seed: int,
) -> dict[str, Any]:
    round_reports: list[dict[str, Any]] = []
    for idx in range(rounds):
        rng = random.Random(seed + idx)
        docs = list(DOCS)
        rng.shuffle(docs)
        group_id = f"eval:skill-compare:r{idx + 1}"
        plain = _run_single_mode(
            mode="plain",
            docs=docs,
            methods=methods,
            top_k=top_k,
            group_id=group_id,
        )
        skill = _run_single_mode(
            mode="skill",
            docs=docs,
            methods=methods,
            top_k=top_k,
            group_id=group_id,
        )
        delta: dict[str, Any] = {}
        for method in methods:
            delta[method] = {
                "recall_at_1_delta": round(
                    skill[method]["recall_at_1"] - plain[method]["recall_at_1"], 4
                ),
                f"recall_at_{top_k}_delta": round(
                    skill[method][f"recall_at_{top_k}"] - plain[method][f"recall_at_{top_k}"],
                    4,
                ),
                "p50_ms_delta": round(skill[method]["p50_ms"] - plain[method]["p50_ms"], 3),
                "p95_ms_delta": round(skill[method]["p95_ms"] - plain[method]["p95_ms"], 3),
            }
        round_reports.append(
            {
                "round": idx + 1,
                "seed": seed + idx,
                "plain": plain,
                "skill": skill,
                "delta": delta,
            }
        )
    return {
        "dataset": "embedded_synthetic_docs_v1",
        "cases_per_round": len(DOCS),
        "methods": methods,
        "rounds": round_reports,
        "summary": _summarize(round_reports, methods),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run skill adapter ablation: plain memories vs /api/v1/ingest/skill."
    )
    parser.add_argument(
        "--methods",
        default="keyword,hybrid",
        help="Comma separated retrieve methods.",
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260305)
    parser.add_argument("--report-out", default="")
    args = parser.parse_args()

    methods = [m.strip() for m in str(args.methods).split(",") if m.strip()]
    report = run_eval(
        methods=methods or ["keyword", "hybrid"],
        rounds=max(1, int(args.rounds)),
        top_k=max(1, int(args.top_k)),
        seed=int(args.seed),
    )

    if args.report_out:
        out_path = Path(args.report_out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

