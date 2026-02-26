# LoCoMo 测评脚本（MiniMem）

本仓库新增了两段 LoCoMo 测评准备脚本，可直接对接官方仓库数据：

- 官方仓库：`https://github.com/snap-research/LoCoMo`

- `tools/prepare_locomo_eval.py`：将 LoCoMo 原始数据转换成评测 JSONL
- `tools/run_locomo_eval.py`：本地端到端导入记忆并计算 `Recall@K / MRR`

## 1) 数据转换

将原始 `json/jsonl`（`conversation + qa`）转换成统一评测格式：

```bash
python tools/prepare_locomo_eval.py \
  --input docs/locomo_raw_sample.json \
  --output docs/locomo_eval_sample.generated.jsonl
```

输出每行格式（简化）：

```json
{
  "case_id": "locomo-demo-1-q1",
  "user_id": "zhangming",
  "group_id": "g-family",
  "query": "我儿子叫什么？",
  "expected_message_ids": ["t1"],
  "memories": [
    {"message_id":"t1","content":"...","sender":"zhangming","create_time":1735603201}
  ]
}
```

## 2) 运行 LoCoMo 检索测评

```bash
python tools/run_locomo_eval.py \
  --dataset docs/locomo_eval_sample.jsonl \
  --method keyword \
  --decision-mode static \
  --top-k 8 \
  --sample-size 20 \
  --sample-seed 42 \
  --isolate-by-case
```

默认行为：

- 使用临时 `LITE_DATA_DIR`，不污染现有本地数据
- 默认 `--sample-size 20`，便于高频回归；传 `--sample-size 0` 可全量评测
- `--ingest-profile keyword`，避免无 embedding key 时触发向量依赖
- 默认不把 `user_id` 传入 `/search`，以兼容 LoCoMo 的多说话人对话
- 输出总指标，`--report-out` 可落盘完整 case 级报告

## 3) 官方 LoCoMo 数据一键流程（locomo10）

```powershell
New-Item -ItemType Directory -Force -Path MiniMem_data\benchmarks\locomo | Out-Null
Invoke-WebRequest -Uri https://raw.githubusercontent.com/snap-research/LoCoMo/main/data/locomo10.json -OutFile MiniMem_data\benchmarks\locomo\locomo10.json
python tools/prepare_locomo_eval.py --input MiniMem_data/benchmarks/locomo/locomo10.json --output MiniMem_data/benchmarks/locomo/locomo10.eval.jsonl
python tools/run_locomo_eval.py --dataset MiniMem_data/benchmarks/locomo/locomo10.eval.jsonl --method keyword --decision-mode static --top-k 8 --sample-size 20 --report-out MiniMem_data/benchmarks/locomo/locomo10.report.json
```

官方 LoCoMo 数据建议不要加 `--isolate-by-case`，因为同一 `group_id` 下有多条 QA，脚本会按 `group_id` 只导入一次对话，显著降低评测耗时。

## 4) 如果要评估向量/混合检索

当你需要 `--method hybrid/vector/agentic` 时，建议同时设置：

1. 可用的 embedding provider 配置（`LITE_EMBEDDING_*`）
2. `--ingest-profile hybrid` 或 `agentic`（确保写入阶段会构建向量索引）

## 5) 相关样例文件

- 原始样例：`docs/locomo_raw_sample.json`
- 转换后样例：`docs/locomo_eval_sample.jsonl`
