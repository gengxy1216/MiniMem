# <img src="https://raw.githubusercontent.com/gengxy1216/MiniMem/main/docs/assets/minimem-icon.svg" alt="MiniMem logo" width="32" /> MiniMem（中文）

[English](README.md) | 简体中文

<p align="center">
  <img src="docs/assets/minimem-banner.svg" alt="MiniMem banner" />
</p>

MiniMem 是一个面向 AI Agent 的本地优先长时记忆系统，也可作为多智能体协同的记忆基础设施。

- 本地运行：`SQLite + LanceDB + 本地图存储`
- 问答/向量模型支持运行时切换
- 对话支持记忆引用与检索轨迹
- 轻量、易部署、便于快速迭代

## 为什么是 MiniMem

- 多智能体协同记忆底座：同一套记忆能力可被多个 Agent 复用
- 本地优先，部署和调试成本低
- 文本/向量/图谱混合检索，并支持 citation 追踪
- 对话/向量/抽取均走真实 provider（核心链路无 mock）
- 支持共享组、分角色、按用户的记忆策略，便于协作与隔离
- 轻量部署，适合边缘和资源受限环境

## 关键定位：多智能体协同的记忆基础设施

MiniMem 不只是单个助手的“记忆库”，而是面向协作系统的记忆底层能力：

- 统一接入层：通过 REST、MCP、插件桥接，为不同 Agent/运行时提供一致记忆服务
- 可控共享边界：支持 `per_role`、共享组、按用户分组等策略
- 可审计可调试：提供检索轨迹与 citation，便于回溯“为何命中这条记忆”
- 本地优先低成本：数据本地可控，适合团队开发、测试与边缘部署

## 快速开始

一键安装：

`pip`：

```bash
pip install minimem-agent
minimem
```

`npm`：

```bash
npm install openclaw-plugin-minimem
```

`OpenClaw 插件`：

```bash
openclaw plugins install openclaw-plugin-minimem
openclaw plugins enable minimem-memory
```

本地安装（保留）：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/install.ps1 -RunAfterInstall
```

```bash
bash scripts/install.sh --run
```

Windows 双击启动文件：`start_minimem.bat`  

或手动安装：

```bash
pip install -e .
minimem
```

访问 UI：

- `http://127.0.0.1:20195/ui`

## 配置

核心环境变量分组：

- 问答模型：`LITE_CHAT_*`
- 向量模型：`LITE_EMBEDDING_*`
- 图谱模块：`LITE_GRAPH_*`
- 检索档位：`LITE_RETRIEVAL_PROFILE`
  - 向量分层持久化：`LITE_VECTOR_LANCEDB_ENABLED`、`LITE_VECTOR_LANCEDB_MIN_IMPORTANCE`
  - 管理接口保护：`LITE_ADMIN_TOKEN`、`LITE_ADMIN_ALLOW_LOCALHOST`
  - 检索时延/召回调参：`LITE_SEARCH_BUDGET_FACTOR`、`LITE_SEARCH_MIN_PROBE_K`、
    `LITE_KEYWORD_CONFIDENT_BEST_SCORE`、`LITE_KEYWORD_CONFIDENT_KTH_SCORE`、
    `LITE_SEMANTIC_VECTOR_BUDGET_CAP`、`LITE_SEMANTIC_KEYWORD_BUDGET_CAP`、
    `LITE_QUERY_EMBED_CACHE_SIZE`、`LITE_QUERY_EMBED_CACHE_TTL_SEC`

  受保护接口（`/api/v1/config/raw`、`/api/v1/model-config*`）支持：

  - `Authorization: Bearer <LITE_ADMIN_TOKEN>`
  - `X-API-Key: <LITE_ADMIN_TOKEN>`

  当 `LITE_ADMIN_TOKEN` 为空时，默认仅允许本机访问（`LITE_ADMIN_ALLOW_LOCALHOST=true`）。

未设置 `LITE_DATA_DIR` 时，默认本地存储目录：

- Windows：`%LOCALAPPDATA%\\MiniMem`
- Linux/macOS：`$XDG_DATA_HOME/minimem` 或 `~/.local/share/minimem`

也可在 UI 中动态调整：

- `Settings` 页面
- `GET/PUT /api/v1/model-config`
- `POST /api/v1/model-config/test`

## API 概览

- 健康检查：`GET /health`
- 写入记忆：`POST /api/v1/memories`
- 检索记忆：`GET /api/v1/memories/search`
- 对话：`POST /api/v1/chat/simple`
- 图谱：`GET /api/v1/graph/search`、`GET /api/v1/graph/neighbors`

## MCP 集成

仓库提供了可独立运行的 MCP 桥接能力，便于其他 Agent 直接集成 MiniMem：

- MCP 服务目录：`integrations/minimem-mcp/server.py`
- Skill 目录：`skills/minimem-mcp-integration`

启动 MCP 服务：

```bash
python skills/minimem-mcp-integration/scripts/run_minimem_mcp.py
```

生成 MCP 客户端配置片段：

```bash
python skills/minimem-mcp-integration/scripts/generate_mcp_config.py
```

执行 MCP 集成测试：

```bash
pytest -q tests/test_minimem_mcp_server.py
```

将 Skill 打包为可分发产物：

```bash
python C:/Users/user/.agents/skills/skill-creator/scripts/package_skill.py skills/minimem-mcp-integration dist
```

打包产物路径：

- `dist/minimem-mcp-integration.skill`

常用环境变量（可选）：

- `MINIMEM_BASE_URL`（默认：`http://127.0.0.1:20195`）
- `MINIMEM_USER_ID`
- `MINIMEM_GROUP_ID`
- `MINIMEM_BEARER_TOKEN` 或 `MINIMEM_BASIC_USER` + `MINIMEM_BASIC_PASSWORD`

## OpenClaw 插件集成

MiniMem 也提供了轻量 OpenClaw 插件桥接：

- 插件目录：`integrations/openclaw-plugin`
- 插件清单：`integrations/openclaw-plugin/openclaw.plugin.json`
- 使用文档：`integrations/openclaw-plugin/README.md`
- 策略模板：`integrations/openclaw-plugin/examples/AGENTS.memory-policy.md`

核心能力：

- 写入对话记录 / bot 画像 / 上下文压缩到 MiniMem
- 按策略检索记忆（`keyword`/`vector`/`hybrid`/`rrf`/`agentic`）
- 返回 `context_for_agent` 供 agent 上下文注入
- 可选自动写入（`agent_end`）与自动注入（`before_agent_start`）
- 支持分角色 / 共享组 / 按用户三种分组策略

一键安装：

```powershell
powershell -ExecutionPolicy Bypass -File integrations/openclaw-plugin/install.ps1
```

```bash
bash integrations/openclaw-plugin/install.sh
```

或使用分发插件包安装：

```bash
openclaw plugins install openclaw-plugin-minimem
openclaw plugins enable minimem-memory
```

进阶插件配置示例：

```json
{
  "path": "C:/MiniMem-main/MiniMem/integrations/openclaw-plugin",
  "enabled": true,
  "config": {
    "baseUrl": "http://127.0.0.1:20195",
    "groupStrategy": "per_role",
    "sharedGroupId": "shared:team",
    "autoSenderFromAgent": true,
    "autoInjectOnStart": true,
    "autoCaptureOnEnd": true
  }
}
```

## 架构分层（L0 -> L2）

MiniMem 使用分层记忆架构，在简单性、召回质量和检索性能之间做平衡：

| 层级 | 名称 | 核心存储 | 检索职责 |
|------|------|----------|----------|
| `L0` | Plain Text | SQLite（`episodic_memory`） | 事实源，关键词与元数据检索 |
| `L0.5` | Vector-Text（热层） | 本地向量缓存（`jsonl` + snapshot） | 本地语义召回，重启后快速恢复 |
| `L1` | Vector（持久层） | LanceDB（`memory_vector_index`） | 面向高重要度记忆的向量检索 |
| `L2` | Graph | 本地三元组图存储 | 结构化关系检索与图扩展 |

### 检索流程

1. 查询先进入策略/模式选择层。
2. 在混合检索中并行执行关键词与向量召回（`L0` + `L0.5` + `L1`）。
3. 若启用图谱，合并 `L2` 关系证据。
4. 最终结果重排，并输出 query-aware 的 citation 片段。

## 文档

- API：`docs/evermemos-lite/api-reference.md`
- 架构：`docs/evermemos-lite/architecture.md`
- 部署档位：`docs/evermemos-lite/deployment-profiles.md`
- 运行时策略：`docs/evermemos-lite/runtime-control.md`
- 冲突策略：`docs/evermemos-lite/conflict-strategy.md`
- 评测：`docs/evermemos-lite/evaluation.md`
- 使用文档：`docs/evermemos-lite/user-guide.md`

## 社区

- 贡献指南：`CONTRIBUTING.md`
- 行为准则：`CODE_OF_CONDUCT.md`
- 安全策略：`SECURITY.md`
- 开源协议：`LICENSE`（MIT）
- 变更记录：`CHANGELOG.md`
- 发布流程：`docs/RELEASING.md`
- GitHub 上线清单：`docs/github-growth-checklist.md`
- 开源运营手册：`docs/OPEN_SOURCE_PLAYBOOK.md`

## 开发

```bash
pytest -q
```
