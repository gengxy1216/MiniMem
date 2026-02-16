# MiniMem（中文）

MiniMem 是一个面向 AI Agent 的本地优先长时记忆系统。

- 本地运行：`SQLite + LanceDB + 可选 Kuzu 图谱`
- 问答/向量模型支持运行时切换
- 对话支持记忆引用与检索轨迹
- 轻量、易部署、便于快速迭代

## 为什么是 MiniMem

- 本地优先，部署和调试成本低
- 对话/向量/抽取均走真实 provider（核心链路无 mock）
- 文本/向量/图谱混合检索，并支持 citation 追踪
- 面向真实 Agent 记忆场景设计

## 快速开始

一键安装并启动：

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
