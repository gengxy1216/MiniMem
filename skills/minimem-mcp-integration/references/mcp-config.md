# MiniMem MCP Config

## Purpose

Provide quick config snippets for MCP clients to connect to MiniMem bridge server.

## Default Command

```json
{
  "command": "python",
  "args": [
    "skills/minimem-mcp-integration/scripts/run_minimem_mcp.py"
  ]
}
```

## Typical Env

```json
{
  "MINIMEM_BASE_URL": "http://127.0.0.1:20195",
  "MINIMEM_USER_ID": "admin",
  "MINIMEM_GROUP_ID": "default:admin"
}
```

## Optional Auth Env

Use one mode only:

- Bearer:
  - `MINIMEM_BEARER_TOKEN`
- Basic:
  - `MINIMEM_BASIC_USER`
  - `MINIMEM_BASIC_PASSWORD`

## Troubleshooting

- `Connection refused`: MiniMem HTTP service is not running.
- `HTTP 404/405`: Base URL or endpoint path mismatch.
- `HTTP 401/403`: Auth env missing or invalid.
- Empty retrieval result: check `user_id` / `group_id` scope.
