# API Reference 📡

> Complete API documentation for MiniMem

## Base URL

```
http://127.0.0.1:20195
```

## Authentication 🔐

All API endpoints (except `/health`) require Basic Auth:

```
Username: admin
Password: admin123
```

---

## Endpoints

### Health

#### Get Health Status

```
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "0.1.0"
}
```

---

### Memories

#### Store Memory

```
POST /api/v1/memories
```

**Request Body:**
```json
{
  "content": "User's message or memory content",
  "metadata": {
    "type": "conversation",
    "conversation_id": "conv_123"
  }
}
```

**Response:**
```json
{
  "id": "mem_abc123",
  "content": "User's message or memory content",
  "created_at": "2024-01-01T12:00:00Z"
}
```

#### Search Memories

```
GET /api/v1/memories/search?query=your+search+query&limit=10
```

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | string | required | Search query |
| `limit` | int | 10 | Max results |
| `profile` | string | "agentic" | Retrieval profile (`keyword` / `hybrid` / `agentic`) |

**Response:**
```json
{
  "results": [
    {
      "id": "mem_abc123",
      "content": "Memory content",
      "score": 0.95,
      "metadata": {}
    }
  ]
}
```

---

### Chat

#### Simple Chat

```
POST /api/v1/chat/simple
```

**Request Body:**
```json
{
  "message": "What do you remember about our previous conversation?",
  "conversation_id": "conv_123"
}
```

**Response:**
```json
{
  "response": "Based on our previous conversation...",
  "citations": [
    {
      "memory_id": "mem_abc123",
      "content": "Previous conversation context",
      "score": 0.92
    }
  ]
}
```

---

### Graph

#### Search Graph

```
GET /api/v1/graph/search?entity=person&limit=10
```

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `entity` | string | required | Entity name to search |
| `limit` | int | 10 | Max results |

**Response:**
```json
{
  "nodes": [
    {
      "id": "node_1",
      "type": "person",
      "name": "John"
    }
  ],
  "edges": [
    {
      "from": "node_1",
      "to": "node_2",
      "type": "knows"
    }
  ]
}
```

#### Get Neighbors

```
GET /api/v1/graph/neighbors?node_id=node_1&depth=2
```

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `node_id` | string | required | Starting node ID |
| `depth` | int | 1 | Search depth |

---

### Model Config

#### Get Configuration

```
GET /api/v1/model-config
```

**Response:**
```json
{
  "chat_provider": "openai",
  "chat_model": "gpt-4o-mini",
  "embedding_provider": "openai",
  "embedding_model": "text-embedding-3-small",
  "graph_enabled": true
}
```

#### Update Configuration

```
PUT /api/v1/model-config
```

**Request Body:**
```json
{
  "chat_provider": "openai",
  "chat_model": "gpt-4o-mini"
}
```

#### Test Connectivity

```
POST /api/v1/model-config/test
```

**Response:**
```json
{
  "chat": {
    "status": "ok",
    "latency_ms": 150
  },
  "embedding": {
    "status": "ok",
    "latency_ms": 80
  }
}
```

---

### Conversation Meta

#### List Conversations

```
GET /api/v1/conversations
```

**Response:**
```json
{
  "conversations": [
    {
      "id": "conv_123",
      "title": "Chat about Python",
      "created_at": "2024-01-01T12:00:00Z",
      "message_count": 10
    }
  ]
}
```

#### Delete Conversation

```
DELETE /api/v1/conversations/{conversation_id}
```

---

## Error Responses

All endpoints may return error responses:

```json
{
  "error": "Error message",
  "code": "ERROR_CODE"
}
```

Common status codes:
- `200` - Success
- `400` - Bad Request
- `401` - Unauthorized
- `500` - Internal Server Error
