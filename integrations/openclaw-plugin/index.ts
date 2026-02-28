type RetrieveMethod = "keyword" | "vector" | "hybrid" | "rrf" | "agentic";
type DecisionMode = "static" | "rule" | "agent";
type MemoryType = "dialogue" | "bot_profile" | "context_compression" | "note";
type GroupStrategy = "shared" | "per_role" | "per_user";

type PluginConfig = {
  baseUrl: string;
  timeoutMs: number;
  bearerToken: string;
  basicUser: string;
  basicPassword: string;
  defaultUserId: string;
  defaultGroupId: string;
  defaultSender: string;
  defaultTopK: number;
  defaultRetrieveMethod: RetrieveMethod;
  defaultDecisionMode: DecisionMode;
  groupStrategy: GroupStrategy;
  sharedGroupId: string;
  autoSenderFromAgent: boolean;
  autoInjectOnStart: boolean;
  bootstrapQuery: string;
  bootstrapTopK: number;
  autoCaptureOnEnd: boolean;
  autoCaptureCompression: boolean;
  compressionMaxChars: number;
  debug: boolean;
};

const RETRIEVE_METHODS = new Set<RetrieveMethod>([
  "keyword",
  "vector",
  "hybrid",
  "rrf",
  "agentic",
]);
const DECISION_MODES = new Set<DecisionMode>(["static", "rule", "agent"]);
const GROUP_STRATEGIES = new Set<GroupStrategy>(["shared", "per_role", "per_user"]);
const MEMORY_TYPES = new Set<MemoryType>([
  "dialogue",
  "bot_profile",
  "context_compression",
  "note",
]);

function toStr(value: unknown): string {
  return String(value ?? "").trim();
}

function isObject(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function parseLoosePairs(value: string): Record<string, unknown> | undefined {
  const raw = value.trim();
  if (!raw || !raw.includes("=")) {
    return undefined;
  }
  const out: Record<string, unknown> = {};
  const parts = raw.split(/[,&]/).map((item) => item.trim()).filter(Boolean);
  for (const part of parts) {
    const idx = part.indexOf("=");
    if (idx <= 0) {
      continue;
    }
    const key = part.slice(0, idx).trim();
    const val = part.slice(idx + 1).trim();
    if (!key) {
      continue;
    }
    out[key] = val;
  }
  return Object.keys(out).length > 0 ? out : undefined;
}

function parseCandidateValue(value: unknown): unknown {
  if (typeof value !== "string") {
    return undefined;
  }
  const raw = value.trim();
  if (!raw) {
    return undefined;
  }
  if ((raw.startsWith("{") && raw.endsWith("}")) || (raw.startsWith("[") && raw.endsWith("]"))) {
    try {
      return JSON.parse(raw);
    } catch {
      // fall through
    }
  }
  return parseLoosePairs(raw);
}

function collectInputCandidates(value: unknown): Record<string, unknown>[] {
  const queue: unknown[] = [value];
  const out: Record<string, unknown>[] = [];
  const seen = new Set<Record<string, unknown>>();
  while (queue.length > 0) {
    const current = queue.shift();
    const parsed = parseCandidateValue(current);
    if (parsed !== undefined) {
      queue.push(parsed);
      continue;
    }
    if (Array.isArray(current)) {
      for (const item of current) {
        queue.push(item);
      }
      continue;
    }
    if (!isObject(current) || seen.has(current)) {
      continue;
    }
    seen.add(current);
    out.push(current);
    for (const child of Object.values(current)) {
      queue.push(child);
    }
  }
  return out;
}

function normalizeToolInput(value: unknown): Record<string, unknown> {
  const candidates = collectInputCandidates(value);
  if (candidates.length === 0) {
    return {};
  }
  const importantKeys = new Set([
    "content",
    "query",
    "sender",
    "group_id",
    "user_id",
    "memory_type",
    "top_k",
    "message_id",
    "create_time",
  ]);
  for (const item of candidates) {
    for (const key of Object.keys(item)) {
      if (importantKeys.has(String(key))) {
        return item;
      }
    }
  }
  return candidates[0];
}

function extractExecuteInput(args: unknown[]): Record<string, unknown> {
  const hints: Record<string, unknown> = {};
  for (const arg of args) {
    const candidates = collectInputCandidates(arg);
    for (const item of candidates) {
      if (!hints.agent_id) {
        const agentObj = isObject(item.agent) ? (item.agent as Record<string, unknown>) : undefined;
        hints.agent_id = item.agent_id || item.agentId || agentObj?.id;
      }
      if (!hints.session_key) {
        hints.session_key = item.session_key || item.sessionKey;
      }
      if (!hints.sender) {
        hints.sender = item.sender;
      }
      if (!hints.user_id) {
        hints.user_id = item.user_id || item.userId;
      }
      if (!hints.group_id) {
        hints.group_id = item.group_id || item.groupId;
      }
    }
  }
  for (const arg of args) {
    const normalized = normalizeToolInput(arg);
    if (Object.keys(normalized).length > 0) {
      return {
        ...hints,
        ...normalized,
      };
    }
  }
  return hints;
}

function clipText(value: string, maxLen: number): string {
  const raw = String(value ?? "");
  return raw.length > maxLen ? raw.slice(0, maxLen) : raw;
}

function envString(...keys: string[]): string {
  for (const key of keys) {
    const val = toStr(process.env[key]);
    if (val) {
      return val;
    }
  }
  return "";
}

function envBool(keys: string[], fallback: boolean): boolean {
  for (const key of keys) {
    const raw = toStr(process.env[key]).toLowerCase();
    if (!raw) {
      continue;
    }
    if (raw === "1" || raw === "true" || raw === "yes" || raw === "on") {
      return true;
    }
    if (raw === "0" || raw === "false" || raw === "no" || raw === "off") {
      return false;
    }
  }
  return fallback;
}

function envNumber(keys: string[], fallback: number): number {
  for (const key of keys) {
    const raw = Number(process.env[key]);
    if (Number.isFinite(raw)) {
      return raw;
    }
  }
  return fallback;
}

function normalizeConfig(api: any): PluginConfig {
  const raw = (api && typeof api.pluginConfig === "object" && api.pluginConfig) || {};
  const defaultTopK = Number.isFinite(raw.defaultTopK)
    ? Number(raw.defaultTopK)
    : envNumber(["MINIMEM_DEFAULT_TOP_K"], 8);
  const bootstrapTopK = Number.isFinite(raw.bootstrapTopK)
    ? Number(raw.bootstrapTopK)
    : envNumber(["MINIMEM_BOOTSTRAP_TOP_K"], 6);
  const timeoutMs = Number.isFinite(raw.timeoutMs)
    ? Number(raw.timeoutMs)
    : envNumber(["MINIMEM_TIMEOUT_MS"], 25000);
  const compressionMaxChars = Number.isFinite(raw.compressionMaxChars)
    ? Number(raw.compressionMaxChars)
    : envNumber(["MINIMEM_COMPRESSION_MAX_CHARS"], 2500);
  const retrieveMethod = (
    toStr(raw.defaultRetrieveMethod) || envString("MINIMEM_DEFAULT_RETRIEVE_METHOD")
  ) as RetrieveMethod;
  const decisionMode = (
    toStr(raw.defaultDecisionMode) || envString("MINIMEM_DEFAULT_DECISION_MODE")
  ) as DecisionMode;
  const groupStrategy = (
    toStr(raw.groupStrategy) || envString("MINIMEM_GROUP_STRATEGY")
  ) as GroupStrategy;
  return {
    baseUrl:
      toStr(raw.baseUrl) ||
      envString("MINIMEM_BASE_URL", "MINIMEM_API_BASE_URL") ||
      "http://127.0.0.1:20195",
    timeoutMs: Math.max(1000, Math.min(120000, timeoutMs)),
    bearerToken: toStr(raw.bearerToken) || envString("MINIMEM_BEARER_TOKEN"),
    basicUser: toStr(raw.basicUser) || envString("MINIMEM_BASIC_USER"),
    basicPassword: toStr(raw.basicPassword) || envString("MINIMEM_BASIC_PASSWORD"),
    defaultUserId: toStr(raw.defaultUserId) || envString("MINIMEM_USER_ID"),
    defaultGroupId: toStr(raw.defaultGroupId) || envString("MINIMEM_GROUP_ID"),
    defaultSender:
      toStr(raw.defaultSender) || envString("MINIMEM_DEFAULT_SENDER") || "openclaw-bot",
    defaultTopK: Math.max(1, Math.min(100, defaultTopK)),
    defaultRetrieveMethod: RETRIEVE_METHODS.has(retrieveMethod) ? retrieveMethod : "agentic",
    defaultDecisionMode: DECISION_MODES.has(decisionMode) ? decisionMode : "rule",
    groupStrategy: GROUP_STRATEGIES.has(groupStrategy) ? groupStrategy : "per_role",
    sharedGroupId:
      toStr(raw.sharedGroupId) || envString("MINIMEM_SHARED_GROUP_ID") || "shared:openclaw",
    autoSenderFromAgent:
      raw.autoSenderFromAgent !== undefined
        ? Boolean(raw.autoSenderFromAgent)
        : envBool(["MINIMEM_AUTO_SENDER_FROM_AGENT"], true),
    autoInjectOnStart:
      raw.autoInjectOnStart !== undefined
        ? Boolean(raw.autoInjectOnStart)
        : envBool(["MINIMEM_AUTO_INJECT_ON_START"], true),
    bootstrapQuery: toStr(raw.bootstrapQuery) || envString("MINIMEM_BOOTSTRAP_QUERY"),
    bootstrapTopK: Math.max(1, Math.min(20, bootstrapTopK)),
    autoCaptureOnEnd:
      raw.autoCaptureOnEnd !== undefined
        ? Boolean(raw.autoCaptureOnEnd)
        : envBool(["MINIMEM_AUTO_CAPTURE_ON_END"], true),
    autoCaptureCompression:
      raw.autoCaptureCompression !== undefined
        ? Boolean(raw.autoCaptureCompression)
        : envBool(["MINIMEM_AUTO_CAPTURE_COMPRESSION"], true),
    compressionMaxChars: Math.max(200, Math.min(12000, compressionMaxChars)),
    debug: raw.debug !== undefined ? Boolean(raw.debug) : envBool(["MINIMEM_DEBUG"], false),
  };
}

function parseSessionRole(sessionKey: unknown): string {
  const text = toStr(sessionKey);
  if (!text) {
    return "";
  }
  const parts = text.split(":").map((item) => item.trim()).filter(Boolean);
  if (parts.length >= 3 && parts[0] === "agent") {
    return parts[1] || parts[2] || "";
  }
  return parts[parts.length - 1] || "";
}

function inferAgentId(event: any): string {
  const candidates = [
    event?.agentId,
    event?.agent_id,
    event?.agent?.id,
    event?.run?.agentId,
    event?.session?.agentId,
    parseSessionRole(event?.sessionKey),
  ];
  for (const item of candidates) {
    const value = toStr(item);
    if (value) {
      return value;
    }
  }
  return "";
}

function resolveScope(
  cfg: PluginConfig,
  input: {
    user_id?: unknown;
    group_id?: unknown;
    sender?: unknown;
    agent_id?: unknown;
    session_key?: unknown;
  }
): { userId: string; groupId: string; sender: string } {
  const derivedAgentId = toStr(input.agent_id) || parseSessionRole(input.session_key);
  const sender = toStr(input.sender)
    || (cfg.autoSenderFromAgent ? derivedAgentId : "")
    || cfg.defaultSender
    || cfg.defaultUserId
    || "openclaw-bot";
  const userId = toStr(input.user_id) || cfg.defaultUserId || sender;
  const explicitGroup = toStr(input.group_id) || cfg.defaultGroupId;
  let groupId = explicitGroup;
  if (!groupId) {
    if (cfg.groupStrategy === "shared") {
      groupId = cfg.sharedGroupId || "shared:openclaw";
    } else if (cfg.groupStrategy === "per_user") {
      groupId = `default:${userId}`;
    } else {
      groupId = `default:${sender}`;
    }
  }
  return { userId, groupId, sender };
}

function memoryPrefix(memoryType: MemoryType): string {
  if (memoryType === "dialogue") {
    return "[DIALOGUE]";
  }
  if (memoryType === "bot_profile") {
    return "[BOT_PROFILE]";
  }
  if (memoryType === "context_compression") {
    return "[CONTEXT_COMPRESSION]";
  }
  return "[NOTE]";
}

function buildWriteContent(input: {
  memory_type?: unknown;
  content?: unknown;
  metadata?: unknown;
}): { memoryType: MemoryType; content: string } {
  const memoryTypeRaw = toStr(input.memory_type).toLowerCase() as MemoryType;
  const memoryType = MEMORY_TYPES.has(memoryTypeRaw) ? memoryTypeRaw : "dialogue";
  const content = toStr(input.content);
  if (!content) {
    throw new Error("content is required");
  }
  const lines = [`${memoryPrefix(memoryType)} ${content}`];
  if (input.metadata && typeof input.metadata === "object") {
    const metadata = JSON.stringify(input.metadata);
    if (metadata && metadata !== "{}") {
      lines.push(`[metadata] ${metadata}`);
    }
  }
  return { memoryType, content: lines.join("\n") };
}

function extractMessageText(content: unknown): string {
  if (typeof content === "string") {
    return content.trim();
  }
  if (Array.isArray(content)) {
    const parts: string[] = [];
    for (const item of content) {
      if (!item || typeof item !== "object") {
        continue;
      }
      const maybeText = toStr((item as Record<string, unknown>).text);
      if (maybeText) {
        parts.push(maybeText);
      }
    }
    return parts.join(" ").trim();
  }
  if (content && typeof content === "object") {
    const obj = content as Record<string, unknown>;
    return toStr(obj.text || obj.content || obj.value);
  }
  return "";
}

function extractLastTurn(messages: unknown): { user: string; assistant: string } {
  if (!Array.isArray(messages)) {
    return { user: "", assistant: "" };
  }
  let userText = "";
  let assistantText = "";
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    const msg = messages[i];
    if (!msg || typeof msg !== "object") {
      continue;
    }
    const role = toStr((msg as Record<string, unknown>).role).toLowerCase();
    const text = extractMessageText((msg as Record<string, unknown>).content);
    if (!text) {
      continue;
    }
    if (!assistantText && role === "assistant") {
      assistantText = text;
      continue;
    }
    if (!userText && role === "user") {
      userText = text;
      if (assistantText) {
        break;
      }
    }
  }
  return { user: userText, assistant: assistantText };
}

function compressMessages(messages: unknown, maxChars: number): string {
  if (!Array.isArray(messages) || messages.length === 0) {
    return "";
  }
  const lines: string[] = [];
  for (const message of messages.slice(-10)) {
    if (!message || typeof message !== "object") {
      continue;
    }
    const obj = message as Record<string, unknown>;
    const role = toStr(obj.role) || "unknown";
    const text = extractMessageText(obj.content);
    if (!text) {
      continue;
    }
    lines.push(`${role}: ${text}`);
  }
  if (!lines.length) {
    return "";
  }
  return clipText(lines.join("\n"), maxChars);
}

function selectBootstrapQuery(cfg: PluginConfig, event: any): string {
  if (cfg.bootstrapQuery) {
    return cfg.bootstrapQuery;
  }
  const prompt = toStr(event?.prompt);
  if (prompt) {
    return prompt;
  }
  const turn = extractLastTurn(event?.messages);
  return turn.user;
}

function normalizeMemories(payload: any): Array<Record<string, any>> {
  const rows = payload?.result?.memories;
  if (!Array.isArray(rows)) {
    return [];
  }
  return rows
    .filter((row) => row && typeof row === "object")
    .map((row) => {
      const item = row as Record<string, any>;
      const content = toStr(item.content || item.episode || item.summary);
      return {
        id: toStr(item.id || item.event_id),
        content,
        summary: toStr(item.summary),
        sender: toStr(item.sender || item.user_id),
        group_id: toStr(item.group_id),
        timestamp: Number(item.timestamp || 0) || 0,
        score: Number(item.score || 0) || 0,
        source: toStr(item.source),
      };
    })
    .filter((item) => item.content);
}

function buildContextBlock(query: string, rows: Array<Record<string, any>>, topK: number): string {
  const picked = rows.slice(0, Math.max(1, topK));
  if (!picked.length) {
    return "";
  }
  const lines = ["## MiniMem Retrieved Context", `Query: ${query}`];
  for (let i = 0; i < picked.length; i += 1) {
    const row = picked[i];
    const content = clipText(toStr(row.content), 900);
    const meta = [
      row.sender ? `sender=${row.sender}` : "",
      row.group_id ? `group=${row.group_id}` : "",
      row.source ? `source=${row.source}` : "",
    ]
      .filter(Boolean)
      .join(", ");
    lines.push(`${i + 1}. ${content}${meta ? ` (${meta})` : ""}`);
  }
  return lines.join("\n");
}

function authHeaders(cfg: PluginConfig): Record<string, string> {
  if (cfg.bearerToken) {
    return { Authorization: `Bearer ${cfg.bearerToken}` };
  }
  if (cfg.basicUser) {
    const token = Buffer.from(`${cfg.basicUser}:${cfg.basicPassword}`, "utf8").toString("base64");
    return { Authorization: `Basic ${token}` };
  }
  return {};
}

async function requestJson(
  cfg: PluginConfig,
  method: "GET" | "POST",
  path: string,
  options: { params?: Record<string, unknown>; body?: Record<string, unknown> } = {}
): Promise<any> {
  const base = cfg.baseUrl.replace(/\/+$/, "");
  const url = new URL(`${base}${path}`);
  if (options.params) {
    for (const [key, value] of Object.entries(options.params)) {
      const text = toStr(value);
      if (!text) {
        continue;
      }
      url.searchParams.set(key, text);
    }
  }
  const headers: Record<string, string> = {
    "Content-Type": "application/json; charset=utf-8",
    ...authHeaders(cfg),
  };
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), cfg.timeoutMs);
  try {
    const response = await fetch(url.toString(), {
      method,
      headers,
      body: options.body ? JSON.stringify(options.body) : undefined,
      signal: controller.signal,
    });
    const raw = await response.text();
    if (!response.ok) {
      throw new Error(`MiniMem API HTTP ${response.status}: ${clipText(raw, 320)}`);
    }
    try {
      return JSON.parse(raw);
    } catch (error) {
      throw new Error(`MiniMem API returned invalid JSON: ${clipText(raw, 280)}`);
    }
  } catch (error) {
    throw new Error(`MiniMem API request failed: ${String(error)}`);
  } finally {
    clearTimeout(timer);
  }
}

export default function minimemOpenclawPlugin(api: any) {
  const cfg = normalizeConfig(api);

  async function writeMemory(input: Record<string, unknown>): Promise<any> {
    const prepared = buildWriteContent(input);
    const scope = resolveScope(cfg, input);
    const senderName = toStr(input.sender_name) || scope.sender;
    const role = toStr(input.role) || "user";
    const messageId = toStr(input.message_id) || `openclaw-${Date.now().toString(36)}`;
    const createTimeRaw = Number(input.create_time);
    const createTime = Number.isFinite(createTimeRaw) && createTimeRaw > 0
      ? Math.floor(createTimeRaw)
      : Math.floor(Date.now() / 1000);
    return requestJson(cfg, "POST", "/api/v1/memories", {
      body: {
        message_id: messageId,
        create_time: createTime,
        sender: scope.sender,
        sender_name: senderName,
        role,
        group_id: scope.groupId,
        content: prepared.content,
      },
    });
  }

  async function retrieveMemory(input: Record<string, unknown>): Promise<any> {
    const query = toStr(input.query);
    if (!query) {
      throw new Error("query is required");
    }
    const scope = resolveScope(cfg, input);
    const topKRaw = Number(input.top_k);
    const topK = Number.isFinite(topKRaw) ? Math.floor(topKRaw) : cfg.defaultTopK;
    const retrieveMethod = (toStr(input.retrieve_method) || cfg.defaultRetrieveMethod) as RetrieveMethod;
    const decisionMode = (toStr(input.decision_mode) || cfg.defaultDecisionMode) as DecisionMode;
    const runtimeProfile = toStr(input.runtime_profile);
    const response = await requestJson(cfg, "GET", "/api/v1/memories/search", {
      params: {
        query,
        user_id: scope.userId,
        group_id: scope.groupId,
        top_k: Math.max(1, Math.min(100, topK)),
        retrieve_method: RETRIEVE_METHODS.has(retrieveMethod)
          ? retrieveMethod
          : cfg.defaultRetrieveMethod,
        decision_mode: DECISION_MODES.has(decisionMode)
          ? decisionMode
          : cfg.defaultDecisionMode,
        runtime_profile: runtimeProfile || undefined,
      },
    });
    const rows = normalizeMemories(response);
    const contextForAgent = buildContextBlock(query, rows, topK);
    return {
      ok: true,
      query,
      top_k: topK,
      count: rows.length,
      memories: rows,
      context_for_agent: contextForAgent,
      raw: response,
    };
  }

  api.registerTool({
    name: "minimem_memory_write",
    description:
      "Write memory into MiniMem. Supports dialogue, bot_profile, and context_compression.",
    parameters: {
      type: "object",
      required: ["content"],
      properties: {
        content: { type: "string", minLength: 1, maxLength: 10000 },
        memory_type: {
          type: "string",
          enum: ["dialogue", "bot_profile", "context_compression", "note"],
          default: "dialogue",
        },
        sender: { type: "string" },
        sender_name: { type: "string" },
        role: { type: "string", default: "user" },
        user_id: { type: "string" },
        group_id: { type: "string" },
        message_id: { type: "string" },
        create_time: { type: "integer" },
        metadata: { type: "object", additionalProperties: true },
      },
    },
    execute: async (...runtimeArgs: unknown[]) => {
      const args = extractExecuteInput(runtimeArgs);
      const result = await writeMemory(args);
      return {
        ok: true,
        message: "memory_written",
        result,
      };
    },
  });

  api.registerTool({
    name: "minimem_memory_retrieve",
    description:
      "Retrieve memory by strategy and return a compact context block for agent prompt injection.",
    parameters: {
      type: "object",
      required: ["query"],
      properties: {
        query: { type: "string", minLength: 1, maxLength: 8000 },
        top_k: { type: "integer", minimum: 1, maximum: 100, default: 8 },
        retrieve_method: {
          type: "string",
          enum: ["keyword", "vector", "hybrid", "rrf", "agentic"],
          default: "agentic",
        },
        decision_mode: {
          type: "string",
          enum: ["static", "rule", "agent"],
          default: "rule",
        },
        runtime_profile: { type: "string" },
        user_id: { type: "string" },
        group_id: { type: "string" },
      },
    },
    execute: async (...runtimeArgs: unknown[]) =>
      retrieveMemory(extractExecuteInput(runtimeArgs)),
  });

  if (typeof api.on === "function" && cfg.autoInjectOnStart) {
    api.on("before_agent_start", async (event: any) => {
      try {
        const agentId = inferAgentId(event);
        const scope = resolveScope(cfg, {
          agent_id: agentId,
          session_key: event?.sessionKey,
        });
        const query = selectBootstrapQuery(cfg, event);
        if (!query) {
          return;
        }
        const recalled = await retrieveMemory({
          query,
          top_k: cfg.bootstrapTopK,
          retrieve_method: cfg.defaultRetrieveMethod,
          decision_mode: cfg.defaultDecisionMode,
          user_id: scope.userId,
          group_id: scope.groupId,
        });
        const context = toStr(recalled?.context_for_agent);
        if (!context) {
          return;
        }
        return { prependContext: context };
      } catch (error) {
        if (cfg.debug) {
          console.error("[minimem-memory] auto inject failed", error);
        }
      }
    });
  }

  if (typeof api.on === "function" && cfg.autoCaptureOnEnd) {
    api.on("agent_end", async (event: any) => {
      try {
        const agentId = inferAgentId(event);
        const scope = resolveScope(cfg, {
          agent_id: agentId,
          session_key: event?.sessionKey,
        });
        const turn = extractLastTurn(event?.messages);
        const sessionKey = toStr(event?.sessionKey);
        if (turn.user || turn.assistant) {
          const dialogue = [
            turn.user ? `User: ${turn.user}` : "",
            turn.assistant ? `Assistant: ${turn.assistant}` : "",
          ]
            .filter(Boolean)
            .join("\n");
          await writeMemory({
            memory_type: "dialogue",
            content: dialogue,
            sender: scope.sender,
            sender_name: scope.sender,
            role: "assistant",
            user_id: scope.userId,
            group_id: scope.groupId,
            metadata: {
              session_key: sessionKey || undefined,
              agent_id: agentId || undefined,
            },
          });
        }
        if (cfg.autoCaptureCompression) {
          const compressed = compressMessages(event?.messages, cfg.compressionMaxChars);
          if (compressed) {
            await writeMemory({
              memory_type: "context_compression",
              content: compressed,
              sender: scope.sender,
              sender_name: scope.sender,
              role: "assistant",
              user_id: scope.userId,
              group_id: scope.groupId,
              metadata: {
                session_key: sessionKey || undefined,
                agent_id: agentId || undefined,
              },
            });
          }
        }
      } catch (error) {
        if (cfg.debug) {
          console.error("[minimem-memory] auto capture failed", error);
        }
      }
    });
  }

  return {
    name: "MiniMem Memory Bridge",
    version: "0.1.0",
  };
}
