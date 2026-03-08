import { appendFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

type RetrieveMethod = "keyword" | "vector" | "hybrid" | "rrf" | "agentic";
type DecisionMode = "static" | "rule" | "agent";
type MemoryType = "dialogue" | "bot_profile" | "context_compression" | "note";
type GroupStrategy = "shared" | "per_role" | "per_user";
type CompressionMode = "truncate" | "llm_summary" | "hybrid";

type PrimaryModelSnapshot = {
  provider: string;
  baseUrl: string;
  model: string;
};

type SharePolicy = {
  readableAgents: string[];
  writableAgents: string[];
};

type PluginConfig = {
  baseUrl: string;
  timeoutMs: number;
  bearerToken: string;
  basicUser: string;
  basicPassword: string;
  defaultUserId: string;
  defaultGroupId: string;
  defaultSender: string;
  defaultAgentId: string;
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
  inheritPrimaryModel: boolean;
  primaryModelSnapshot: PrimaryModelSnapshot;
  primaryModelSyncStatus: string;
  senderMap: Record<string, string>;
  channelGroupMap: Record<string, string>;
  sharePolicy: Record<string, SharePolicy>;
  compressionMode: CompressionMode;
  compressionTimeoutMs: number;
  compressionMinTurns: number;
  compressionLlmBaseUrl: string;
  compressionLlmApiKey: string;
  compressionLlmModel: string;
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
const COMPRESSION_MODES = new Set<CompressionMode>(["truncate", "llm_summary", "hybrid"]);
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
    "agent_id",
    "channel",
    "task_id",
    "trace_id",
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
      if (!hints.channel) {
        hints.channel = item.channel || item.channel_id || item.channelId;
      }
      if (!hints.task_id) {
        hints.task_id = item.task_id || item.taskId || item.run_id || item.runId;
      }
      if (!hints.trace_id) {
        hints.trace_id = item.trace_id || item.traceId || item.request_id || item.requestId;
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

function normalizeStringMap(raw: unknown): Record<string, string> {
  if (!isObject(raw)) {
    return {};
  }
  const out: Record<string, string> = {};
  for (const [key, value] of Object.entries(raw)) {
    const k = toStr(key);
    const v = toStr(value);
    if (!k || !v) {
      continue;
    }
    out[k] = v;
  }
  return out;
}

function normalizeAgentList(raw: unknown): string[] {
  if (Array.isArray(raw)) {
    return Array.from(
      new Set(
        raw.map((item) => toStr(item)).filter(Boolean)
      )
    );
  }
  const text = toStr(raw);
  if (!text) {
    return [];
  }
  return Array.from(
    new Set(
      text
        .split(/[,\s]+/)
        .map((item) => item.trim())
        .filter(Boolean)
    )
  );
}

function normalizeSharePolicy(raw: unknown): Record<string, SharePolicy> {
  if (!isObject(raw)) {
    return {};
  }
  const out: Record<string, SharePolicy> = {};
  for (const [group, policyRaw] of Object.entries(raw)) {
    const groupId = toStr(group);
    if (!groupId || !isObject(policyRaw)) {
      continue;
    }
    const readableAgents = normalizeAgentList(
      policyRaw.readableAgents ?? policyRaw.readable_agents ?? policyRaw.readers
    );
    const writableAgents = normalizeAgentList(
      policyRaw.writableAgents ?? policyRaw.writable_agents ?? policyRaw.writers
    );
    out[groupId] = { readableAgents, writableAgents };
  }
  return out;
}

function normalizeConfig(api: any): PluginConfig {
  const raw = (api && typeof api.pluginConfig === "object" && api.pluginConfig) || {};
  const primaryModelSnapshotRaw = isObject(raw.primaryModelSnapshot)
    ? (raw.primaryModelSnapshot as Record<string, unknown>)
    : {};
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
  const compressionTimeoutMs = Number.isFinite(raw.compressionTimeoutMs)
    ? Number(raw.compressionTimeoutMs)
    : envNumber(["MINIMEM_COMPRESSION_TIMEOUT_MS"], 280);
  const compressionMinTurns = Number.isFinite(raw.compressionMinTurns)
    ? Number(raw.compressionMinTurns)
    : envNumber(["MINIMEM_COMPRESSION_MIN_TURNS"], 2);
  const compressionModeRaw = (
    toStr(raw.compressionMode) || envString("MINIMEM_COMPRESSION_MODE") || "truncate"
  ) as CompressionMode;
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
    defaultAgentId:
      toStr(raw.defaultAgentId) || envString("MINIMEM_DEFAULT_AGENT_ID") || "main",
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
    inheritPrimaryModel:
      raw.inheritPrimaryModel !== undefined
        ? Boolean(raw.inheritPrimaryModel)
        : envBool(["MINIMEM_INHERIT_PRIMARY_MODEL"], true),
    primaryModelSnapshot: {
      provider: toStr(primaryModelSnapshotRaw.provider),
      baseUrl: toStr(primaryModelSnapshotRaw.baseUrl),
      model: toStr(primaryModelSnapshotRaw.model),
    },
    primaryModelSyncStatus: toStr(raw.primaryModelSyncStatus) || "not_synced",
    senderMap: normalizeStringMap(raw.senderMap),
    channelGroupMap: normalizeStringMap(raw.channelGroupMap),
    sharePolicy: normalizeSharePolicy(raw.sharePolicy),
    compressionMode: COMPRESSION_MODES.has(compressionModeRaw) ? compressionModeRaw : "truncate",
    compressionTimeoutMs: Math.max(120, Math.min(5000, compressionTimeoutMs)),
    compressionMinTurns: Math.max(1, Math.min(24, Math.floor(compressionMinTurns))),
    compressionLlmBaseUrl:
      toStr(raw.compressionLlmBaseUrl)
      || envString("MINIMEM_COMPRESSION_LLM_BASE_URL")
      || toStr(primaryModelSnapshotRaw.baseUrl),
    compressionLlmApiKey:
      toStr(raw.compressionLlmApiKey)
      || envString("MINIMEM_COMPRESSION_LLM_API_KEY"),
    compressionLlmModel:
      toStr(raw.compressionLlmModel)
      || envString("MINIMEM_COMPRESSION_LLM_MODEL")
      || toStr(primaryModelSnapshotRaw.model),
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
    event?.agent,
    event?.agentId,
    event?.agent_id,
    event?.agent?.agentId,
    event?.agent?.agent_id,
    event?.agent?.id,
    event?.run?.agentId,
    event?.run?.agent_id,
    event?.run?.agent?.id,
    event?.session?.agentId,
    event?.session?.agent_id,
    event?.session?.agent?.id,
    parseSessionRole(event?.sessionKey),
    parseSessionRole(event?.session_key),
    parseSessionRole(event?.run?.sessionKey),
    parseSessionRole(event?.run?.session_key),
    parseSessionRole(event?.session?.key),
    parseSessionRole(event?.session?.sessionKey),
    parseSessionRole(event?.session?.session_key),
  ];
  for (const item of candidates) {
    const value = toStr(item);
    if (value) {
      return value;
    }
  }
  return "";
}

function inferChannel(event: any): string {
  const candidates = [
    event?.channel,
    event?.channel_id,
    event?.channelId,
    event?.run?.channel,
    event?.run?.channel_id,
    event?.run?.channelId,
    event?.session?.channel,
    event?.session?.channel_id,
    event?.session?.channelId,
  ];
  for (const item of candidates) {
    const value = toStr(item);
    if (value) {
      return value;
    }
  }
  return "";
}

function inferTaskId(event: any): string {
  const candidates = [
    event?.taskId,
    event?.task_id,
    event?.runId,
    event?.run_id,
    event?.run?.id,
    event?.run?.taskId,
    event?.run?.task_id,
    event?.id,
  ];
  for (const item of candidates) {
    const value = toStr(item);
    if (value) {
      return value;
    }
  }
  return "";
}

function inferTraceId(event: any): string {
  const candidates = [
    event?.traceId,
    event?.trace_id,
    event?.requestId,
    event?.request_id,
    event?.run?.traceId,
    event?.run?.trace_id,
    event?.run?.requestId,
    event?.run?.request_id,
  ];
  for (const item of candidates) {
    const value = toStr(item);
    if (value) {
      return value;
    }
  }
  return "";
}

function defaultGroupIdByStrategy(cfg: PluginConfig, sender: string, userId: string): string {
  if (cfg.groupStrategy === "shared") {
    return cfg.sharedGroupId || "shared:openclaw";
  }
  if (cfg.groupStrategy === "per_user") {
    return `default:${userId}`;
  }
  return `default:${sender}`;
}

function groupAccessAllowed(
  cfg: PluginConfig,
  groupId: string,
  agentId: string,
  operation: "read" | "write"
): boolean {
  const policy = cfg.sharePolicy[groupId];
  if (!policy || !agentId) {
    return true;
  }
  const allowList = operation === "write" ? policy.writableAgents : policy.readableAgents;
  if (!allowList.length) {
    return true;
  }
  return allowList.includes(agentId);
}

function resolveScope(
  cfg: PluginConfig,
  input: {
    user_id?: unknown;
    group_id?: unknown;
    sender?: unknown;
    agent_id?: unknown;
    session_key?: unknown;
    channel?: unknown;
    operation?: "read" | "write";
  }
) : {
  userId: string;
  groupId: string;
  sender: string;
  agentId: string;
  channel: string;
  aclFallback: boolean;
} {
  const derivedAgentId = toStr(input.agent_id) || parseSessionRole(input.session_key) || cfg.defaultAgentId;
  const mappedSender = derivedAgentId ? toStr(cfg.senderMap[derivedAgentId]) : "";
  const sender = toStr(input.sender)
    || (cfg.autoSenderFromAgent ? mappedSender || derivedAgentId : "")
    || cfg.defaultSender
    || cfg.defaultUserId
    || "openclaw-bot";
  const userId = toStr(input.user_id) || cfg.defaultUserId || sender;
  const channel = toStr(input.channel);
  const mappedGroup = channel ? toStr(cfg.channelGroupMap[channel]) : "";
  const explicitGroup = toStr(input.group_id) || cfg.defaultGroupId;
  let groupId = explicitGroup || mappedGroup || defaultGroupIdByStrategy(cfg, sender, userId);
  const operation = input.operation || "read";
  let aclFallback = false;
  if (!groupAccessAllowed(cfg, groupId, derivedAgentId, operation)) {
    groupId = defaultGroupIdByStrategy(cfg, sender, userId);
    aclFallback = true;
  }
  return { userId, groupId, sender, agentId: derivedAgentId, channel, aclFallback };
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

function buildRoutingMetadata(
  input: Record<string, unknown>,
  scope: {
    agentId: string;
    channel: string;
    aclFallback: boolean;
  }
): Record<string, unknown> {
  const base = isObject(input.metadata) ? { ...(input.metadata as Record<string, unknown>) } : {};
  const agentId = toStr(input.agent_id) || scope.agentId;
  const channel = toStr(input.channel) || scope.channel;
  const taskId = toStr(input.task_id);
  const traceId = toStr(input.trace_id);
  if (agentId) {
    base.agent_id = agentId;
  }
  if (channel) {
    base.channel = channel;
  }
  if (taskId) {
    base.task_id = taskId;
  }
  if (traceId) {
    base.trace_id = traceId;
  }
  if (scope.aclFallback) {
    base.route_acl_fallback = true;
  }
  return base;
}

function debugLog(cfg: PluginConfig, label: string, payload: Record<string, unknown>): void {
  if (!cfg.debug) {
    return;
  }
  try {
    const line = JSON.stringify({
      ts: new Date().toISOString(),
      label,
      payload,
    });
    appendFileSync(join(tmpdir(), "minimem-openclaw-debug.jsonl"), `${line}\n`, "utf8");
  } catch (_error) {
    // no-op: debug logging must never break the main path
  }
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

type CompressionResult = {
  content: string;
  modeRequested: CompressionMode;
  modeUsed: string;
  turnCount: number;
  sourceChars: number;
  latencyMs: number;
  fallbackReason: string;
};

function collectCompressionSource(
  messages: unknown,
  maxMessages: number = 12,
): { source: string; sourceChars: number; turnCount: number } {
  if (!Array.isArray(messages) || messages.length === 0) {
    return { source: "", sourceChars: 0, turnCount: 0 };
  }
  const lines: string[] = [];
  let turnCount = 0;
  for (const message of messages.slice(-maxMessages)) {
    if (!message || typeof message !== "object") {
      continue;
    }
    const obj = message as Record<string, unknown>;
    const role = toStr(obj.role).toLowerCase() || "unknown";
    const text = extractMessageText(obj.content);
    if (!text) {
      continue;
    }
    if (role === "user" || role === "assistant") {
      turnCount += 1;
    }
    lines.push(`${role}: ${text}`);
  }
  const source = lines.join("\n");
  return { source, sourceChars: source.length, turnCount };
}

function buildChatCompletionsUrl(baseUrl: string): string {
  const base = toStr(baseUrl).replace(/\/+$/, "");
  if (!base) {
    return "";
  }
  if (base.endsWith("/chat/completions")) {
    return base;
  }
  if (base.endsWith("/v1")) {
    return `${base}/chat/completions`;
  }
  return `${base}/chat/completions`;
}

async function summarizeWithLlm(cfg: PluginConfig, source: string): Promise<string> {
  const baseUrl = buildChatCompletionsUrl(cfg.compressionLlmBaseUrl);
  const apiKey = toStr(cfg.compressionLlmApiKey);
  const model = toStr(cfg.compressionLlmModel);
  if (!baseUrl || !apiKey || !model) {
    throw new Error("compression_llm_not_configured");
  }
  const prompt = clipText(source, 6000);
  const payload = {
    model,
    temperature: 0.2,
    messages: [
      {
        role: "system",
        content:
          "Summarize dialogue for memory retrieval. Keep key facts, decisions, constraints and next actions. Output plain text only.",
      },
      {
        role: "user",
        content: `Compress this dialogue into <= ${cfg.compressionMaxChars} chars:\n${prompt}`,
      },
    ],
  };
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), cfg.compressionTimeoutMs);
  try {
    const response = await fetch(baseUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json; charset=utf-8",
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
    const raw = await response.text();
    if (!response.ok) {
      throw new Error(`compression_llm_http_${response.status}:${clipText(raw, 220)}`);
    }
    let body: any = {};
    try {
      body = JSON.parse(raw);
    } catch (_error) {
      throw new Error("compression_llm_invalid_json");
    }
    const content = body?.choices?.[0]?.message?.content;
    let text = "";
    if (typeof content === "string") {
      text = content.trim();
    } else if (Array.isArray(content)) {
      text = content
        .map((item) => (isObject(item) ? toStr(item.text) : ""))
        .join("")
        .trim();
    } else {
      text = toStr(content);
    }
    if (!text) {
      throw new Error("compression_llm_empty");
    }
    return clipText(text, cfg.compressionMaxChars);
  } finally {
    clearTimeout(timer);
  }
}

async function compressMessagesForCapture(
  cfg: PluginConfig,
  messages: unknown,
): Promise<CompressionResult> {
  const startedAt = Date.now();
  const mode = cfg.compressionMode;
  const source = collectCompressionSource(messages);
  const truncateOutput = clipText(source.source, cfg.compressionMaxChars);
  const resultBase: Omit<CompressionResult, "content" | "modeUsed" | "fallbackReason" | "latencyMs"> = {
    modeRequested: mode,
    turnCount: source.turnCount,
    sourceChars: source.sourceChars,
  };
  if (!source.source) {
    return {
      ...resultBase,
      content: "",
      modeUsed: "skip.empty_source",
      fallbackReason: "",
      latencyMs: Date.now() - startedAt,
    };
  }
  if (source.turnCount < cfg.compressionMinTurns) {
    return {
      ...resultBase,
      content: "",
      modeUsed: "skip.min_turns",
      fallbackReason: "",
      latencyMs: Date.now() - startedAt,
    };
  }
  if (mode === "truncate") {
    return {
      ...resultBase,
      content: truncateOutput,
      modeUsed: "truncate",
      fallbackReason: "",
      latencyMs: Date.now() - startedAt,
    };
  }
  const shouldTryLlm = mode === "llm_summary" || (mode === "hybrid" && source.sourceChars > cfg.compressionMaxChars);
  if (!shouldTryLlm) {
    return {
      ...resultBase,
      content: truncateOutput,
      modeUsed: "hybrid.truncate_short_input",
      fallbackReason: "",
      latencyMs: Date.now() - startedAt,
    };
  }
  try {
    const llmText = await summarizeWithLlm(cfg, source.source);
    return {
      ...resultBase,
      content: llmText,
      modeUsed: "llm_summary",
      fallbackReason: "",
      latencyMs: Date.now() - startedAt,
    };
  } catch (error) {
    return {
      ...resultBase,
      content: truncateOutput,
      modeUsed: `${mode}.fallback_truncate`,
      fallbackReason: clipText(String(error || "compression_failed"), 180),
      latencyMs: Date.now() - startedAt,
    };
  }
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
  const currentConfig = (): PluginConfig => normalizeConfig(api);

  async function writeMemory(input: Record<string, unknown>): Promise<any> {
    const cfg = currentConfig();
    const scope = resolveScope(cfg, {
      ...input,
      operation: "write",
    });
    const metadata = buildRoutingMetadata(input, scope);
    const prepared = buildWriteContent({
      ...input,
      metadata,
    });
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
    const cfg = currentConfig();
    const query = toStr(input.query);
    if (!query) {
      throw new Error("query is required");
    }
    const scope = resolveScope(cfg, {
      ...input,
      operation: "read",
    });
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
      scope: {
        user_id: scope.userId,
        group_id: scope.groupId,
        sender: scope.sender,
        agent_id: scope.agentId,
        channel: scope.channel,
        acl_fallback: scope.aclFallback,
      },
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
        agent_id: { type: "string" },
        channel: { type: "string" },
        task_id: { type: "string" },
        trace_id: { type: "string" },
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
        agent_id: { type: "string" },
        channel: { type: "string" },
        task_id: { type: "string" },
        trace_id: { type: "string" },
      },
    },
    execute: async (...runtimeArgs: unknown[]) =>
      retrieveMemory(extractExecuteInput(runtimeArgs)),
  });

  if (typeof api.on === "function") {
    api.on("before_agent_start", async (event: any) => {
      const cfg = currentConfig();
      try {
        if (!cfg.autoInjectOnStart) {
          return;
        }
        const agentId = inferAgentId(event);
        const channel = inferChannel(event);
        const taskId = inferTaskId(event);
        const traceId = inferTraceId(event);
        const scope = resolveScope(cfg, {
          agent_id: agentId,
          session_key: event?.sessionKey,
          channel,
          operation: "read",
        });
        debugLog(cfg, "before_agent_start.scope", {
          inferred: { agentId, channel, taskId, traceId },
          event: {
            sessionKey: toStr(event?.sessionKey),
            session_key: toStr(event?.session_key),
            agentId: toStr(event?.agentId),
            agent_id: toStr(event?.agent_id),
            eventAgent: toStr(event?.agent),
            runAgentId: toStr(event?.run?.agentId),
            runAgent_id: toStr(event?.run?.agent_id),
            sessionAgentId: toStr(event?.session?.agentId),
            sessionAgent_id: toStr(event?.session?.agent_id),
          },
          scope,
        });
        const query = selectBootstrapQuery(cfg, event);
        if (!query) {
          return;
        }
        const injectStartMs = Date.now();
        const recalled = await retrieveMemory({
          query,
          top_k: cfg.bootstrapTopK,
          retrieve_method: cfg.defaultRetrieveMethod,
          decision_mode: cfg.defaultDecisionMode,
          user_id: scope.userId,
          group_id: scope.groupId,
          agent_id: scope.agentId,
          channel,
          task_id: taskId,
          trace_id: traceId,
        });
        const injectLatencyMs = Date.now() - injectStartMs;
        debugLog(cfg, "before_agent_start.inject_latency", {
          latency_ms: injectLatencyMs,
          memory_count: Number(recalled?.count ?? 0),
          context_chars: toStr(recalled?.context_for_agent).length,
          query_chars: query.length,
          scope,
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

  if (typeof api.on === "function") {
    api.on("agent_end", async (event: any) => {
      const cfg = currentConfig();
      try {
        if (!cfg.autoCaptureOnEnd) {
          return;
        }
        const agentId = inferAgentId(event);
        const channel = inferChannel(event);
        const taskId = inferTaskId(event);
        const traceId = inferTraceId(event);
        const scope = resolveScope(cfg, {
          agent_id: agentId,
          session_key: event?.sessionKey,
          channel,
          operation: "write",
        });
        debugLog(cfg, "agent_end.scope", {
          inferred: { agentId, channel, taskId, traceId },
          event: {
            sessionKey: toStr(event?.sessionKey),
            session_key: toStr(event?.session_key),
            agentId: toStr(event?.agentId),
            agent_id: toStr(event?.agent_id),
            eventAgent: toStr(event?.agent),
            runAgentId: toStr(event?.run?.agentId),
            runAgent_id: toStr(event?.run?.agent_id),
            sessionAgentId: toStr(event?.session?.agentId),
            sessionAgent_id: toStr(event?.session?.agent_id),
          },
          scope,
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
            agent_id: scope.agentId,
            channel,
            task_id: taskId,
            trace_id: traceId,
            metadata: {
              session_key: sessionKey || undefined,
              agent_id: scope.agentId || undefined,
              channel: channel || undefined,
              task_id: taskId || undefined,
              trace_id: traceId || undefined,
            },
          });
        }
        if (cfg.autoCaptureCompression) {
          const compression = await compressMessagesForCapture(cfg, event?.messages);
          debugLog(cfg, "agent_end.compression", {
            mode_requested: compression.modeRequested,
            mode_used: compression.modeUsed,
            turn_count: compression.turnCount,
            source_chars: compression.sourceChars,
            output_chars: compression.content.length,
            latency_ms: compression.latencyMs,
            fallback_reason: compression.fallbackReason || undefined,
          });
          if (compression.content) {
            await writeMemory({
              memory_type: "context_compression",
              content: compression.content,
              sender: scope.sender,
              sender_name: scope.sender,
              role: "assistant",
              user_id: scope.userId,
              group_id: scope.groupId,
              agent_id: scope.agentId,
              channel,
              task_id: taskId,
              trace_id: traceId,
              metadata: {
                session_key: sessionKey || undefined,
                agent_id: scope.agentId || undefined,
                channel: channel || undefined,
                task_id: taskId || undefined,
                trace_id: traceId || undefined,
                compression_mode_requested: compression.modeRequested,
                compression_mode_used: compression.modeUsed,
                compression_turn_count: compression.turnCount,
                compression_source_chars: compression.sourceChars,
                compression_latency_ms: compression.latencyMs,
                compression_fallback_reason: compression.fallbackReason || undefined,
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
    version: "0.1.3",
  };
}
