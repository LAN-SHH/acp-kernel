import type {
  CcrConfig,
  Config,
  CompressionState,
  CoreMessage,
} from "./types.js";
import { refForRaw, BLOCKED_REF } from "./refs.js";
import { ACP_TOOL_NAMES } from "./compress-tools.js";
import { isMessageProtected, matchToolPattern } from "./protected.js";
import { hasStoredRef, retrieveByRef, storeOriginal } from "./content-store.js";
import type {
  MessageContentStore,
  RetrieveResult,
  StoredEntry,
} from "./content-store.js";
import type { NodeIO, PipelineContext, PipelineNode } from "./pipeline.js";

/**
 * CCR — content-cached retrieval (issue #352 / billion-context#1097).
 *
 * Lossless alternative to absorb's lossy distillation: oversized tool results
 * are stored once at arrival in the per-session MessageContentStore, and the
 * visible copy is replaced with a deterministic placeholder carrying enough
 * signal (kind, size, command/head preview, ref) to judge relevance without
 * retrieving. The model pulls the original back via the retrieve tool; the
 * retrieved text rides back as an ephemeral trailing message that never
 * consumes a ref and never enters the fold space.
 *
 * Replace-once-at-arrival: the pipeline node only acts on results it has not
 * stored yet (marker check + store lookup). After replacement the visible
 * bytes never change again → prefix-cache stable.
 */

export const RETRIEVE_TOOL_NAME = "acp_retrieve";

export const DEFAULT_CCR_CONFIG: CcrConfig = {
  enabled: false,
  toolName: RETRIEVE_TOOL_NAME,
  minToolTokens: 4000,
  excludeTools: [],
  maxHeadChars: 96,
};

export function resolveCcrConfig(config: Config): CcrConfig {
  return { ...DEFAULT_CCR_CONFIG, ...config.ccr };
}

/** Marker embedded in every placeholder. Exclusion signal for absorb
 *  (ID-reference wins over distill); recognition is via the STRICT parser
 *  below, never a substring scan (billion-context#1456). */
export const STORED_PLACEHOLDER_MARKER = "[acp-stored";

/** Reserved id prefix for ephemeral retrieval injections (mirrors
 *  acp_summary_): no ref is ever assigned, fold-space collection skips them. */
export const RETRIEVED_ID_PREFIX = "acp_retrieved_";

const KIND_LABELS: Record<string, string> = {
  bash: "shell output",
  shell: "shell output",
  exec: "shell output",
  execute_command: "shell output",
  run: "shell output",
  terminal: "shell output",
  read: "file read",
  read_file: "file read",
  cat: "file read",
  open: "file read",
  grep: "search output",
  rg: "search output",
  search: "search output",
  glob: "search output",
  find: "search output",
  webfetch: "web fetch",
  web_fetch: "web fetch",
  fetch: "web fetch",
  curl: "web fetch",
};

export function classifyKind(toolName?: string): string {
  if (!toolName) return "tool result";
  return KIND_LABELS[toolName.toLowerCase()] ?? "tool result";
}

function groupThousands(value: number): string {
  return String(Math.max(0, Math.round(value))).replace(
    /\B(?=(\d{3})+(?!\d))/g,
    ",",
  );
}

/** Collapse whitespace to single spaces and cap length. Deterministic — the
 *  preview is baked into the placeholder at arrival and must be byte-stable. */
export function normalizeHead(text: string, maxChars: number): string {
  const singleLine = (text || "").replace(/\s+/g, " ").trim();
  if (singleLine.length <= maxChars) return singleLine;
  return singleLine.slice(0, maxChars) + "…";
}

const COMMAND_FIELDS = ["command", "cmd", "script", "query", "path", "url"];

/** Best-effort extraction of the call's subject (command/path/query) from the
 *  paired tool-call args JSON. Returns undefined when args are not JSON or no
 *  known field holds a short string — the head preview covers that case. */
export function extractCommand(
  toolCallText: string | undefined,
  maxChars: number,
): string | undefined {
  if (!toolCallText) return undefined;
  let parsed: unknown;
  try {
    parsed = JSON.parse(toolCallText);
  } catch {
    return undefined;
  }
  if (typeof parsed !== "object" || parsed === null) return undefined;
  const record = parsed as Record<string, unknown>;
  for (const field of COMMAND_FIELDS) {
    const value = record[field];
    if (typeof value === "string" && value.trim().length > 0) {
      const normalized = value.replace(/\s+/g, " ").trim();
      return normalized.length <= maxChars
        ? normalized
        : normalized.slice(0, maxChars) + "…";
    }
  }
  return undefined;
}

export interface StoredPlaceholderInput {
  ref: string;
  kind: string;
  tokens: number;
  head: string;
  command?: string;
  retrieveToolName: string;
}

/** Deterministic two-line placeholder. No timestamps, no randomness — the
 *  same inputs always produce identical bytes (prefix-cache friendly). */
export function buildStoredPlaceholder(input: StoredPlaceholderInput): string {
  const title = input.command ?? input.head;
  const titlePart = title ? ` \`${title}\`` : "";
  return (
    `📦 ${STORED_PLACEHOLDER_MARKER} #${input.ref} · ${input.kind} · ` +
    `${groupThousands(input.tokens)} tok]${titlePart}` +
    `\n   → ${input.retrieveToolName}("${input.ref}") returns the full text`
  );
}

// Built from hex escapes: literal history-tag sequences in source get mangled
// by content sanitizers upstream of git push (same reason render-refs.ts uses LT/GT).
const LEADING_TAG_RE = new RegExp("^\\x3cacp [^>]*>[^\\x3c]*\\x3c\\/acp>\\n?");

export function stripLeadingTag(text: string): string {
  return text.replace(LEADING_TAG_RE, "");
}

/** Parsed canonical CCR placeholder (whole-body shape of
 *  buildStoredPlaceholder output). */
export interface ParsedStoredPlaceholder {
  ref: string;
  kind: string;
  tokens: number;
  retrieveToolName: string;
  /** Backtick title from line 1 (command/head preview), when present. */
  title?: string;
}

// Strict whole-body grammar of buildStoredPlaceholder output. Refs are always
// 5-digit zero-padded (refs.ts REF_WIDTH); tokens use groupThousands grouping
// (no leading zeros, comma groups); both lines cite the same ref. A bare
// substring mention of "[acp-stored" anywhere in larger text matches NOTHING
// here (billion-context#1456).
const PLACEHOLDER_LINE1_RE = new RegExp(
  "^📦 \\[acp-stored #(m\\d{5}) · (.+?) · (0|[1-9]\\d{0,2}(?:,\\d{3})*) tok\\](?: `(.+)`)?$",
);
const PLACEHOLDER_LINE2_RE = /^   → (.+?)\("(m\d{5})"\) returns the full text$/;

/** Parse a WHOLE-BODY canonical CCR placeholder; null otherwise.
 *
 *  Accepts exactly the two-line structure emitted by buildStoredPlaceholder,
 *  modulo one trusted leading render tag (hosts round-trip rendered output)
 *  and one optional trailing newline. Ordinary source, docs or logs that
 *  merely MENTION the marker — including embedded full placeholder examples
 *  with surrounding text — parse to null. The residual ambiguity is only user
 *  content byte-equal to a canonical placeholder, which no pure-text protocol
 *  can disambiguate (structured provenance is the long-term fix; deliberately
 *  out of scope here). Callers that know the message's own ref MUST compare
 *  `parsed.ref === ownRef` before trusting the result as an internal
 *  placeholder. */
export function parseStoredPlaceholder(
  text: string,
): ParsedStoredPlaceholder | null {
  const body = stripLeadingTag(text);
  const trimmed = body.endsWith("\n") ? body.slice(0, -1) : body;
  const nl = trimmed.indexOf("\n");
  if (nl <= 0) return null;
  const line1 = trimmed.slice(0, nl);
  const line2 = trimmed.slice(nl + 1);
  if (line2.includes("\n")) return null;
  const head = PLACEHOLDER_LINE1_RE.exec(line1);
  if (!head) return null;
  const hint = PLACEHOLDER_LINE2_RE.exec(line2);
  if (!hint) return null;
  const ref = head[1];
  const kind = head[2];
  const tokenGroup = head[3];
  const toolName = hint[1];
  const hintRef = hint[2];
  if (
    !ref ||
    !hintRef ||
    ref !== hintRef ||
    !kind ||
    !tokenGroup ||
    !toolName
  ) {
    return null;
  }
  const parsed: ParsedStoredPlaceholder = {
    ref,
    kind,
    tokens: Number(tokenGroup.replace(/,/g, "")),
    retrieveToolName: toolName,
  };
  const title = head[4];
  if (title !== undefined) parsed.title = title;
  return parsed;
}

/** True iff the WHOLE body is a canonical CCR placeholder — i.e. the kernel
 *  already replaced this tool result and it must not be stored, re-compressed
 *  or absorbed again. Replaces the former `includes("[acp-stored")` substring
 *  check that misclassified ordinary text merely mentioning the marker
 *  (billion-context#1456). */
export function isStoredPlaceholderText(text: string): boolean {
  return parseStoredPlaceholder(text) !== null;
}

export function retrievedMessageId(ref: string): string {
  return RETRIEVED_ID_PREFIX + ref;
}

export function isRetrievedMessage(message: CoreMessage): boolean {
  return (
    message.id.startsWith(RETRIEVED_ID_PREFIX) &&
    message.role === "system" &&
    message.contentType === "text"
  );
}

export interface RetrievalInjection {
  /** Short deterministic ack — rides as the tool result so OpenAI-family
   *  wire pairing (every tool_call needs a response) stays intact. */
  ackText: string;
  /** Full-text trailing request-only message. Hosts strip it before
   *  persisting (same channel as nudge); if it round-trips anyway it is
   *  structurally excluded from refs and the fold space. */
  injection: CoreMessage;
}

export function buildRetrievalInjection(
  ref: string,
  entry: StoredEntry,
  text: string,
): RetrievalInjection {
  const header = `[acp-retrieved #${ref} · ${entry.kind} · ${groupThousands(entry.tokens)} tok]`;
  return {
    ackText: `retrieved ${ref}: ${groupThousands(entry.tokens)} tok (${entry.chars} chars)`,
    injection: {
      id: retrievedMessageId(ref),
      role: "system",
      contentType: "text",
      text: `${header}\n${text}`,
    },
  };
}

export interface ApplyRetrieveInput {
  store: MessageContentStore;
  ref: string;
}

export type ApplyRetrieveResult =
  | {
      ok: true;
      text: string;
      ackText: string;
      injection: CoreMessage;
      entry: StoredEntry;
    }
  | { ok: false; reason: "not-found"; ackText: string };

/** Kernel primitive behind acp_retrieve: resolve a ref to its stored
 *  original plus the wire-safe injection pair. Hallucinated refs → not-found
 *  (cost: one tool call, by design). */
export function applyRetrieve(input: ApplyRetrieveInput): ApplyRetrieveResult {
  const ref = input.ref.trim();
  const found: RetrieveResult = retrieveByRef(input.store, ref);
  if (!found.ok) {
    return {
      ok: false,
      reason: "not-found",
      ackText: `retrieve ${ref}: not found — no stored original for this ref`,
    };
  }
  const built = buildRetrievalInjection(ref, found.entry, found.text);
  return {
    ok: true,
    text: found.text,
    ackText: built.ackText,
    injection: built.injection,
    entry: found.entry,
  };
}

export interface StoreLargeResultsInput {
  messages: CoreMessage[];
  state: CompressionState;
  store: MessageContentStore;
  config: Config;
  countTokens: (text: string) => number;
}

export interface StoreLargeResultsResult {
  messages: CoreMessage[];
  store: MessageContentStore;
  storedCount: number;
}

/** Core transform: store oversized tool-result originals and replace their
 *  visible text with placeholders. Pure over its inputs; returns new arrays.
 *  Only shrinks tool-result content in place — message ids, roles, toolCallId
 *  and the assistant tool_calls half are untouched (tool-pair integrity). */
export function storeLargeResults(
  input: StoreLargeResultsInput,
): StoreLargeResultsResult {
  const cfg = resolveCcrConfig(input.config);
  if (!cfg.enabled)
    return { messages: input.messages, store: input.store, storedCount: 0 };

  let current = input.store;
  let storedCount = 0;
  const callById = new Map<string, CoreMessage>();
  for (const message of input.messages) {
    if (message.contentType === "tool-call" && message.toolCallId) {
      callById.set(message.toolCallId, message);
    }
  }

  const updated = input.messages.map((message) => {
    if (message.contentType !== "tool-result") return message;
    const text = message.text ?? "";
    if (text.length === 0) return message;
    const ref = refForRaw(input.state.messageRefs, message.id);
    // (#1456) strict gate: only a WHOLE-BODY canonical placeholder carrying
    // this message's own ref skips storage. Ordinary content that merely
    // mentions the marker (source defining it, docs quoting an example) falls
    // through to normal storage below.
    const placeholder = parseStoredPlaceholder(text);
    if (placeholder && (!ref || placeholder.ref === ref)) return message;
    if (!message.toolCallId) return message;
    const toolName = message.toolName;
    if (
      toolName &&
      (ACP_TOOL_NAMES.has(toolName) || toolName === cfg.toolName)
    ) {
      return message;
    }
    if (
      toolName &&
      cfg.excludeTools.some((pattern) => matchToolPattern(toolName, pattern))
    ) {
      return message;
    }
    if (isMessageProtected(message, input.config)) return message;
    const tokens = input.countTokens(text);
    if (tokens < cfg.minToolTokens) return message;
    if (!ref || ref === BLOCKED_REF) return message;
    if (hasStoredRef(current, ref)) return message;
    const kind = classifyKind(toolName);
    const head = normalizeHead(text, cfg.maxHeadChars);
    const command = extractCommand(
      callById.get(message.toolCallId)?.text,
      cfg.maxHeadChars,
    );
    current = storeOriginal(current, {
      ref,
      rawId: message.id,
      text,
      kind,
      toolName,
      tokens,
      head,
    });
    storedCount += 1;
    return {
      ...message,
      text: buildStoredPlaceholder({
        ref,
        kind,
        tokens,
        head,
        command,
        retrieveToolName: cfg.toolName,
      }),
    };
  });

  return { messages: updated, store: current, storedCount };
}

/** Effect shape carried in NodeIO.effects.ccr by ccrStoreNode. */
export interface CcrEffect {
  store: MessageContentStore;
  storedCount: number;
}

/** Pipeline node: runs after prune (new results live in the preserved tail,
 *  already referenced by assign-refs) and before absorb (placeholder-marked
 *  text is not an absorb candidate → ID-reference priority). */
export const ccrStoreNode: PipelineNode = {
  name: "ccr-store",
  enabled: (_io, ctx) => resolveCcrConfig(ctx.config).enabled,
  run(io: NodeIO, ctx: PipelineContext): NodeIO {
    const applied = storeLargeResults({
      messages: io.messages,
      state: io.state,
      store: ctx.contentStore,
      config: ctx.config,
      countTokens: ctx.countTokens,
    });
    const effect: CcrEffect = {
      store: applied.store,
      storedCount: applied.storedCount,
    };
    const stats =
      applied.storedCount > 0
        ? {
            ...io.state.stats,
            storedCount:
              (io.state.stats.storedCount ?? 0) + applied.storedCount,
          }
        : io.state.stats;
    return {
      ...io,
      messages: applied.messages,
      state: stats === io.state.stats ? io.state : { ...io.state, stats },
      effects: { ...io.effects, ccr: effect },
    };
  },
};

/** Adapter glue for v2 groundwork: persist originals of newly covered
 *  messages after applyCompression succeeds, so retrieve-by-ref works for
 *  folded content. First-write-wins keeps earlier CCR arrivals authoritative.
 *  Reasoning messages are skipped (thinking content is not retrievable). */
export function storeCoveredOriginals(
  store: MessageContentStore,
  messages: CoreMessage[],
  state: CompressionState,
  blockIds: readonly string[],
  countTokens: (text: string) => number,
  maxHeadChars: number = DEFAULT_CCR_CONFIG.maxHeadChars,
): MessageContentStore {
  const covered = new Set<string>();
  for (const block of state.blocks) {
    if (!block.active || !blockIds.includes(block.blockId)) continue;
    for (const id of block.effectiveMessageIds) covered.add(id);
  }
  let current = store;
  for (const message of messages) {
    if (!covered.has(message.id)) continue;
    if (message.contentType === "reasoning") continue;
    const text = stripLeadingTag(message.text ?? "");
    if (text.length === 0) continue;
    const ref = refForRaw(state.messageRefs, message.id);
    if (!ref || ref === BLOCKED_REF) continue;
    // Placeholder text is never an original: a placeholder whose ref is
    // missing from the store means the true original was already lost (fork
    // without the store envelope, corrupted companion file, host migration
    // without the store). Storing the placeholder bytes would turn every
    // later retrieve-by-ref into a fake hit echoing the placeholder itself —
    // the symmetric skip mirrors the arrival-time path in storeLargeResults.
    // (#1456) strict gate: only a WHOLE-BODY canonical placeholder carrying
    // THIS message's own ref skips; ordinary content embedding a marker
    // example, or placeholder-shaped text citing a foreign ref, stores as a
    // genuine original.
    if (parseStoredPlaceholder(text)?.ref === ref) continue;
    current = storeOriginal(current, {
      ref,
      rawId: message.id,
      text,
      kind: classifyKind(message.toolName),
      toolName: message.toolName,
      tokens: countTokens(text),
      head: normalizeHead(text, maxHeadChars),
    });
  }
  return current;
}

/** Bump the cumulative retrieval counter (host calls after a successful
 *  applyRetrieve; feeds the status report's retrieve-rate metric). */
export function noteRetrieval(state: CompressionState): CompressionState {
  return {
    ...state,
    stats: {
      ...state.stats,
      retrievalCount: (state.stats.retrievalCount ?? 0) + 1,
    },
  };
}
