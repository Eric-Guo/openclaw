import type { StreamFn } from "@mariozechner/pi-agent-core";
import { streamSimple } from "@mariozechner/pi-ai";
import type { ProviderWrapStreamFnContext } from "openclaw/plugin-sdk/plugin-entry";

const TOOL_CALLS_SECTION_BEGIN = "<|tool_calls_section_begin|>";
const TOOL_CALLS_SECTION_END = "<|tool_calls_section_end|>";
const TOOL_CALL_BEGIN = "<|tool_call_begin|>";
const TOOL_CALL_ARGUMENT_BEGIN = "<|tool_call_argument_begin|>";
const TOOL_CALL_END = "<|tool_call_end|>";
const INLINE_TOOL_CALL_ID_PREFIX = "call_kimi_inline_";

type KimiToolCallBlock = {
  type: "toolCall";
  id: string;
  name: string;
  arguments: Record<string, unknown>;
};

function stripTaggedToolCallCounter(value: string): string {
  return value.trim().replace(/:\d+$/, "");
}

function resolveCaseInsensitiveAllowedToolName(
  rawName: string,
  allowedToolNames?: Set<string>,
): string | null {
  if (!allowedToolNames || allowedToolNames.size === 0) {
    return null;
  }
  const folded = rawName.trim().toLowerCase();
  let match: string | null = null;
  for (const allowedName of allowedToolNames) {
    if (allowedName.toLowerCase() !== folded) {
      continue;
    }
    if (match && match !== allowedName) {
      return null;
    }
    match = allowedName;
  }
  return match;
}

function normalizeToolNameCandidate(value: string): string {
  return value.trim().replace(/\//g, ".");
}

function resolveTaggedToolCallName(rawName: string, allowedToolNames?: Set<string>): string {
  const trimmed = rawName.trim();
  if (!trimmed || !allowedToolNames || allowedToolNames.size === 0) {
    return trimmed;
  }
  if (allowedToolNames.has(trimmed)) {
    return trimmed;
  }

  const normalizedDelimiter = normalizeToolNameCandidate(trimmed);
  if (allowedToolNames.has(normalizedDelimiter)) {
    return normalizedDelimiter;
  }

  const segments = normalizedDelimiter
    .split(".")
    .map((segment) => segment.trim())
    .filter(Boolean);
  for (let index = 1; index < segments.length; index += 1) {
    const suffix = segments.slice(index).join(".");
    if (allowedToolNames.has(suffix)) {
      return suffix;
    }
    const caseInsensitiveSuffix = resolveCaseInsensitiveAllowedToolName(suffix, allowedToolNames);
    if (caseInsensitiveSuffix) {
      return caseInsensitiveSuffix;
    }
  }

  return (
    resolveCaseInsensitiveAllowedToolName(trimmed, allowedToolNames) ??
    resolveCaseInsensitiveAllowedToolName(normalizedDelimiter, allowedToolNames) ??
    trimmed
  );
}

function parseKimiTaggedToolCallSection(
  sectionText: string,
  allowedToolNames?: Set<string>,
): KimiToolCallBlock[] | null {
  const trimmed = sectionText.trim();
  if (!trimmed.startsWith(TOOL_CALLS_SECTION_BEGIN) || !trimmed.endsWith(TOOL_CALLS_SECTION_END)) {
    return null;
  }

  let cursor = TOOL_CALLS_SECTION_BEGIN.length;
  const sectionEndIndex = trimmed.length - TOOL_CALLS_SECTION_END.length;
  const toolCalls: KimiToolCallBlock[] = [];

  while (cursor < sectionEndIndex) {
    while (cursor < sectionEndIndex && /\s/.test(trimmed[cursor] ?? "")) {
      cursor += 1;
    }
    if (cursor >= sectionEndIndex) {
      break;
    }
    if (!trimmed.startsWith(TOOL_CALL_BEGIN, cursor)) {
      return null;
    }

    const nameStart = cursor + TOOL_CALL_BEGIN.length;
    const argMarkerIndex = trimmed.indexOf(TOOL_CALL_ARGUMENT_BEGIN, nameStart);
    if (argMarkerIndex < 0 || argMarkerIndex >= sectionEndIndex) {
      return null;
    }

    const rawId = trimmed.slice(nameStart, argMarkerIndex).trim();
    if (!rawId) {
      return null;
    }

    const argsStart = argMarkerIndex + TOOL_CALL_ARGUMENT_BEGIN.length;
    const callEndIndex = trimmed.indexOf(TOOL_CALL_END, argsStart);
    if (callEndIndex < 0 || callEndIndex > sectionEndIndex) {
      return null;
    }

    const rawArgs = trimmed.slice(argsStart, callEndIndex).trim();
    let parsedArgs: unknown;
    try {
      parsedArgs = JSON.parse(rawArgs);
    } catch {
      return null;
    }
    if (!parsedArgs || typeof parsedArgs !== "object" || Array.isArray(parsedArgs)) {
      return null;
    }

    const name = resolveTaggedToolCallName(stripTaggedToolCallCounter(rawId), allowedToolNames);
    if (!name) {
      return null;
    }

    toolCalls.push({
      type: "toolCall",
      id: rawId,
      name,
      arguments: parsedArgs as Record<string, unknown>,
    });

    cursor = callEndIndex + TOOL_CALL_END.length;
  }

  return toolCalls.length > 0 ? toolCalls : null;
}

type KimiRewrittenTextBlock =
  | KimiToolCallBlock
  | {
      type: "text";
      text: string;
    };

type BalancedJsonSlice = {
  json: string;
  endIndex: number;
};

type ParenthesizedJsonSlice = {
  json: string;
  endIndex: number;
};

function pushTextBlock(blocks: KimiRewrittenTextBlock[], text: string): void {
  if (!text.trim()) {
    return;
  }
  blocks.push({
    type: "text",
    text,
  });
}

function parseKimiTaggedContentBlocks(
  text: string,
  allowedToolNames?: Set<string>,
): KimiRewrittenTextBlock[] | null {
  const sectionStart = text.indexOf(TOOL_CALLS_SECTION_BEGIN);
  if (sectionStart < 0) {
    return null;
  }

  const blocks: KimiRewrittenTextBlock[] = [];
  let cursor = 0;
  let foundSection = false;

  while (cursor < text.length) {
    const nextSectionStart = text.indexOf(TOOL_CALLS_SECTION_BEGIN, cursor);
    if (nextSectionStart < 0) {
      pushTextBlock(blocks, text.slice(cursor));
      break;
    }

    pushTextBlock(blocks, text.slice(cursor, nextSectionStart));

    const nextSectionEnd = text.indexOf(TOOL_CALLS_SECTION_END, nextSectionStart);
    if (nextSectionEnd < 0) {
      return null;
    }

    const sectionText = text.slice(
      nextSectionStart,
      nextSectionEnd + TOOL_CALLS_SECTION_END.length,
    );
    const parsedSection = parseKimiTaggedToolCallSection(sectionText, allowedToolNames);
    if (!parsedSection) {
      return null;
    }

    blocks.push(...parsedSection);
    foundSection = true;
    cursor = nextSectionEnd + TOOL_CALLS_SECTION_END.length;
  }

  return foundSection ? blocks : null;
}

function extractBalancedJsonSlice(text: string, startIndex: number): BalancedJsonSlice | null {
  let cursor = startIndex;
  while (cursor < text.length && /\s/.test(text[cursor] ?? "")) {
    cursor += 1;
  }
  if ((text[cursor] ?? "") !== "{") {
    return null;
  }

  let depth = 0;
  let inString = false;
  let escaped = false;
  for (let index = cursor; index < text.length; index += 1) {
    const char = text[index];
    if (char === undefined) {
      break;
    }
    if (inString) {
      if (escaped) {
        escaped = false;
      } else if (char === "\\") {
        escaped = true;
      } else if (char === '"') {
        inString = false;
      }
      continue;
    }
    if (char === '"') {
      inString = true;
      continue;
    }
    if (char === "{") {
      depth += 1;
      continue;
    }
    if (char === "}") {
      depth -= 1;
      if (depth === 0) {
        return {
          json: text.slice(cursor, index + 1),
          endIndex: index + 1,
        };
      }
    }
  }

  return null;
}

function extractParenthesizedJsonSlice(
  text: string,
  openingParenIndex: number,
): ParenthesizedJsonSlice | null {
  if ((text[openingParenIndex] ?? "") !== "(") {
    return null;
  }

  const jsonSlice = extractBalancedJsonSlice(text, openingParenIndex + 1);
  if (!jsonSlice) {
    return null;
  }

  let cursor = jsonSlice.endIndex;
  while (cursor < text.length && /\s/.test(text[cursor] ?? "")) {
    cursor += 1;
  }
  if ((text[cursor] ?? "") !== ")") {
    return null;
  }

  return {
    json: jsonSlice.json,
    endIndex: cursor + 1,
  };
}

function hasStandaloneToolCallBoundary(previousChar: string | undefined): boolean {
  return !previousChar || !/[\p{L}\p{N}._/-]/u.test(previousChar);
}

const INLINE_TOOL_CALL_RE = /((?:functions?|tools?)(?:[./_-][a-zA-Z0-9_-]+)+:\d+)\s+/g;
const PAREN_TOOL_CALL_RE =
  /((?:(?:functions?|tools?)[./_-])?[a-zA-Z][a-zA-Z0-9_.-]*(?::\d+)?)\s*\(/g;

function parseKimiInlineToolCalls(
  text: string,
  allowedToolNames?: Set<string>,
): KimiRewrittenTextBlock[] | null {
  const blocks: KimiRewrittenTextBlock[] = [];
  let cursor = 0;
  let changed = false;

  while (cursor < text.length) {
    INLINE_TOOL_CALL_RE.lastIndex = cursor;
    const match = INLINE_TOOL_CALL_RE.exec(text);
    if (!match) {
      pushTextBlock(blocks, text.slice(cursor));
      break;
    }

    const rawId = match[1]?.trim();
    if (!rawId) {
      pushTextBlock(blocks, text.slice(cursor));
      break;
    }

    const matchIndex = match.index;
    const previousChar = matchIndex > 0 ? text[matchIndex - 1] : undefined;
    if (!hasStandaloneToolCallBoundary(previousChar)) {
      cursor = matchIndex + 1;
      continue;
    }

    const parsedArguments = extractBalancedJsonSlice(text, INLINE_TOOL_CALL_RE.lastIndex);
    if (!parsedArguments) {
      cursor = matchIndex + 1;
      continue;
    }

    let argumentsObject: unknown;
    try {
      argumentsObject = JSON.parse(parsedArguments.json);
    } catch {
      cursor = matchIndex + 1;
      continue;
    }
    if (!argumentsObject || typeof argumentsObject !== "object" || Array.isArray(argumentsObject)) {
      cursor = matchIndex + 1;
      continue;
    }

    pushTextBlock(blocks, text.slice(cursor, matchIndex));
    blocks.push({
      type: "toolCall",
      id: rawId,
      name: resolveTaggedToolCallName(stripTaggedToolCallCounter(rawId), allowedToolNames),
      arguments: argumentsObject as Record<string, unknown>,
    });
    changed = true;
    cursor = parsedArguments.endIndex;
  }

  return changed ? blocks : null;
}

function isLikelyAllowedParenToolCallName(
  rawId: string,
  resolvedName: string,
  allowedToolNames?: Set<string>,
): boolean {
  if (!allowedToolNames || allowedToolNames.size === 0) {
    return true;
  }
  if (allowedToolNames.has(resolvedName)) {
    return true;
  }
  return /^(?:functions?|tools?)[./_-]/i.test(rawId);
}

function createSyntheticInlineToolCallId(counter: number): string {
  return `${INLINE_TOOL_CALL_ID_PREFIX}${counter}`;
}

function parseKimiParenthesizedToolCalls(
  text: string,
  allowedToolNames?: Set<string>,
): KimiRewrittenTextBlock[] | null {
  const blocks: KimiRewrittenTextBlock[] = [];
  let cursor = 0;
  let changed = false;
  let syntheticIdCounter = 0;

  while (cursor < text.length) {
    PAREN_TOOL_CALL_RE.lastIndex = cursor;
    const match = PAREN_TOOL_CALL_RE.exec(text);
    if (!match) {
      pushTextBlock(blocks, text.slice(cursor));
      break;
    }

    const rawId = match[1]?.trim();
    if (!rawId) {
      pushTextBlock(blocks, text.slice(cursor));
      break;
    }

    const matchIndex = match.index;
    const previousChar = matchIndex > 0 ? text[matchIndex - 1] : undefined;
    if (!hasStandaloneToolCallBoundary(previousChar)) {
      cursor = matchIndex + 1;
      continue;
    }

    const resolvedName = resolveTaggedToolCallName(
      stripTaggedToolCallCounter(rawId),
      allowedToolNames,
    );
    if (!isLikelyAllowedParenToolCallName(rawId, resolvedName, allowedToolNames)) {
      cursor = matchIndex + 1;
      continue;
    }

    const openingParenIndex = PAREN_TOOL_CALL_RE.lastIndex - 1;
    const parsedArguments = extractParenthesizedJsonSlice(text, openingParenIndex);
    if (!parsedArguments) {
      cursor = matchIndex + 1;
      continue;
    }

    let argumentsObject: unknown;
    try {
      argumentsObject = JSON.parse(parsedArguments.json);
    } catch {
      cursor = matchIndex + 1;
      continue;
    }
    if (!argumentsObject || typeof argumentsObject !== "object" || Array.isArray(argumentsObject)) {
      cursor = matchIndex + 1;
      continue;
    }

    pushTextBlock(blocks, text.slice(cursor, matchIndex));
    blocks.push({
      type: "toolCall",
      id:
        /:\d+$/.test(rawId) || rawId.startsWith(INLINE_TOOL_CALL_ID_PREFIX)
          ? rawId
          : createSyntheticInlineToolCallId(++syntheticIdCounter),
      name: resolvedName,
      arguments: argumentsObject as Record<string, unknown>,
    });
    changed = true;
    cursor = parsedArguments.endIndex;
  }

  return changed ? blocks : null;
}

function rewriteKimiInlineToolText(
  text: string,
  allowedToolNames?: Set<string>,
): KimiRewrittenTextBlock[] | null {
  const parsers = [parseKimiInlineToolCalls, parseKimiParenthesizedToolCalls] as const;
  let blocks: KimiRewrittenTextBlock[] = [{ type: "text", text }];
  let changed = false;

  for (const parser of parsers) {
    const nextBlocks: KimiRewrittenTextBlock[] = [];
    let parserChanged = false;

    for (const block of blocks) {
      if (block.type !== "text") {
        nextBlocks.push(block);
        continue;
      }

      const parsedBlocks = parser(block.text, allowedToolNames);
      if (!parsedBlocks) {
        nextBlocks.push(block);
        continue;
      }

      nextBlocks.push(...parsedBlocks);
      parserChanged = true;
    }

    blocks = nextBlocks;
    changed ||= parserChanged;
  }

  return changed ? blocks : null;
}

function rewriteKimiTaggedToolCallsInMessage(
  message: unknown,
  allowedToolNames?: Set<string>,
): void {
  if (!message || typeof message !== "object") {
    return;
  }

  const content = (message as { content?: unknown }).content;
  if (!Array.isArray(content)) {
    return;
  }

  let changed = false;
  const nextContent: unknown[] = [];
  for (const block of content) {
    if (!block || typeof block !== "object") {
      nextContent.push(block);
      continue;
    }
    const typedBlock = block as { type?: unknown; text?: unknown };
    if (typedBlock.type !== "text" || typeof typedBlock.text !== "string") {
      nextContent.push(block);
      continue;
    }

    const taggedBlocks = parseKimiTaggedContentBlocks(typedBlock.text, allowedToolNames);
    const candidateBlocks = taggedBlocks ?? [{ type: "text", text: typedBlock.text }];
    const parsed: KimiRewrittenTextBlock[] = [];
    let blockChanged = taggedBlocks !== null;

    for (const candidate of candidateBlocks) {
      if (candidate.type !== "text") {
        parsed.push(candidate);
        continue;
      }
      const inlineBlocks = rewriteKimiInlineToolText(candidate.text, allowedToolNames);
      if (!inlineBlocks) {
        parsed.push(candidate);
        continue;
      }
      parsed.push(...inlineBlocks);
      blockChanged = true;
    }

    if (!blockChanged) {
      nextContent.push(block);
      continue;
    }

    nextContent.push(...parsed);
    changed = true;
  }

  if (!changed) {
    return;
  }

  (message as { content: unknown[] }).content = nextContent;
  const typedMessage = message as { stopReason?: unknown };
  if (typedMessage.stopReason === "stop") {
    typedMessage.stopReason = "toolUse";
  }
}

function wrapKimiTaggedToolCalls(
  stream: ReturnType<typeof streamSimple>,
  allowedToolNames?: Set<string>,
): ReturnType<typeof streamSimple> {
  const originalResult = stream.result.bind(stream);
  stream.result = async () => {
    const message = await originalResult();
    rewriteKimiTaggedToolCallsInMessage(message, allowedToolNames);
    return message;
  };

  const originalAsyncIterator = stream[Symbol.asyncIterator].bind(stream);
  (stream as { [Symbol.asyncIterator]: typeof originalAsyncIterator })[Symbol.asyncIterator] =
    function () {
      const iterator = originalAsyncIterator();
      return {
        async next() {
          const result = await iterator.next();
          if (!result.done && result.value && typeof result.value === "object") {
            const event = result.value as {
              partial?: unknown;
              message?: unknown;
            };
            rewriteKimiTaggedToolCallsInMessage(event.partial, allowedToolNames);
            rewriteKimiTaggedToolCallsInMessage(event.message, allowedToolNames);
          }
          return result;
        },
        async return(value?: unknown) {
          return iterator.return?.(value) ?? { done: true as const, value: undefined };
        },
        async throw(error?: unknown) {
          return iterator.throw?.(error) ?? { done: true as const, value: undefined };
        },
      };
    };

  return stream;
}

export function createKimiToolCallMarkupWrapper(baseStreamFn: StreamFn | undefined): StreamFn {
  const underlying = baseStreamFn ?? streamSimple;
  return (model, context, options) => {
    const allowedToolNames = new Set(
      Array.isArray(context?.tools)
        ? context.tools
            .map((tool) =>
              tool && typeof tool === "object" ? (tool as { name?: unknown }).name : undefined,
            )
            .filter((name): name is string => typeof name === "string" && name.trim().length > 0)
        : [],
    );
    const maybeStream = underlying(model, context, options);
    if (maybeStream && typeof maybeStream === "object" && "then" in maybeStream) {
      return Promise.resolve(maybeStream).then((stream) =>
        wrapKimiTaggedToolCalls(stream, allowedToolNames),
      );
    }
    return wrapKimiTaggedToolCalls(maybeStream, allowedToolNames);
  };
}

export function wrapKimiProviderStream(ctx: ProviderWrapStreamFnContext): StreamFn {
  return createKimiToolCallMarkupWrapper(ctx.streamFn);
}
