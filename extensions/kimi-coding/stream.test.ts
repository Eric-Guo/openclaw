import type { StreamFn } from "@mariozechner/pi-agent-core";
import { Type, type Context, type Model } from "@mariozechner/pi-ai";
import { describe, expect, it } from "vitest";
import { createKimiToolCallMarkupWrapper, wrapKimiProviderStream } from "./stream.js";

type FakeStream = {
  result: () => Promise<unknown>;
  [Symbol.asyncIterator]: () => AsyncIterator<unknown>;
};

const TEST_TOOLS = [
  {
    name: "read",
    description: "Read a file",
    parameters: Type.Object({ file_path: Type.String() }),
  },
  {
    name: "write",
    description: "Write a file",
    parameters: Type.Object({
      file_path: Type.String(),
      content: Type.String(),
    }),
  },
  {
    name: "exec",
    description: "Execute a command",
    parameters: Type.Object({ command: Type.String() }),
  },
  {
    name: "browser",
    description: "Control browser",
    parameters: Type.Object({
      action: Type.String(),
      url: Type.Optional(Type.String()),
    }),
  },
] satisfies Context["tools"];

function createFakeStream(params: { events: unknown[]; resultMessage: unknown }): FakeStream {
  return {
    async result() {
      return params.resultMessage;
    },
    [Symbol.asyncIterator]() {
      return (async function* () {
        for (const event of params.events) {
          yield event;
        }
      })();
    },
  };
}

const KIMI_TOOL_TEXT =
  ' <|tool_calls_section_begin|> <|tool_call_begin|> functions.read:0 <|tool_call_argument_begin|> {"file_path":"./package.json"} <|tool_call_end|> <|tool_calls_section_end|>';
const KIMI_MULTI_TOOL_TEXT =
  ' <|tool_calls_section_begin|> <|tool_call_begin|> functions.read:0 <|tool_call_argument_begin|> {"file_path":"./package.json"} <|tool_call_end|> <|tool_call_begin|> functions.write:1 <|tool_call_argument_begin|> {"file_path":"./out.txt","content":"done"} <|tool_call_end|> <|tool_calls_section_end|>';
const KIMI_BROWSER_TOOL_TEXT =
  '我来用浏览器打开这篇文章看看。<|tool_calls_section_begin|><|tool_call_begin|>functions.browser:0<|tool_call_argument_begin|>{"action":"browse","url":"https://zhuanlan.zhihu.com/p/2022015752258027715"}<|tool_call_end|><|tool_calls_section_end|>';
const KIMI_MEMORY_READ_TEXT =
  '<|tool_calls_section_begin|><|tool_call_begin|>functions.read:0<|tool_call_argument_begin|>{"file_path":"/Users/guoshuyi/.openclaw/workspace/SOUL.md"}<|tool_call_end|><|tool_call_begin|>functions.read:1<|tool_call_argument_begin|>{"file_path":"/Users/guoshuyi/.openclaw/workspace/USER.md"}<|tool_call_end|><|tool_call_begin|>functions.read:2<|tool_call_argument_begin|>{"file_path":"/Users/guoshuyi/.openclaw/workspace/MEMORY.md"}<|tool_call_end|><|tool_call_begin|>functions.read:3<|tool_call_argument_begin|>{"file_path":"/Users/guoshuyi/.openclaw/workspace/memory/2026-04-08.md"}<|tool_call_end|><|tool_call_begin|>functions.read:4<|tool_call_argument_begin|>{"file_path":"/Users/guoshuyi/.openclaw/workspace/memory/2026-04-07.md"}<|tool_call_end|><|tool_calls_section_end|>\n晚上好，过纯中。🏠 刚读完今天的记忆文件，我已经准备好了。有什么想做的吗？';
const KIMI_MIXED_TOOL_TEXT = [
  "<|tool_calls_section_begin|>",
  '<|tool_call_begin|>functions.read:0<|tool_call_argument_begin|>{"file_path":"/Users/guoshuyi/.openclaw/workspace/SOUL.md"}<|tool_call_end|>',
  '<|tool_call_begin|>functions.read:1<|tool_call_argument_begin|>{"file_path":"/Users/guoshuyi/.openclaw/workspace/USER.md"}<|tool_call_end|>',
  "<|tool_calls_section_end|>\n",
  'functions.exec:5 {"command":"cat /Users/guoshuyi/.openclaw/workspace/SOUL.md"}  ',
  'functions.exec:6 {"command":"cat /Users/guoshuyi/.openclaw/workspace/memory/2026-04-08.md 2>/dev/null || echo \\"File not found\\""}\n',
  "晚上好！🏠 我是过家家，刚完成今天的记忆同步。有什么想聊的，或者需要我帮忙做什么？",
].join("");

describe("kimi tool-call markup wrapper", () => {
  it("converts tagged Kimi tool-call text into structured tool calls", async () => {
    const partial = {
      role: "assistant",
      content: [{ type: "text", text: KIMI_TOOL_TEXT }],
      stopReason: "stop",
    };
    const message = {
      role: "assistant",
      content: [{ type: "text", text: KIMI_TOOL_TEXT }],
      stopReason: "stop",
    };
    const finalMessage = {
      role: "assistant",
      content: [
        { type: "thinking", thinking: "Need to read the file first." },
        { type: "text", text: KIMI_TOOL_TEXT },
      ],
      stopReason: "stop",
    };

    const baseStreamFn: StreamFn = () =>
      createFakeStream({
        events: [{ type: "message_end", partial, message }],
        resultMessage: finalMessage,
      }) as ReturnType<StreamFn>;

    const wrapped = createKimiToolCallMarkupWrapper(baseStreamFn);
    const stream = wrapped(
      { api: "anthropic-messages", provider: "kimi", id: "k2p5" } as Model<"anthropic-messages">,
      { messages: [], tools: TEST_TOOLS } as Context,
      {},
    ) as FakeStream;

    const events: unknown[] = [];
    for await (const event of stream) {
      events.push(event);
    }
    const result = (await stream.result()) as {
      content: unknown[];
      stopReason: string;
    };

    expect(events).toEqual([
      {
        type: "message_end",
        partial: {
          role: "assistant",
          content: [
            {
              type: "toolCall",
              id: "functions.read:0",
              name: "read",
              arguments: { file_path: "./package.json" },
            },
          ],
          stopReason: "toolUse",
        },
        message: {
          role: "assistant",
          content: [
            {
              type: "toolCall",
              id: "functions.read:0",
              name: "read",
              arguments: { file_path: "./package.json" },
            },
          ],
          stopReason: "toolUse",
        },
      },
    ]);
    expect(result).toEqual({
      role: "assistant",
      content: [
        { type: "thinking", thinking: "Need to read the file first." },
        {
          type: "toolCall",
          id: "functions.read:0",
          name: "read",
          arguments: { file_path: "./package.json" },
        },
      ],
      stopReason: "toolUse",
    });
  });

  it("leaves normal assistant text unchanged", async () => {
    const finalMessage = {
      role: "assistant",
      content: [{ type: "text", text: "normal response" }],
      stopReason: "stop",
    };
    const baseStreamFn: StreamFn = () =>
      createFakeStream({
        events: [],
        resultMessage: finalMessage,
      }) as ReturnType<StreamFn>;

    const wrapped = createKimiToolCallMarkupWrapper(baseStreamFn);
    const stream = wrapped(
      { api: "anthropic-messages", provider: "kimi", id: "k2p5" } as Model<"anthropic-messages">,
      { messages: [], tools: TEST_TOOLS } as Context,
      {},
    ) as FakeStream;

    await expect(stream.result()).resolves.toBe(finalMessage);
  });

  it("supports async stream functions", async () => {
    const finalMessage = {
      role: "assistant",
      content: [{ type: "text", text: KIMI_TOOL_TEXT }],
      stopReason: "stop",
    };
    const baseStreamFn: StreamFn = async () =>
      createFakeStream({
        events: [],
        resultMessage: finalMessage,
      }) as ReturnType<StreamFn>;

    const wrapped = createKimiToolCallMarkupWrapper(baseStreamFn);
    const stream = (await wrapped(
      { api: "anthropic-messages", provider: "kimi", id: "k2p5" } as Model<"anthropic-messages">,
      { messages: [], tools: TEST_TOOLS } as Context,
      {},
    )) as FakeStream;

    await expect(stream.result()).resolves.toEqual({
      role: "assistant",
      content: [
        {
          type: "toolCall",
          id: "functions.read:0",
          name: "read",
          arguments: { file_path: "./package.json" },
        },
      ],
      stopReason: "toolUse",
    });
  });

  it("parses multiple tagged tool calls in one section", async () => {
    const finalMessage = {
      role: "assistant",
      content: [{ type: "text", text: KIMI_MULTI_TOOL_TEXT }],
      stopReason: "stop",
    };
    const baseStreamFn: StreamFn = () =>
      createFakeStream({
        events: [],
        resultMessage: finalMessage,
      }) as ReturnType<StreamFn>;

    const wrapped = createKimiToolCallMarkupWrapper(baseStreamFn);
    const stream = wrapped(
      { api: "anthropic-messages", provider: "kimi", id: "k2p5" } as Model<"anthropic-messages">,
      { messages: [], tools: TEST_TOOLS } as Context,
      {},
    ) as FakeStream;

    await expect(stream.result()).resolves.toEqual({
      role: "assistant",
      content: [
        {
          type: "toolCall",
          id: "functions.read:0",
          name: "read",
          arguments: { file_path: "./package.json" },
        },
        {
          type: "toolCall",
          id: "functions.write:1",
          name: "write",
          arguments: { file_path: "./out.txt", content: "done" },
        },
      ],
      stopReason: "toolUse",
    });
  });

  it("extracts tagged tool calls from assistant narration text", async () => {
    const finalMessage = {
      role: "assistant",
      content: [{ type: "text", text: KIMI_BROWSER_TOOL_TEXT }],
      stopReason: "stop",
    };
    const baseStreamFn: StreamFn = () =>
      createFakeStream({
        events: [],
        resultMessage: finalMessage,
      }) as ReturnType<StreamFn>;

    const wrapped = createKimiToolCallMarkupWrapper(baseStreamFn);
    const stream = wrapped(
      { api: "anthropic-messages", provider: "kimi", id: "k2p5" } as Model<"anthropic-messages">,
      { messages: [], tools: TEST_TOOLS } as Context,
      {},
    ) as FakeStream;

    await expect(stream.result()).resolves.toEqual({
      role: "assistant",
      content: [
        { type: "text", text: "我来用浏览器打开这篇文章看看。" },
        {
          type: "toolCall",
          id: "functions.browser:0",
          name: "browser",
          arguments: {
            action: "browse",
            url: "https://zhuanlan.zhihu.com/p/2022015752258027715",
          },
        },
      ],
      stopReason: "toolUse",
    });
  });

  it("keeps trailing assistant narration after tagged tool calls", async () => {
    const finalMessage = {
      role: "assistant",
      content: [{ type: "text", text: KIMI_MEMORY_READ_TEXT }],
      stopReason: "stop",
    };
    const baseStreamFn: StreamFn = () =>
      createFakeStream({
        events: [],
        resultMessage: finalMessage,
      }) as ReturnType<StreamFn>;

    const wrapped = createKimiToolCallMarkupWrapper(baseStreamFn);
    const stream = wrapped(
      { api: "anthropic-messages", provider: "kimi", id: "k2p5" } as Model<"anthropic-messages">,
      { messages: [], tools: TEST_TOOLS } as Context,
      {},
    ) as FakeStream;

    await expect(stream.result()).resolves.toEqual({
      role: "assistant",
      content: [
        {
          type: "toolCall",
          id: "functions.read:0",
          name: "read",
          arguments: { file_path: "/Users/guoshuyi/.openclaw/workspace/SOUL.md" },
        },
        {
          type: "toolCall",
          id: "functions.read:1",
          name: "read",
          arguments: { file_path: "/Users/guoshuyi/.openclaw/workspace/USER.md" },
        },
        {
          type: "toolCall",
          id: "functions.read:2",
          name: "read",
          arguments: { file_path: "/Users/guoshuyi/.openclaw/workspace/MEMORY.md" },
        },
        {
          type: "toolCall",
          id: "functions.read:3",
          name: "read",
          arguments: { file_path: "/Users/guoshuyi/.openclaw/workspace/memory/2026-04-08.md" },
        },
        {
          type: "toolCall",
          id: "functions.read:4",
          name: "read",
          arguments: { file_path: "/Users/guoshuyi/.openclaw/workspace/memory/2026-04-07.md" },
        },
        {
          type: "text",
          text: "\n晚上好，过纯中。🏠 刚读完今天的记忆文件，我已经准备好了。有什么想做的吗？",
        },
      ],
      stopReason: "toolUse",
    });
  });

  it("extracts inline tool-call text that follows a tagged tool-call section", async () => {
    const finalMessage = {
      role: "assistant",
      content: [{ type: "text", text: KIMI_MIXED_TOOL_TEXT }],
      stopReason: "stop",
    };
    const baseStreamFn: StreamFn = () =>
      createFakeStream({
        events: [],
        resultMessage: finalMessage,
      }) as ReturnType<StreamFn>;

    const wrapped = createKimiToolCallMarkupWrapper(baseStreamFn);
    const stream = wrapped(
      { api: "anthropic-messages", provider: "kimi", id: "k2p5" } as Model<"anthropic-messages">,
      { messages: [], tools: TEST_TOOLS } as Context,
      {},
    ) as FakeStream;

    await expect(stream.result()).resolves.toEqual({
      role: "assistant",
      content: [
        {
          type: "toolCall",
          id: "functions.read:0",
          name: "read",
          arguments: { file_path: "/Users/guoshuyi/.openclaw/workspace/SOUL.md" },
        },
        {
          type: "toolCall",
          id: "functions.read:1",
          name: "read",
          arguments: { file_path: "/Users/guoshuyi/.openclaw/workspace/USER.md" },
        },
        {
          type: "toolCall",
          id: "functions.exec:5",
          name: "exec",
          arguments: { command: "cat /Users/guoshuyi/.openclaw/workspace/SOUL.md" },
        },
        {
          type: "toolCall",
          id: "functions.exec:6",
          name: "exec",
          arguments: {
            command:
              'cat /Users/guoshuyi/.openclaw/workspace/memory/2026-04-08.md 2>/dev/null || echo "File not found"',
          },
        },
        {
          type: "text",
          text: "\n晚上好！🏠 我是过家家，刚完成今天的记忆同步。有什么想聊的，或者需要我帮忙做什么？",
        },
      ],
      stopReason: "toolUse",
    });
  });

  it("adapts provider stream context without changing wrapper behavior", async () => {
    const finalMessage = {
      role: "assistant",
      content: [{ type: "text", text: KIMI_TOOL_TEXT }],
      stopReason: "stop",
    };
    const baseStreamFn: StreamFn = () =>
      createFakeStream({
        events: [],
        resultMessage: finalMessage,
      }) as ReturnType<StreamFn>;

    const wrapped = wrapKimiProviderStream({
      streamFn: baseStreamFn,
    } as never);
    const stream = wrapped(
      { api: "anthropic-messages", provider: "kimi", id: "k2p5" } as Model<"anthropic-messages">,
      { messages: [], tools: TEST_TOOLS } as Context,
      {},
    ) as FakeStream;

    await expect(stream.result()).resolves.toEqual({
      role: "assistant",
      content: [
        {
          type: "toolCall",
          id: "functions.read:0",
          name: "read",
          arguments: { file_path: "./package.json" },
        },
      ],
      stopReason: "toolUse",
    });
  });
});
