// Client/provider wire translation and response rendering.
//
// Every provider in this competition answers with an echo of the request it
// received, and the client checks that echo field by field against the
// original request. Translation therefore preserves, exactly: the system
// text, the last user message's text, the last image's bytes, the tool names
// in order, max_tokens, the stop sequences, and the JSON-mode flag. The one
// asymmetry is JSON mode: only the OpenAI and Gemini wires can carry it, so
// an OpenAI client request with response_format never routes to an
// Anthropic-wire provider (the router enforces that), and an Anthropic
// client's request never gains a JSON-mode flag in translation.
//
// Providers are always called non-streaming; the client's stream is rendered
// locally from the completed answer, which keeps one upstream code path and
// full control over client event order. Nothing here reads the clock or any
// randomness source.

import { isRecord, type Json, type Wire } from './brief.ts';

export type ClientWire = 'openai' | 'anthropic';

export interface ImagePart {
  kind: 'image';
  /** Base64 payload when the image arrived inline; null for a remote URL. */
  data: string | null;
  mediaType: string;
  /** Remote URL when the image arrived by reference; null for inline data. */
  url: string | null;
}

export interface TextPart {
  kind: 'text';
  text: string;
}

export type Part = TextPart | ImagePart;

export interface ToolCall {
  id: string;
  name: string;
  /** Arguments as a JSON string, as the OpenAI wire carries them. */
  argumentsJson: string;
}

export interface Message {
  role: 'user' | 'assistant' | 'system' | 'tool';
  parts: Part[];
  toolCalls: ToolCall[];
  toolCallId: string | null;
}

export interface ToolDef {
  name: string;
  description: string;
  parameters: Json;
}

export interface Canonical {
  systemText: string;
  messages: Message[];
  tools: ToolDef[];
  /** The client's tool choice, kept only for same-family passthrough. */
  toolChoice: unknown;
  maxTokens: number | null;
  stops: string[];
  jsonMode: boolean;
  jsonModeType: 'json_object' | 'json_schema';
  temperature: number | null;
  topP: number | null;
}

export interface ProviderAnswer {
  text: string;
  inputTokens: number;
  outputTokens: number;
  /** Stop family: 0 stop, 1 length, 2 tool calls, 3 refusal. */
  stopFamily: number;
}

const FIXED_CREATED = 1767225600;
const FIXED_CREATED_AT = '2026-01-01T00:00:00Z';
const ANTHROPIC_VERSION = '2023-06-01';

function textOf(value: unknown): string {
  if (typeof value === 'string') return value;
  if (!Array.isArray(value)) return '';
  const texts: string[] = [];
  for (const block of value) {
    if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') texts.push(block.text);
  }
  return texts.join('\n');
}

function numberOr(value: unknown, fallback: number | null): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

/** Split a data URL into its media type and base64 payload; null when not a data URL. */
function parseDataUrl(url: string): { mediaType: string; data: string } | null {
  const match = /^data:([^;,]+)?(;base64)?,(.*)$/s.exec(url);
  if (match === null || match[2] !== ';base64') return null;
  return { mediaType: match[1] === undefined || match[1] === '' ? 'image/png' : match[1], data: match[3] };
}

function openAIParts(content: unknown): Part[] {
  if (typeof content === 'string') return content === '' ? [] : [{ kind: 'text', text: content }];
  if (!Array.isArray(content)) return [];
  const parts: Part[] = [];
  for (const raw of content) {
    if (!isRecord(raw)) continue;
    if (raw.type === 'text' && typeof raw.text === 'string') {
      parts.push({ kind: 'text', text: raw.text });
    } else if (raw.type === 'image_url' && isRecord(raw.image_url) && typeof raw.image_url.url === 'string') {
      const url = raw.image_url.url;
      const parsed = parseDataUrl(url);
      if (parsed !== null) parts.push({ kind: 'image', data: parsed.data, mediaType: parsed.mediaType, url: null });
      else parts.push({ kind: 'image', data: null, mediaType: 'image/png', url });
    }
  }
  return parts;
}

function anthropicParts(content: unknown): { parts: Part[]; toolResults: { id: string; text: string }[] } {
  const parts: Part[] = [];
  const toolResults: { id: string; text: string }[] = [];
  if (typeof content === 'string') {
    if (content !== '') parts.push({ kind: 'text', text: content });
    return { parts, toolResults };
  }
  if (!Array.isArray(content)) return { parts, toolResults };
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') {
      parts.push({ kind: 'text', text: block.text });
    } else if (block.type === 'image' && isRecord(block.source)) {
      const source = block.source;
      if (source.type === 'base64' && typeof source.data === 'string') {
        parts.push({
          kind: 'image',
          data: source.data,
          mediaType: typeof source.media_type === 'string' ? source.media_type : 'image/png',
          url: null,
        });
      } else if (source.type === 'url' && typeof source.url === 'string') {
        parts.push({ kind: 'image', data: null, mediaType: 'image/png', url: source.url });
      }
    } else if (block.type === 'tool_result') {
      toolResults.push({
        id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
        text: textOf(block.content),
      });
    }
  }
  return { parts, toolResults };
}

/** An OpenAI chat-completions body in canonical form. */
export function parseOpenAIRequest(body: Json): Canonical {
  const messages: Message[] = [];
  let systemText = '';
  const systemChunks: string[] = [];
  const rawMessages = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of rawMessages) {
    if (!isRecord(raw)) continue;
    const role = raw.role;
    const parts = openAIParts(raw.content);
    if (role === 'system') {
      systemChunks.push(
        parts.filter((p): p is TextPart => p.kind === 'text').map((p) => p.text).join('\n'),
      );
      continue;
    }
    if (role === 'tool') {
      messages.push({
        role: 'tool',
        parts,
        toolCalls: [],
        toolCallId: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : null,
      });
      continue;
    }
    const toolCalls: ToolCall[] = [];
    if (Array.isArray(raw.tool_calls)) {
      for (const call of raw.tool_calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        toolCalls.push({
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          argumentsJson: typeof call.function.arguments === 'string' ? call.function.arguments : '{}',
        });
      }
    }
    messages.push({ role: role === 'assistant' ? 'assistant' : 'user', parts, toolCalls, toolCallId: null });
  }
  systemText = systemChunks.join('\n');

  const tools: ToolDef[] = [];
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw) || !isRecord(raw.function)) continue;
      const name = typeof raw.function.name === 'string' ? raw.function.name : '';
      if (name === '') continue;
      tools.push({
        name,
        description: typeof raw.function.description === 'string' ? raw.function.description : '',
        parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
      });
    }
  }

  let stops: string[] = [];
  if (typeof body.stop === 'string') stops = [body.stop];
  else if (Array.isArray(body.stop)) stops = body.stop.filter((s): s is string => typeof s === 'string');

  const format = isRecord(body.response_format) ? body.response_format : null;
  const jsonMode = format !== null && (format.type === 'json_object' || format.type === 'json_schema');

  return {
    systemText,
    messages,
    tools,
    toolChoice: body.tool_choice,
    maxTokens: numberOr(body.max_tokens, numberOr(body.max_completion_tokens, null)),
    stops,
    jsonMode,
    jsonModeType: format !== null && format.type === 'json_schema' ? 'json_schema' : 'json_object',
    temperature: numberOr(body.temperature, null),
    topP: numberOr(body.top_p, null),
  };
}

/** An Anthropic messages body in canonical form. */
export function parseAnthropicRequest(body: Json): Canonical {
  const systemText = textOf(body.system);
  const messages: Message[] = [];
  const rawMessages = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of rawMessages) {
    if (!isRecord(raw)) continue;
    const { parts, toolResults } = anthropicParts(raw.content);
    for (const result of toolResults) {
      messages.push({
        role: 'tool',
        parts: result.text === '' ? [] : [{ kind: 'text', text: result.text }],
        toolCalls: [],
        toolCallId: result.id,
      });
    }
    if (raw.role === 'assistant') {
      const toolCalls: ToolCall[] = [];
      if (Array.isArray(raw.content)) {
        for (const block of raw.content) {
          if (!isRecord(block) || block.type !== 'tool_use') continue;
          toolCalls.push({
            id: typeof block.id === 'string' ? block.id : '',
            name: typeof block.name === 'string' ? block.name : '',
            argumentsJson: JSON.stringify(isRecord(block.input) ? block.input : {}),
          });
        }
      }
      messages.push({ role: 'assistant', parts, toolCalls, toolCallId: null });
    } else if (parts.length > 0) {
      messages.push({ role: 'user', parts, toolCalls: [], toolCallId: null });
    }
  }

  const tools: ToolDef[] = [];
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw) || typeof raw.name !== 'string' || raw.name === '') continue;
      tools.push({
        name: raw.name,
        description: typeof raw.description === 'string' ? raw.description : '',
        parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
      });
    }
  }

  const stops = Array.isArray(body.stop_sequences)
    ? body.stop_sequences.filter((s): s is string => typeof s === 'string')
    : [];

  return {
    systemText,
    messages,
    tools,
    toolChoice: body.tool_choice,
    maxTokens: numberOr(body.max_tokens, null),
    stops,
    jsonMode: false,
    jsonModeType: 'json_object',
    temperature: numberOr(body.temperature, null),
    topP: numberOr(body.top_p, null),
  };
}

/** The text of a message's text parts, joined the way every wire's echo does. */
function messageText(message: Message): string {
  return message.parts.filter((p): p is TextPart => p.kind === 'text').map((p) => p.text).join('\n');
}

function imageUrlOf(part: ImagePart): string {
  if (part.data !== null) return `data:${part.mediaType};base64,${part.data}`;
  return part.url ?? '';
}

/** Canonical form as an OpenAI chat-completions body naming `model`. */
export function toOpenAIRequest(canonical: Canonical, model: string): Json {
  const messages: Json[] = [];
  if (canonical.systemText !== '') messages.push({ role: 'system', content: canonical.systemText });
  for (const message of canonical.messages) {
    if (message.role === 'tool') {
      messages.push({
        role: 'tool',
        tool_call_id: message.toolCallId ?? '',
        content: messageText(message),
      });
      continue;
    }
    const hasImage = message.parts.some((p) => p.kind === 'image');
    let content: unknown;
    if (!hasImage && message.parts.length <= 1) {
      content = messageText(message);
    } else {
      content = message.parts.map((part) =>
        part.kind === 'text'
          ? { type: 'text', text: part.text }
          : { type: 'image_url', image_url: { url: imageUrlOf(part) } },
      );
    }
    const out: Json = { role: message.role, content };
    if (message.toolCalls.length > 0) {
      out.tool_calls = message.toolCalls.map((call) => ({
        id: call.id,
        type: 'function',
        function: { name: call.name, arguments: call.argumentsJson },
      }));
      if (messageText(message) === '') out.content = null;
    }
    messages.push(out);
  }

  const out: Json = { model, messages, stream: false };
  if (canonical.maxTokens !== null) out.max_tokens = canonical.maxTokens;
  if (canonical.temperature !== null) out.temperature = canonical.temperature;
  if (canonical.topP !== null) out.top_p = canonical.topP;
  if (canonical.stops.length > 0) out.stop = [...canonical.stops];
  if (canonical.tools.length > 0) {
    out.tools = canonical.tools.map((tool) => ({
      type: 'function',
      function: { name: tool.name, description: tool.description, parameters: tool.parameters },
    }));
  }
  const choice = canonical.toolChoice;
  if (typeof choice === 'string') out.tool_choice = choice;
  else if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') {
      out.tool_choice = { type: 'function', function: { name: choice.name } };
    }
  }
  if (canonical.jsonMode) out.response_format = { type: canonical.jsonModeType };
  return out;
}

/** Canonical form as an Anthropic messages body naming `model`. Never carries JSON mode. */
export function toAnthropicRequest(canonical: Canonical, model: string): Json {
  const messages: Json[] = [];
  for (const message of canonical.messages) {
    if (message.role === 'tool') {
      messages.push({
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: message.toolCallId ?? '', content: messageText(message) }],
      });
      continue;
    }
    const content: Json[] = [];
    for (const part of message.parts) {
      if (part.kind === 'text') content.push({ type: 'text', text: part.text });
      else if (part.data !== null) {
        content.push({ type: 'image', source: { type: 'base64', media_type: part.mediaType, data: part.data } });
      } else {
        content.push({ type: 'image', source: { type: 'url', url: part.url ?? '' } });
      }
    }
    for (const call of message.toolCalls) {
      let input: Json = {};
      try {
        const parsed: unknown = JSON.parse(call.argumentsJson);
        if (isRecord(parsed)) input = parsed;
      } catch {
        input = {};
      }
      content.push({ type: 'tool_use', id: call.id, name: call.name, input });
    }
    if (content.length === 1 && content[0].type === 'text' && message.toolCalls.length === 0) {
      messages.push({ role: message.role, content: content[0].text });
    } else if (content.length > 0) {
      messages.push({ role: message.role, content });
    }
  }

  const out: Json = { model, messages, stream: false };
  if (canonical.systemText !== '') out.system = canonical.systemText;
  if (canonical.maxTokens !== null) out.max_tokens = canonical.maxTokens;
  if (canonical.temperature !== null) out.temperature = canonical.temperature;
  if (canonical.topP !== null) out.top_p = canonical.topP;
  if (canonical.stops.length > 0) out.stop_sequences = [...canonical.stops];
  if (canonical.tools.length > 0) {
    out.tools = canonical.tools.map((tool) => ({
      name: tool.name,
      description: tool.description,
      input_schema: tool.parameters,
    }));
  }
  const choice = canonical.toolChoice;
  if (isRecord(choice) && typeof choice.type === 'string') {
    if (choice.type === 'auto') out.tool_choice = { type: 'auto' };
    else if (choice.type === 'any') out.tool_choice = { type: 'any' };
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'tool', name: choice.name };
  } else if (choice === 'required') {
    out.tool_choice = { type: 'any' };
  } else if (choice === 'auto' || choice === 'none') {
    out.tool_choice = { type: choice };
  }
  return out;
}

/** Canonical form as a Gemini generateContent body naming `model` (the model goes in the URL). */
export function toGeminiRequest(canonical: Canonical): Json {
  const contents: Json[] = [];
  for (const message of canonical.messages) {
    if (message.role === 'tool') {
      contents.push({
        role: 'function',
        parts: [{ functionResponse: { name: message.toolCallId ?? '', response: { result: messageText(message) } } }],
      });
      continue;
    }
    const parts: Json[] = [];
    for (const part of message.parts) {
      if (part.kind === 'text') parts.push({ text: part.text });
      else if (part.data !== null) parts.push({ inlineData: { mimeType: part.mediaType, data: part.data } });
      else parts.push({ fileData: { fileUri: part.url ?? '' } });
    }
    for (const call of message.toolCalls) {
      let args: Json = {};
      try {
        const parsed: unknown = JSON.parse(call.argumentsJson);
        if (isRecord(parsed)) args = parsed;
      } catch {
        args = {};
      }
      parts.push({ functionCall: { name: call.name, args } });
    }
    if (parts.length > 0) contents.push({ role: message.role === 'assistant' ? 'model' : 'user', parts });
  }

  const out: Json = { contents };
  if (canonical.systemText !== '') out.systemInstruction = { parts: [{ text: canonical.systemText }] };
  if (canonical.tools.length > 0) {
    out.tools = [
      {
        functionDeclarations: canonical.tools.map((tool) => ({
          name: tool.name,
          description: tool.description,
          parameters: tool.parameters,
        })),
      },
    ];
  }
  const config: Json = {};
  if (canonical.maxTokens !== null) config.maxOutputTokens = canonical.maxTokens;
  if (canonical.stops.length > 0) config.stopSequences = [...canonical.stops];
  if (canonical.temperature !== null) config.temperature = canonical.temperature;
  if (canonical.topP !== null) config.topP = canonical.topP;
  if (canonical.jsonMode) config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/** Build the upstream request body for a provider wire, from the canonical form. */
export function buildUpstreamBody(canonical: Canonical, wire: Wire, model: string): Json {
  if (wire === 'openai') return toOpenAIRequest(canonical, model);
  if (wire === 'anthropic') return toAnthropicRequest(canonical, model);
  return toGeminiRequest(canonical);
}

/** The upstream path for a provider wire (Gemini names the model in the path). */
export function upstreamPath(wire: Wire, model: string): string {
  if (wire === 'openai') return '/v1/chat/completions';
  if (wire === 'anthropic') return '/v1/messages';
  return `/v1beta/models/${encodeURIComponent(model)}:generateContent`;
}

/** Extra headers a provider wire needs beyond auth. */
export function upstreamHeaders(wire: Wire): Record<string, string> {
  const headers: Record<string, string> = { 'content-type': 'application/json' };
  if (wire === 'anthropic') headers['anthropic-version'] = ANTHROPIC_VERSION;
  return headers;
}

function stopFamilyOf(reason: unknown): number {
  if (reason === 'length' || reason === 'max_tokens' || reason === 'MAX_TOKENS') return 1;
  if (reason === 'tool_calls' || reason === 'function_call' || reason === 'tool_use') return 2;
  if (reason === 'content_filter' || reason === 'refusal' || reason === 'SAFETY') return 3;
  return 0;
}

function blocksText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .filter((b): b is Json => isRecord(b) && b.type === 'text' && typeof b.text === 'string')
    .map((b) => b.text as string)
    .join('');
}

/** Extract the answer from an OpenAI-wire provider reply; null when unusable. */
function answerFromOpenAI(body: unknown): ProviderAnswer | null {
  if (!isRecord(body) || !Array.isArray(body.choices)) return null;
  const choice = body.choices[0];
  if (!isRecord(choice) || !isRecord(choice.message)) return null;
  const text = blocksText(choice.message.content);
  if (typeof choice.message.content !== 'string' && !Array.isArray(choice.message.content)) return null;
  const usage = isRecord(body.usage) ? body.usage : {};
  const input = numberOr(usage.prompt_tokens, null);
  const output = numberOr(usage.completion_tokens, null);
  if (input === null || output === null) return null;
  return { text, inputTokens: input, outputTokens: output, stopFamily: stopFamilyOf(choice.finish_reason) };
}

/** Extract the answer from an Anthropic-wire provider reply; null when unusable. */
function answerFromAnthropic(body: unknown): ProviderAnswer | null {
  if (!isRecord(body) || !Array.isArray(body.content)) return null;
  const text = blocksText(body.content);
  const usage = isRecord(body.usage) ? body.usage : {};
  const input = numberOr(usage.input_tokens, null);
  const output = numberOr(usage.output_tokens, null);
  if (input === null || output === null) return null;
  return { text, inputTokens: input, outputTokens: output, stopFamily: stopFamilyOf(body.stop_reason) };
}

/** Extract the answer from a Gemini-wire provider reply; null when unusable. */
function answerFromGemini(body: unknown): ProviderAnswer | null {
  if (!isRecord(body) || !Array.isArray(body.candidates)) return null;
  const candidate = body.candidates[0];
  if (!isRecord(candidate) || !isRecord(candidate.content) || !Array.isArray(candidate.content.parts)) return null;
  const text = candidate.content.parts
    .filter((p): p is Json => isRecord(p) && typeof p.text === 'string')
    .map((p) => p.text as string)
    .join('');
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  const input = numberOr(usage.promptTokenCount, null);
  const output = numberOr(usage.candidatesTokenCount, null);
  if (input === null || output === null) return null;
  return { text, inputTokens: input, outputTokens: output, stopFamily: stopFamilyOf(candidate.finishReason) };
}

/** Extract a provider answer from its wire's reply shape; null when unusable. */
export function extractAnswer(wire: Wire, body: unknown): ProviderAnswer | null {
  if (wire === 'openai') return answerFromOpenAI(body);
  if (wire === 'anthropic') return answerFromAnthropic(body);
  return answerFromGemini(body);
}

function openAIStopReason(family: number): string {
  if (family === 1) return 'length';
  if (family === 2) return 'tool_calls';
  if (family === 3) return 'content_filter';
  return 'stop';
}

function anthropicStopReason(family: number): string {
  if (family === 1) return 'max_tokens';
  if (family === 2) return 'tool_use';
  if (family === 3) return 'refusal';
  return 'end_turn';
}

/** Split answer text into streaming pieces; concatenation reproduces the text exactly. */
function splitPieces(text: string): string[] {
  if (text === '') return [];
  const pieces: string[] = [];
  const size = 64;
  for (let i = 0; i < text.length; i += size) pieces.push(text.slice(i, i + size));
  return pieces;
}

function sseFrame(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** Render an answer as an OpenAI chat-completion object. */
export function renderOpenAIJson(answer: ProviderAnswer, model: string): Json {
  return {
    id: 'chatcmpl-aggregator',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: answer.text },
        finish_reason: openAIStopReason(answer.stopFamily),
        logprobs: null,
      },
    ],
    usage: {
      prompt_tokens: answer.inputTokens,
      completion_tokens: answer.outputTokens,
      total_tokens: answer.inputTokens + answer.outputTokens,
    },
  };
}

/** Render an answer as an OpenAI SSE stream: role, content, finish, usage, [DONE]. */
export function renderOpenAIStream(answer: ProviderAnswer, model: string): string {
  const frames: string[] = [];
  const base = { id: 'chatcmpl-aggregator', object: 'chat.completion.chunk', created: FIXED_CREATED, model };
  frames.push(
    sseFrame(null, { ...base, choices: [{ index: 0, delta: { role: 'assistant' }, finish_reason: null }] }),
  );
  for (const piece of splitPieces(answer.text)) {
    frames.push(
      sseFrame(null, { ...base, choices: [{ index: 0, delta: { content: piece }, finish_reason: null }] }),
    );
  }
  frames.push(
    sseFrame(null, {
      ...base,
      choices: [{ index: 0, delta: {}, finish_reason: openAIStopReason(answer.stopFamily) }],
    }),
  );
  frames.push(
    sseFrame(null, {
      ...base,
      choices: [],
      usage: {
        prompt_tokens: answer.inputTokens,
        completion_tokens: answer.outputTokens,
        total_tokens: answer.inputTokens + answer.outputTokens,
      },
    }),
  );
  return `${frames.join('')}data: [DONE]\n\n`;
}

/** Render an answer as an Anthropic messages object. */
export function renderAnthropicJson(answer: ProviderAnswer, model: string): Json {
  return {
    id: 'msg_aggregator',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: answer.text }],
    stop_reason: anthropicStopReason(answer.stopFamily),
    stop_sequence: null,
    usage: { input_tokens: answer.inputTokens, output_tokens: answer.outputTokens },
  };
}

/**
 * Render an answer as an Anthropic SSE stream: message_start, one text
 * block's start/deltas/stop, message_delta with the stop reason and output
 * tokens, then message_stop.
 */
export function renderAnthropicStream(answer: ProviderAnswer, model: string): string {
  const frames: string[] = [];
  frames.push(
    sseFrame('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_aggregator',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.inputTokens, output_tokens: 0 },
      },
    }),
  );
  frames.push(
    sseFrame('content_block_start', {
      type: 'content_block_start',
      index: 0,
      content_block: { type: 'text', text: '' },
    }),
  );
  for (const piece of splitPieces(answer.text)) {
    frames.push(
      sseFrame('content_block_delta', {
        type: 'content_block_delta',
        index: 0,
        delta: { type: 'text_delta', text: piece },
      }),
    );
  }
  frames.push(sseFrame('content_block_stop', { type: 'content_block_stop', index: 0 }));
  frames.push(
    sseFrame('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: anthropicStopReason(answer.stopFamily), stop_sequence: null },
      usage: { output_tokens: answer.outputTokens },
    }),
  );
  frames.push(sseFrame('message_stop', { type: 'message_stop' }));
  return frames.join('');
}

/** An error body in the client's wire shape. */
export function errorBody(wire: ClientWire, status: number, message: string): Json {
  if (wire === 'openai') {
    const type = status >= 500 ? 'server_error' : 'invalid_request_error';
    return { error: { message, type, code: null } };
  }
  const type = status === 400 ? 'invalid_request_error' : status === 429 ? 'rate_limit_error' : 'api_error';
  return { type: 'error', error: { type, message } };
}

export interface ServedModel {
  id: string;
  provider: string;
}

/** GET /v1/models in the OpenAI list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models in the Anthropic list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.id, created_at: FIXED_CREATED_AT })),
    has_more: false,
  };
}
