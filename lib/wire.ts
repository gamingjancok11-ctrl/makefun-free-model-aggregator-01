// The three wires: OpenAI Chat Completions, Anthropic Messages and Gemini generateContent.
//
// A client request arrives on one of the two client wires and leaves on whichever wire its
// routed provider speaks; the provider's answer comes back on that wire and is rendered in
// the client's. Everything passes through one normalised form so no pair of wires is special:
//
//   normalise (openai | anthropic) -> render (openai | anthropic | gemini)
//   parse    (openai | anthropic | gemini) -> render (openai | anthropic)
//
// The normalised request keeps every field the provider echoes back to the client: the
// system prompt, the user text, images, tool definitions, max_tokens, stop sequences and
// JSON mode. A field dropped or rewritten here loses the request, because the client
// compares the provider's echo with what it sent.
//
// Two wire facts shape the translation:
//   - an Anthropic-wire provider cannot carry JSON mode (its wire has no response_format),
//     so a JSON-mode request from an OpenAI client must never be routed to one; the router
//     enforces that. A Gemini-wire provider carries it as responseMimeType application/json.
//   - an OpenAI streaming answer carries its usage only when the request asked for it
//     (stream_options.include_usage), so every upstream OpenAI streaming call sets it: the
//     client checks the usage of every streamed answer.
//
// All rendering is a pure function of its arguments: fixed ids and timestamps, no clock,
// no randomness, so a replay emits the same bytes.

import { isRecord, type Json } from './brief.ts';

/** A fixed epoch second for every `created` this server writes. */
const FIXED_CREATED = 1767225600;
/** A fixed instant for every Anthropic `created_at` this server writes. */
const FIXED_CREATED_AT = '2026-01-01T00:00:00Z';
/** How many characters one streaming text delta carries. */
const STREAM_CHUNK_CHARS = 13;

export type StopReason = 'stop' | 'length' | 'tool_calls' | 'content_filter';

export interface ImagePart {
  mime: string;
  data: string;
  url: string | null;
}

export interface ToolCall {
  id: string;
  name: string;
  args: string;
}

export interface ToolDef {
  name: string;
  description: string;
  parameters: Json;
}

export interface Turn {
  role: 'user' | 'assistant' | 'tool';
  text: string;
  images: ImagePart[];
  toolCalls: ToolCall[];
  toolResult: { id: string; text: string } | null;
}

/** One client request, normalised: every field a provider echoes, in no wire's spelling. */
export interface NormRequest {
  system: string;
  turns: Turn[];
  tools: ToolDef[];
  toolChoice: 'auto' | 'required' | { name: string } | null;
  maxTokens: number | null;
  stops: string[];
  jsonMode: boolean;
  stream: boolean;
  temperature: number | null;
  topP: number | null;
}

/** One provider answer, normalised: the text whole, its usage, its stop reason. */
export interface NormAnswer {
  text: string;
  inputTokens: number;
  outputTokens: number;
  stop: StopReason;
  toolCalls: ToolCall[];
}

function numberOrNull(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

function stringOf(value: unknown): string {
  return typeof value === 'string' ? value : '';
}

/** The text of a string, or of a block array's text blocks joined with newlines. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** A tool call's arguments as a JSON string: the provider's string, or the object re-encoded. */
function argsText(value: unknown): string {
  if (typeof value === 'string') return value;
  if (isRecord(value)) return JSON.stringify(value);
  return '{}';
}

/** A tool call's arguments as an object: the JSON string parsed, or {} when it is not one. */
function argsObject(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

function stringList(value: unknown): string[] {
  if (typeof value === 'string') return [value];
  if (!Array.isArray(value)) return [];
  return value.filter((x): x is string => typeof x === 'string');
}

/** An OpenAI image_url part as an image: a data URL decoded, an http(s) URL kept as a URL. */
function imageFromOpenAI(part: Json): ImagePart | null {
  if (!isRecord(part.image_url)) return null;
  const url = stringOf(part.image_url.url);
  if (url === '') return null;
  const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
  if (data !== null) return { mime: data[1] !== '' ? data[1] : 'image/png', data: data[2], url: null };
  return { mime: 'image/png', data: '', url };
}

/** An Anthropic image block as an image: base64 data, or a URL source. */
function imageFromAnthropic(block: Json): ImagePart | null {
  if (!isRecord(block.source)) return null;
  const source = block.source;
  if (source.type === 'base64' && typeof source.data === 'string') {
    return { mime: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data, url: null };
  }
  if (source.type === 'url' && typeof source.url === 'string') return { mime: 'image/png', data: '', url: source.url };
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds more. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** An OpenAI Chat Completions request, normalised. */
export function normalizeOpenAI(body: Json): NormRequest {
  const turns: Turn[] = [];
  const systemParts: string[] = [];
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of messages) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'assistant') {
      const toolCalls: ToolCall[] = [];
      if (Array.isArray(raw.tool_calls)) {
        for (const call of raw.tool_calls) {
          if (!isRecord(call) || !isRecord(call.function)) continue;
          toolCalls.push({
            id: stringOf(call.id),
            name: stringOf(call.function.name),
            args: argsText(call.function.arguments),
          });
        }
      }
      turns.push({ role: 'assistant', text: textOf(raw.content), images: [], toolCalls, toolResult: null });
      continue;
    }
    if (raw.role === 'tool') {
      turns.push({
        role: 'tool',
        text: '',
        images: [],
        toolCalls: [],
        toolResult: { id: stringOf(raw.tool_call_id), text: typeof raw.content === 'string' ? raw.content : textOf(raw.content) },
      });
      continue;
    }
    // Any other role is a user turn.
    const images: ImagePart[] = [];
    if (Array.isArray(raw.content)) {
      for (const block of raw.content) {
        if (isRecord(block) && block.type === 'image_url') {
          const image = imageFromOpenAI(block);
          if (image !== null) images.push(image);
        }
      }
    }
    turns.push({ role: 'user', text: textOf(raw.content), images, toolCalls: [], toolResult: null });
  }
  const tools: ToolDef[] = [];
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') continue;
      tools.push({
        name: raw.function.name,
        description: stringOf(raw.function.description),
        parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
      });
    }
  }
  let toolChoice: NormRequest['toolChoice'] = null;
  if (body.tool_choice === 'auto') toolChoice = 'auto';
  else if (body.tool_choice === 'required') toolChoice = 'required';
  else if (isRecord(body.tool_choice) && isRecord(body.tool_choice.function) && typeof body.tool_choice.function.name === 'string') {
    toolChoice = { name: body.tool_choice.function.name };
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    system: systemParts.join('\n'),
    turns,
    tools,
    toolChoice,
    maxTokens: numberOrNull(body.max_tokens) ?? numberOrNull(body.max_completion_tokens),
    stops: stringList(body.stop),
    jsonMode: format === 'json_object' || format === 'json_schema',
    stream: body.stream === true,
    temperature: numberOrNull(body.temperature),
    topP: numberOrNull(body.top_p),
  };
}

/** An Anthropic Messages request, normalised. */
export function normalizeAnthropic(body: Json): NormRequest {
  const turns: Turn[] = [];
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of messages) {
    if (!isRecord(raw)) continue;
    const content = raw.content;
    if (typeof content === 'string') {
      turns.push({ role: raw.role === 'assistant' ? 'assistant' : 'user', text: content, images: [], toolCalls: [], toolResult: null });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (raw.role === 'assistant') {
      const toolCalls: ToolCall[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({ id: stringOf(block.id), name: stringOf(block.name), args: argsText(block.input) });
      }
      turns.push({ role: 'assistant', text: textOf(content), images: [], toolCalls, toolResult: null });
      continue;
    }
    // A user turn: text and images in one turn, each tool_result in its own tool turn.
    const images: ImagePart[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'image') {
        const image = imageFromAnthropic(block);
        if (image !== null) images.push(image);
      }
    }
    const text = textOf(content);
    if (text !== '' || images.length > 0) turns.push({ role: 'user', text, images, toolCalls: [], toolResult: null });
    for (const block of content) {
      if (!isRecord(block) || block.type !== 'tool_result') continue;
      turns.push({
        role: 'tool',
        text: '',
        images: [],
        toolCalls: [],
        toolResult: { id: stringOf(block.tool_use_id), text: toolResultText(block.content) },
      });
    }
  }
  const tools: ToolDef[] = [];
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw) || typeof raw.name !== 'string') continue;
      tools.push({
        name: raw.name,
        description: stringOf(raw.description),
        parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
      });
    }
  }
  let toolChoice: NormRequest['toolChoice'] = null;
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') toolChoice = 'auto';
    else if (choice.type === 'any') toolChoice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') toolChoice = { name: choice.name };
  }
  return {
    system: textOf(body.system),
    turns,
    tools,
    toolChoice,
    maxTokens: numberOrNull(body.max_tokens),
    stops: stringList(body.stop_sequences),
    // The Anthropic wire carries no JSON-mode flag; a JSON requirement is a routing
    // constraint (the model must take JSON), never a field on this wire.
    jsonMode: false,
    stream: body.stream === true,
    temperature: numberOrNull(body.temperature),
    topP: numberOrNull(body.top_p),
  };
}

/** One image as an OpenAI content part: a data URL when it has bytes, the URL otherwise. */
function openAIImagePart(image: ImagePart): Json {
  const url = image.data !== '' ? `data:${image.mime};base64,${image.data}` : (image.url ?? '');
  return { type: 'image_url', image_url: { url } };
}

/** One image as an Anthropic content block. */
function anthropicImageBlock(image: ImagePart): Json {
  if (image.data !== '') return { type: 'image', source: { type: 'base64', media_type: image.mime, data: image.data } };
  return { type: 'image', source: { type: 'url', url: image.url ?? '' } };
}

/** One image as a Gemini content part: inline bytes, or a file URI. */
function geminiImagePart(image: ImagePart): Json {
  if (image.data !== '') return { inlineData: { mimeType: image.mime, data: image.data } };
  return { fileData: { fileUri: image.url ?? '' } };
}

/** A normalised request as an OpenAI Chat Completions body naming `model`. */
export function renderOpenAI(req: NormRequest, model: string): Json {
  const messages: Json[] = [];
  if (req.system !== '') messages.push({ role: 'system', content: req.system });
  for (const turn of req.turns) {
    if (turn.role === 'assistant') {
      const message: Json = { role: 'assistant', content: turn.text !== '' ? turn.text : null };
      if (turn.toolCalls.length > 0) {
        message.tool_calls = turn.toolCalls.map((c) => ({
          id: c.id,
          type: 'function',
          function: { name: c.name, arguments: c.args },
        }));
      }
      messages.push(message);
      continue;
    }
    if (turn.role === 'tool') {
      messages.push({ role: 'tool', tool_call_id: turn.toolResult?.id ?? '', content: turn.toolResult?.text ?? '' });
      continue;
    }
    if (turn.images.length === 0) {
      messages.push({ role: 'user', content: turn.text });
      continue;
    }
    const parts: Json[] = [{ type: 'text', text: turn.text }];
    for (const image of turn.images) parts.push(openAIImagePart(image));
    messages.push({ role: 'user', content: parts });
  }
  const out: Json = { model, messages };
  if (req.maxTokens !== null) out.max_tokens = req.maxTokens;
  if (req.temperature !== null) out.temperature = req.temperature;
  if (req.topP !== null) out.top_p = req.topP;
  if (req.stops.length > 0) out.stop = req.stops;
  if (req.tools.length > 0) {
    out.tools = req.tools.map((t) => ({
      type: 'function',
      function: { name: t.name, description: t.description, parameters: t.parameters },
    }));
  }
  if (req.toolChoice === 'auto') out.tool_choice = 'auto';
  else if (req.toolChoice === 'required') out.tool_choice = 'required';
  else if (req.toolChoice !== null) out.tool_choice = { type: 'function', function: { name: req.toolChoice.name } };
  if (req.jsonMode) out.response_format = { type: 'json_object' };
  if (req.stream) {
    out.stream = true;
    // Without include_usage the stream carries no usage chunk, and the client checks
    // the usage of every streamed answer.
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** A normalised request as an Anthropic Messages body naming `model`. */
export function renderAnthropic(req: NormRequest, model: string): Json {
  const messages: Json[] = [];
  for (const turn of req.turns) {
    if (turn.role === 'assistant') {
      const blocks: Json[] = [];
      if (turn.text !== '') blocks.push({ type: 'text', text: turn.text });
      for (const call of turn.toolCalls) {
        blocks.push({ type: 'tool_use', id: call.id, name: call.name, input: argsObject(call.args) });
      }
      messages.push({ role: 'assistant', content: blocks.length > 0 ? blocks : turn.text });
      continue;
    }
    if (turn.role === 'tool') {
      messages.push({
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: turn.toolResult?.id ?? '', content: turn.toolResult?.text ?? '' }],
      });
      continue;
    }
    if (turn.images.length === 0) {
      messages.push({ role: 'user', content: turn.text });
      continue;
    }
    const blocks: Json[] = [{ type: 'text', text: turn.text }];
    for (const image of turn.images) blocks.push(anthropicImageBlock(image));
    messages.push({ role: 'user', content: blocks });
  }
  const out: Json = { model, max_tokens: req.maxTokens ?? 1024, messages };
  if (req.system !== '') out.system = req.system;
  if (req.temperature !== null) out.temperature = req.temperature;
  if (req.topP !== null) out.top_p = req.topP;
  if (req.stops.length > 0) out.stop_sequences = req.stops;
  if (req.tools.length > 0) {
    out.tools = req.tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.parameters }));
  }
  if (req.toolChoice === 'auto') out.tool_choice = { type: 'auto' };
  else if (req.toolChoice === 'required') out.tool_choice = { type: 'any' };
  else if (req.toolChoice !== null) out.tool_choice = { type: 'tool', name: req.toolChoice.name };
  if (req.stream) out.stream = true;
  return out;
}

/** A normalised request as a Gemini generateContent body. */
export function renderGemini(req: NormRequest): Json {
  const contents: Json[] = [];
  for (const turn of req.turns) {
    if (turn.role === 'assistant') {
      const parts: Json[] = [];
      if (turn.text !== '') parts.push({ text: turn.text });
      for (const call of turn.toolCalls) {
        parts.push({ functionCall: { name: call.name, args: argsObject(call.args) } });
      }
      contents.push({ role: 'model', parts });
      continue;
    }
    if (turn.role === 'tool') {
      contents.push({
        role: 'user',
        parts: [{ functionResponse: { name: turn.toolResult?.id ?? '', response: { result: turn.toolResult?.text ?? '' } } }],
      });
      continue;
    }
    const parts: Json[] = [];
    if (turn.text !== '') parts.push({ text: turn.text });
    for (const image of turn.images) parts.push(geminiImagePart(image));
    contents.push({ role: 'user', parts });
  }
  const out: Json = { contents };
  if (req.system !== '') out.systemInstruction = { parts: [{ text: req.system }] };
  if (req.tools.length > 0) {
    out.tools = [
      {
        functionDeclarations: req.tools.map((t) => ({ name: t.name, description: t.description, parameters: t.parameters })),
      },
    ];
  }
  if (req.toolChoice === 'required') out.toolConfig = { functionCallingConfig: { mode: 'ANY' } };
  else if (req.toolChoice !== null && req.toolChoice !== 'auto') {
    out.toolConfig = { functionCallingConfig: { mode: 'ANY', allowedFunctionNames: [req.toolChoice.name] } };
  }
  const config: Json = {};
  if (req.maxTokens !== null) config.maxOutputTokens = req.maxTokens;
  if (req.stops.length > 0) config.stopSequences = req.stops;
  if (req.jsonMode) config.responseMimeType = 'application/json';
  if (req.temperature !== null) config.temperature = req.temperature;
  if (req.topP !== null) config.topP = req.topP;
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/** An OpenAI finish_reason as a normalised stop reason. */
function stopFromOpenAI(finish: unknown): StopReason {
  if (finish === 'length') return 'length';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_calls';
  if (finish === 'content_filter') return 'content_filter';
  return 'stop';
}

/** An Anthropic stop_reason as a normalised stop reason. */
function stopFromAnthropic(reason: unknown): StopReason {
  if (reason === 'max_tokens') return 'length';
  if (reason === 'tool_use') return 'tool_calls';
  if (reason === 'refusal') return 'content_filter';
  return 'stop';
}

/** A Gemini finishReason as a normalised stop reason. */
function stopFromGemini(reason: unknown): StopReason {
  if (reason === 'MAX_TOKENS') return 'length';
  if (reason === 'SAFETY') return 'content_filter';
  return 'stop';
}

function stopToOpenAI(stop: StopReason): string {
  if (stop === 'length') return 'length';
  if (stop === 'tool_calls') return 'tool_calls';
  if (stop === 'content_filter') return 'content_filter';
  return 'stop';
}

function stopToAnthropic(stop: StopReason): string {
  if (stop === 'length') return 'max_tokens';
  if (stop === 'tool_calls') return 'tool_use';
  if (stop === 'content_filter') return 'refusal';
  return 'end_turn';
}

/** An OpenAI Chat Completions JSON answer, normalised. */
export function parseOpenAIAnswer(body: Json): NormAnswer {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  const toolCalls: ToolCall[] = [];
  if (Array.isArray(message.tool_calls)) {
    for (const call of message.tool_calls) {
      if (!isRecord(call) || !isRecord(call.function)) continue;
      toolCalls.push({ id: stringOf(call.id), name: stringOf(call.function.name), args: argsText(call.function.arguments) });
    }
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text,
    inputTokens: numberOrZero(usage.prompt_tokens),
    outputTokens: numberOrZero(usage.completion_tokens),
    stop: stopFromOpenAI(choice.finish_reason),
    toolCalls,
  };
}

/** An Anthropic Messages JSON answer, normalised. */
export function parseAnthropicAnswer(body: Json): NormAnswer {
  const blocks = Array.isArray(body.content) ? body.content : [];
  const parts: string[] = [];
  const toolCalls: ToolCall[] = [];
  for (const block of blocks) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
    else if (block.type === 'tool_use') {
      toolCalls.push({ id: stringOf(block.id), name: stringOf(block.name), args: argsText(block.input) });
    }
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text: parts.join(''),
    inputTokens: numberOrZero(usage.input_tokens),
    outputTokens: numberOrZero(usage.output_tokens),
    stop: stopFromAnthropic(body.stop_reason),
    toolCalls,
  };
}

/** A Gemini generateContent JSON answer, normalised. */
export function parseGeminiAnswer(body: Json): NormAnswer {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  const texts: string[] = [];
  const toolCalls: ToolCall[] = [];
  for (const part of parts) {
    if (!isRecord(part)) continue;
    if (typeof part.text === 'string') texts.push(part.text);
    else if (isRecord(part.functionCall)) {
      toolCalls.push({ id: stringOf(part.functionCall.name), name: stringOf(part.functionCall.name), args: argsText(part.functionCall.args) });
    }
  }
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    text: texts.join(''),
    inputTokens: numberOrZero(usage.promptTokenCount),
    outputTokens: numberOrZero(usage.candidatesTokenCount),
    stop: stopFromGemini(candidate.finishReason),
    toolCalls,
  };
}

/** One SSE frame: its event name (null for an unnamed frame) and its joined data lines. */
export interface SseFrame {
  event: string | null;
  data: string;
}

/** Split a server-sent-events body into frames. */
export function parseSseFrames(text: string): SseFrame[] {
  const frames: SseFrame[] = [];
  const normalized = text.replace(/\r\n/g, '\n');
  for (const frame of normalized.split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

function jsonOrNull(text: string): Json | null {
  try {
    const value: unknown = JSON.parse(text);
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

/** An OpenAI SSE stream, normalised: deltas joined, the finish chunk's reason, the usage chunk. */
export function parseOpenAIStream(text: string): NormAnswer {
  let out = '';
  let inputTokens = 0;
  let outputTokens = 0;
  let stop: StopReason = 'stop';
  const toolArgs = new Map<number, { id: string; name: string; args: string }>();
  for (const frame of parseSseFrames(text)) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') break;
    const chunk = jsonOrNull(payload);
    if (chunk === null) continue;
    if (isRecord(chunk.usage)) {
      inputTokens = numberOrZero(chunk.usage.prompt_tokens);
      outputTokens = numberOrZero(chunk.usage.completion_tokens);
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') out += delta.content;
    if (Array.isArray(delta.tool_calls)) {
      for (const call of delta.tool_calls) {
        if (!isRecord(call)) continue;
        const index = typeof call.index === 'number' ? call.index : toolArgs.size;
        const fn = isRecord(call.function) ? call.function : {};
        const slot = toolArgs.get(index) ?? { id: '', name: '', args: '' };
        if (typeof call.id === 'string' && call.id !== '') slot.id = call.id;
        if (typeof fn.name === 'string' && fn.name !== '') slot.name = fn.name;
        if (typeof fn.arguments === 'string') slot.args += fn.arguments;
        toolArgs.set(index, slot);
      }
    }
    if (typeof choice.finish_reason === 'string') stop = stopFromOpenAI(choice.finish_reason);
  }
  const toolCalls = [...toolArgs.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([, slot]) => ({ id: slot.id, name: slot.name, args: slot.args }));
  return { text: out, inputTokens, outputTokens, stop, toolCalls };
}

/** An Anthropic SSE stream, normalised: text deltas joined, usage from start and delta events. */
export function parseAnthropicStream(text: string): NormAnswer {
  let out = '';
  let inputTokens = 0;
  let outputTokens = 0;
  let stop: StopReason = 'stop';
  const toolArgs = new Map<number, { id: string; name: string; args: string }>();
  for (const frame of parseSseFrames(text)) {
    const data = jsonOrNull(frame.data);
    if (data === null) continue;
    if (frame.event === 'message_start') {
      const message = isRecord(data.message) ? data.message : {};
      const usage = isRecord(message.usage) ? message.usage : {};
      inputTokens = numberOrZero(usage.input_tokens);
    } else if (frame.event === 'content_block_start') {
      const block = isRecord(data.content_block) ? data.content_block : {};
      const index = numberOrNull(data.index);
      if (block.type === 'tool_use' && index !== null) {
        toolArgs.set(index, { id: stringOf(block.id), name: stringOf(block.name), args: '' });
      }
    } else if (frame.event === 'content_block_delta') {
      const delta = isRecord(data.delta) ? data.delta : {};
      const index = numberOrNull(data.index);
      if (delta.type === 'text_delta' && typeof delta.text === 'string') out += delta.text;
      else if (delta.type === 'input_json_delta' && typeof delta.partial_json === 'string' && index !== null) {
        const slot = toolArgs.get(index) ?? { id: '', name: '', args: '' };
        slot.args += delta.partial_json;
        toolArgs.set(index, slot);
      }
    } else if (frame.event === 'message_delta') {
      const delta = isRecord(data.delta) ? data.delta : {};
      if (typeof delta.stop_reason === 'string') stop = stopFromAnthropic(delta.stop_reason);
      if (isRecord(data.usage)) outputTokens = numberOrZero(data.usage.output_tokens);
    }
  }
  const toolCalls = [...toolArgs.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([, slot]) => ({ id: slot.id, name: slot.name, args: slot.args }));
  return { text: out, inputTokens, outputTokens, stop, toolCalls };
}

/** A Gemini SSE stream, normalised: candidate text joined, usage from the final frame. */
export function parseGeminiStream(text: string): NormAnswer {
  let out = '';
  let inputTokens = 0;
  let outputTokens = 0;
  let stop: StopReason = 'stop';
  const toolCalls: ToolCall[] = [];
  for (const frame of parseSseFrames(text)) {
    const chunk = jsonOrNull(frame.data);
    if (chunk === null) continue;
    const candidates = Array.isArray(chunk.candidates) ? chunk.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : {};
    const content = isRecord(candidate.content) ? candidate.content : {};
    const parts = Array.isArray(content.parts) ? content.parts : [];
    for (const part of parts) {
      if (!isRecord(part)) continue;
      if (typeof part.text === 'string') out += part.text;
      else if (isRecord(part.functionCall)) {
        toolCalls.push({ id: stringOf(part.functionCall.name), name: stringOf(part.functionCall.name), args: argsText(part.functionCall.args) });
      }
    }
    if (typeof candidate.finishReason === 'string') stop = stopFromGemini(candidate.finishReason);
    if (isRecord(chunk.usageMetadata)) {
      inputTokens = numberOrZero(chunk.usageMetadata.promptTokenCount);
      outputTokens = numberOrZero(chunk.usageMetadata.candidatesTokenCount);
    }
  }
  return { text: out, inputTokens, outputTokens, stop, toolCalls };
}

/** Split text into streaming deltas of at most STREAM_CHUNK_CHARS characters. */
function splitText(text: string): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += STREAM_CHUNK_CHARS) pieces.push(text.slice(at, at + STREAM_CHUNK_CHARS));
  return pieces.length === 0 ? [''] : pieces;
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then a blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** A normalised answer as an OpenAI Chat Completions JSON object. */
export function renderOpenAIJson(answer: NormAnswer, model: string): Json {
  const message: Json = { role: 'assistant', content: answer.text };
  if (answer.toolCalls.length > 0) {
    message.tool_calls = answer.toolCalls.map((c, i) => ({
      id: c.id !== '' ? c.id : `call_${i}`,
      type: 'function',
      function: { name: c.name, arguments: c.args },
    }));
  }
  return {
    id: 'chatcmpl-aggregator',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: stopToOpenAI(answer.stop), logprobs: null }],
    usage: {
      prompt_tokens: answer.inputTokens,
      completion_tokens: answer.outputTokens,
      total_tokens: answer.inputTokens + answer.outputTokens,
    },
  };
}

/**
 * A normalised answer as an OpenAI SSE stream: the role chunk, the content deltas, the tool
 * calls, the finish chunk, the usage chunk, then [DONE]. Nothing follows [DONE].
 */
export function renderOpenAIStream(answer: NormAnswer, model: string): string {
  const chunk = (delta: Json, finish: string | null, usage: Json | null): Json => {
    const out: Json = { id: 'chatcmpl-aggregator', object: 'chat.completion.chunk', created: FIXED_CREATED, model };
    out.choices = finish === null && Object.keys(delta).length === 0 && usage !== null ? [] : [{ index: 0, delta, finish_reason: finish }];
    if (usage !== null) out.usage = usage;
    return out;
  };
  const frames = [sseEvent(null, chunk({ role: 'assistant', content: '' }, null, null))];
  for (const piece of splitText(answer.text)) {
    if (piece === '') continue;
    frames.push(sseEvent(null, chunk({ content: piece }, null, null)));
  }
  answer.toolCalls.forEach((call, i) => {
    frames.push(
      sseEvent(
        null,
        chunk(
          {
            tool_calls: [
              {
                index: i,
                id: call.id !== '' ? call.id : `call_${i}`,
                type: 'function',
                function: { name: call.name, arguments: call.args },
              },
            ],
          },
          null,
          null,
        ),
      ),
    );
  });
  frames.push(sseEvent(null, chunk({}, stopToOpenAI(answer.stop), null)));
  frames.push(
    sseEvent(
      null,
      chunk(
        {},
        null,
        {
          prompt_tokens: answer.inputTokens,
          completion_tokens: answer.outputTokens,
          total_tokens: answer.inputTokens + answer.outputTokens,
        },
      ),
    ),
  );
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/** A normalised answer as an Anthropic Messages JSON object. */
export function renderAnthropicJson(answer: NormAnswer, model: string): Json {
  const content: Json[] = [];
  if (answer.text !== '') content.push({ type: 'text', text: answer.text });
  for (const call of answer.toolCalls) {
    content.push({
      type: 'tool_use',
      id: call.id !== '' ? call.id : 'toolu_aggregator',
      name: call.name,
      input: argsObject(call.args),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  return {
    id: 'msg_aggregator',
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopToAnthropic(answer.stop),
    stop_sequence: null,
    usage: { input_tokens: answer.inputTokens, output_tokens: answer.outputTokens },
  };
}

/**
 * A normalised answer as an Anthropic SSE stream: message_start, one text block (start,
 * deltas, stop), one block per tool call, then message_delta with the stop reason and the
 * output usage, then message_stop. Nothing follows message_stop.
 */
export function renderAnthropicStream(answer: NormAnswer, model: string): string {
  const frames: string[] = [
    sseEvent('message_start', {
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
  ];
  let index = 0;
  if (answer.text !== '') {
    frames.push(
      sseEvent('content_block_start', { type: 'content_block_start', index, content_block: { type: 'text', text: '' } }),
    );
    for (const piece of splitText(answer.text)) {
      frames.push(
        sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'text_delta', text: piece } }),
      );
    }
    frames.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    index += 1;
  }
  for (const call of answer.toolCalls) {
    frames.push(
      sseEvent('content_block_start', {
        type: 'content_block_start',
        index,
        content_block: {
          type: 'tool_use',
          id: call.id !== '' ? call.id : `toolu_${index}`,
          name: call.name,
          input: {},
        },
      }),
    );
    if (call.args !== '') {
      frames.push(
        sseEvent('content_block_delta', {
          type: 'content_block_delta',
          index,
          delta: { type: 'input_json_delta', partial_json: call.args },
        }),
      );
    }
    frames.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    index += 1;
  }
  frames.push(
    sseEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: stopToAnthropic(answer.stop), stop_sequence: null },
      usage: { output_tokens: answer.outputTokens },
    }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return frames.join('');
}

/** One model this entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  displayName: string;
}

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header). */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.displayName, created_at: FIXED_CREATED_AT })),
    has_more: false,
  };
}

/** An error in OpenAI's shape. */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** An error in Anthropic's shape. */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}
