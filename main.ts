// free-model-aggregator-01: one server, two client APIs, ten simulated free-tier providers.
//
// The server reads the trial brief (the provider directory, the quality table and the
// workload manifest), probes every provider's catalog route in its own wire, builds one
// catalog, registers its port and catalog with the env, and then serves the env client's
// 20 requests: POST /v1/chat/completions and POST /v1/messages, each in JSON or in
// server-sent events, answered by the best free model that can serve them. When the env
// says the workload is done it exits 0. `node main.ts --list` prints the catalog document
// and exits without starting the server.
//
// ROUTING (quality is 60% of the score)
//   Every request is served by the highest-rated free model that meets its requirements.
//   Candidates are ranked by rating, with a small penalty per call already made to their
//   provider: the best models cluster on a few providers whose free-tier budgets are drawn
//   per trial and never published, so the penalty spreads the load across the near-best
//   providers instead of draining the top one and failing over late. A provider that fails
//   twice in a row is deprioritised heavily (it is probably out of budget or down) but kept
//   as a last resort. The ranking is a pure function of the brief and the calls made so
//   far: no clock, no randomness, so a replay routes identically.
//
//   One wire constraint shapes the candidates: a JSON-mode request from an OpenAI client
//   must go to an OpenAI- or Gemini-wire provider, because the Anthropic wire cannot carry
//   JSON mode and its provider would echo it absent. (An Anthropic client's JSON request
//   carries no JSON-mode field on its own wire, so any wire may serve it.)
//
// FAILOVER (10% of the score)
//   A provider refuses with 402 when its free allowance is gone (one provider) or 429 on
//   the other nine, sometimes with a retry-after. A refusal is never the answer: the top
//   candidate is retried once after its retry-after (a scheduled refusal spends that call
//   number, so the next call number usually answers), and then the request fails over to
//   the next candidate, until one answers. The retry is spent only on a provider that has
//   never refused before: one that already refused is probably spent, and retrying it
//   would burn a call for nothing. A 401 or 400 is not retried on the same
//   provider. Only the first success is returned: exactly one provider call answers each
//   request. Upstream calls are serialised, one at a time, so the provider that serves one
//   call at a time never sees a concurrent second call.
//
// THE WIRES
//   Requests are normalised out of the client's wire and rendered into the provider's;
//   answers are parsed out of the provider's wire and rendered into the client's (see
//   lib/wire.ts). Every field the provider echoes — system prompt, user text, images,
//   tools, max_tokens, stop sequences, JSON mode — passes through untouched. Streaming
//   answers are buffered, parsed and re-emitted in the client's event order, so a
//   translated stream is well-formed in its own wire rather than a foreign stream passed
//   through.
//
// Only the Node 22 standard library is used; nothing is installed before the server runs.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { readBrief, ratingRowFor, joinBase, authHeaders, isRecord, type Brief, type Json, type Provider, type Requirement } from './lib/brief.ts';
import { buildCatalog, modelCapabilities, modelIsFree, probeProvider, servedModels, type Catalog, type ListedModel, type Probe } from './lib/catalog.ts';
import {
  anthropicError,
  anthropicModelList,
  normalizeAnthropic,
  normalizeOpenAI,
  openAIError,
  openAIModelList,
  parseAnthropicAnswer,
  parseAnthropicStream,
  parseGeminiAnswer,
  parseGeminiStream,
  parseOpenAIAnswer,
  parseOpenAIStream,
  renderAnthropic,
  renderAnthropicJson,
  renderAnthropicStream,
  renderGemini,
  renderOpenAI,
  renderOpenAIJson,
  renderOpenAIStream,
  type NormAnswer,
  type NormRequest,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one upstream provider call may take, in milliseconds. */
const UPSTREAM_TIMEOUT_MS = 10000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** Rating milli-points charged per call already made to a provider: spreads the load. */
const LOAD_PENALTY_MILLI = 2;
/** Rating milli-points charged per refusal beyond the first, per provider: a provider that
 *  refused twice is probably out of budget or down, not merely on its failure schedule. */
const REFUSAL_PENALTY_MILLI = 400;
/** Rating milli-points charged to a provider that failed twice in a row: near-exclusion. */
const SUSPECT_PENALTY_MILLI = 600;
/** The longest a retry-after is honoured, in milliseconds. */
const MAX_RETRY_AFTER_MS = 2000;
/** The Anthropic API version sent to Anthropic-wire providers. */
const ANTHROPIC_VERSION = '2023-06-01';

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief: Brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: Catalog = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one at a time. */
let manifestIndex = 0;
/** Calls attempted per provider, successes and refusals alike: what the budgets count. */
const callsMade = new Map<string, number>();
/** Consecutive failures per provider, reset by a success. */
const consecutiveFailures = new Map<string, number>();
/** Refusals (402, 429 or 5xx) per provider over the whole trial: exhaustion leaves a trail. */
const refusals = new Map<string, number>();
/** POST handlers run one at a time, so providers never see a concurrent second call. */
let queue: Promise<void> = Promise.resolve();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(probes, brief);
}

/** One free model that could serve a request, with its provider and its rating. */
interface Candidate {
  provider: Provider;
  model: ListedModel;
  rating: number;
}

/** Whether a model's capabilities meet one request's requirements. Unknown never meets one. */
function meets(
  requires: Requirement,
  caps: { contextLength: number | null; tools: boolean | null; vision: boolean | null; json: boolean | null },
): boolean {
  if (requires.tools === true && caps.tools !== true) return false;
  if (requires.vision === true && caps.vision !== true) return false;
  if (requires.json === true && caps.json !== true) return false;
  if (requires.minContext !== undefined && (caps.contextLength === null || caps.contextLength < requires.minContext)) return false;
  return true;
}

/**
 * The free candidates for one request, best first: rating, less a small penalty per call
 * already made to the provider (which spreads the load across near-best providers), less
 * a penalty per refusal beyond the first (a provider that refused twice is probably out
 * of budget or down, not merely on its failure schedule), less a large penalty for a
 * provider that failed twice in a row. Ties break deterministically on rating, then
 * load, then provider id, then model id. A JSON-mode request from an OpenAI client
 * keeps only OpenAI- and Gemini-wire providers: the Anthropic wire cannot carry
 * JSON mode.
 */
function candidatesFor(requires: Requirement, clientApi: 'openai' | 'anthropic', req: NormRequest): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (clientApi === 'openai' && req.jsonMode && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingRowFor(brief, probe.provider.id, model.id);
      if (!modelIsFree(model, row)) continue;
      if (!meets(requires, modelCapabilities(model, row))) continue;
      out.push({ provider: probe.provider, model, rating: row?.ratingMilli ?? 0 });
    }
  }
  const scoreOf = (c: Candidate): number => {
    const load = callsMade.get(c.provider.id) ?? 0;
    const refused = refusals.get(c.provider.id) ?? 0;
    const suspect = (consecutiveFailures.get(c.provider.id) ?? 0) >= 2 ? SUSPECT_PENALTY_MILLI : 0;
    return c.rating - LOAD_PENALTY_MILLI * load - REFUSAL_PENALTY_MILLI * Math.max(0, refused - 1) - suspect;
  };
  out.sort((a, b) => {
    const byScore = scoreOf(b) - scoreOf(a);
    if (byScore !== 0) return byScore;
    if (b.rating !== a.rating) return b.rating - a.rating;
    const loadA = callsMade.get(a.provider.id) ?? 0;
    const loadB = callsMade.get(b.provider.id) ?? 0;
    if (loadA !== loadB) return loadA - loadB;
    if (a.provider.id !== b.provider.id) return a.provider.id < b.provider.id ? -1 : 1;
    return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
  });
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(clientApi: 'openai' | 'anthropic', body: Json): Requirement {
  const entry = brief.manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(clientApi, body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(clientApi: 'openai' | 'anthropic', body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (!isRecord(block)) continue;
      if (block.type === 'image' || block.type === 'image_url') vision = true;
    }
  }
  const tools = Array.isArray(body.tools) && body.tools.length > 0;
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    tools,
    vision,
    json: clientApi === 'openai' && (format === 'json_object' || format === 'json_schema'),
  };
}

/** One upstream call's outcome: an answer, or a refusal with its status, code and retry-after. */
type UpstreamResult =
  | { ok: true; answer: NormAnswer }
  | { ok: false; status: number; code: string; retryAfterMs: number; retryable: boolean };

/** A retry-after in milliseconds, from a response header or an error body; 0 when there is none. */
function retryAfterOf(res: Response, body: unknown): number {
  const header = res.headers.get('retry-after');
  if (header !== null) {
    const seconds = Number(header);
    if (Number.isFinite(seconds) && seconds > 0) return Math.min(seconds * 1000, MAX_RETRY_AFTER_MS);
  }
  if (isRecord(body) && isRecord(body.error)) {
    const seconds = body.error.retry_after;
    if (typeof seconds === 'number' && Number.isFinite(seconds) && seconds > 0) return Math.min(seconds * 1000, MAX_RETRY_AFTER_MS);
  }
  return 0;
}

/** The error code of a provider's refusal, in whichever wire's envelope it came. */
function errorCodeOf(body: unknown): string {
  if (!isRecord(body)) return '';
  if (isRecord(body.error)) {
    if (typeof body.error.code === 'string') return body.error.code;
    if (typeof body.error.type === 'string') return body.error.type;
    if (typeof body.error.status === 'string') return body.error.status;
  }
  return '';
}

/**
 * One call to one provider: the request rendered in the provider's wire, the answer parsed
 * out of it. A refusal (402, 429, 5xx) and a network failure are retryable; a 400 or 401
 * is not. Streaming answers are buffered whole and then parsed, so a translated stream is
 * emitted well-formed in the client's wire.
 */
async function callProvider(candidate: Candidate, req: NormRequest): Promise<UpstreamResult> {
  const provider = candidate.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  let url: string;
  let upstreamBody: Json;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    upstreamBody = renderOpenAI(req, candidate.model.id);
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = ANTHROPIC_VERSION;
    upstreamBody = renderAnthropic(req, candidate.model.id);
  } else {
    const action = req.stream ? 'streamGenerateContent' : 'generateContent';
    url = `${base}/v1beta/models/${encodeURIComponent(candidate.model.id)}:${action}`;
    upstreamBody = renderGemini(req);
  }
  if (req.stream) headers.accept = 'text/event-stream';
  let res: Response;
  try {
    res = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(upstreamBody),
      signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
    });
  } catch (err) {
    return { ok: false, status: 0, code: 'unreachable', retryAfterMs: 0, retryable: true };
  }
  const text = await res.text();
  if (!res.ok) {
    let parsed: unknown = null;
    try {
      parsed = JSON.parse(text);
    } catch {
      // An unreadable error body still refuses.
    }
    const retryable = res.status === 402 || res.status === 429 || res.status >= 500;
    return { ok: false, status: res.status, code: errorCodeOf(parsed), retryAfterMs: retryAfterOf(res, parsed), retryable };
  }
  try {
    if (req.stream) {
      if (provider.wire === 'openai') return { ok: true, answer: parseOpenAIStream(text) };
      if (provider.wire === 'anthropic') return { ok: true, answer: parseAnthropicStream(text) };
      return { ok: true, answer: parseGeminiStream(text) };
    }
    const parsed: unknown = JSON.parse(text);
    if (!isRecord(parsed)) return { ok: false, status: 502, code: 'bad_answer', retryAfterMs: 0, retryable: true };
    if (provider.wire === 'openai') return { ok: true, answer: parseOpenAIAnswer(parsed) };
    if (provider.wire === 'anthropic') return { ok: true, answer: parseAnthropicAnswer(parsed) };
    return { ok: true, answer: parseGeminiAnswer(parsed) };
  } catch {
    return { ok: false, status: 502, code: 'bad_answer', retryAfterMs: 0, retryable: true };
  }
}

function recordCall(providerId: string, failed: boolean, refused: boolean): void {
  callsMade.set(providerId, (callsMade.get(providerId) ?? 0) + 1);
  consecutiveFailures.set(providerId, failed ? (consecutiveFailures.get(providerId) ?? 0) + 1 : 0);
  if (refused) refusals.set(providerId, (refusals.get(providerId) ?? 0) + 1);
}

/**
 * Serve one request: try the candidates best-first. The top candidate is retried once
 * after its retry-after — but only when it has never refused before, so a retry is
 * spent on a scheduled refusal (whose next call number usually answers) and never on a
 * provider that already looks spent. Every later candidate gets one attempt. The first
 * success wins, so exactly one provider call answers the request.
 */
async function serve(
  clientApi: 'openai' | 'anthropic',
  req: NormRequest,
  requires: Requirement,
): Promise<{ answer: NormAnswer; model: string } | null> {
  const candidates = candidatesFor(requires, clientApi, req);
  for (let i = 0; i < candidates.length; i++) {
    const candidate = candidates[i];
    const refusedBefore = refusals.get(candidate.provider.id) ?? 0;
    const first = await callProvider(candidate, req);
    recordCall(candidate.provider.id, !first.ok, !first.ok && first.retryable);
    if (first.ok) return { answer: first.answer, model: candidate.model.id };
    if (i === 0 && first.retryable && refusedBefore === 0) {
      // One retry on the top candidate, after the retry-after it asked for.
      if (first.retryAfterMs > 0) await sleep(first.retryAfterMs);
      const retry = await callProvider(candidate, req);
      recordCall(candidate.provider.id, !retry.ok, !retry.ok && retry.retryable);
      if (retry.ok) return { answer: retry.answer, model: candidate.model.id };
      log(
        `${candidate.provider.id}/${candidate.model.id} refused twice (${first.status} ${first.code}, then ${retry.status} ${retry.code})`,
      );
    } else {
      log(`${candidate.provider.id}/${candidate.model.id} refused (${first.status} ${first.code}); failing over`);
    }
  }
  return null;
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendStream(res: ServerResponse, text: string): void {
  res.writeHead(200, {
    'content-type': 'text/event-stream',
    'cache-control': 'no-cache',
    connection: 'keep-alive',
    'content-length': Buffer.byteLength(text),
  });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels(probes, brief);
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** POST /v1/chat/completions. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const norm = normalizeOpenAI(body);
  const served = await serve('openai', norm, requirementsFor('openai', body));
  if (served === null) return sendError(res, 'openai', 503, 'no_free_model', 'no free model could serve this request');
  if (norm.stream) sendStream(res, renderOpenAIStream(served.answer, served.model));
  else sendJson(res, 200, renderOpenAIJson(served.answer, served.model));
}

/** POST /v1/messages. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const norm = normalizeAnthropic(body);
  const served = await serve('anthropic', norm, requirementsFor('anthropic', body));
  if (served === null) return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model could serve this request');
  if (norm.stream) sendStream(res, renderAnthropicStream(served.answer, served.model));
  else sendJson(res, 200, renderAnthropicJson(served.answer, served.model));
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  // POST handlers are serialised: providers never see a concurrent second call.
  if (req.method === 'POST' && (url.pathname === '/v1/chat/completions' || url.pathname === '/v1/messages')) {
    const run = queue.then(() => (url.pathname === '/v1/messages' ? messages(req, res) : chat(req, res)));
    queue = run.then(
      () => undefined,
      () => undefined,
    );
    await run;
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; retries briefly, then gives up. */
async function register(port: number, report: Catalog): Promise<void> {
  let lastError = '';
  for (let attempt = 0; attempt < 5; attempt++) {
    try {
      const res = await fetch(`${ENV_URL}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
        signal: AbortSignal.timeout(5000),
      });
      if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
      const reply: unknown = await res.json();
      if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
      return;
    } catch (err) {
      lastError = (err as Error).message;
      await sleep(500);
    }
  }
  throw new Error(`register failed: ${lastError}`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object
    // and stop before any server starts. Wait for stdout to flush so the tail is not lost.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) {
        sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
      }
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels(probes, brief).length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
