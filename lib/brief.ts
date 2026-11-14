// The trial brief: the provider directory, the quality table and the workload manifest.
//
// The brief is a prose file with three JSON documents embedded in it. Each document carries a
// `schema` member, which is how this file finds them; the prose around them is ignored. The
// three documents are:
//
//   - providers (free-model-aggregator-01/providers/v1): one row per provider: id, name, wire
//     (openai | anthropic | gemini), auth scheme, the key environment variable a real deployment
//     would read, the key the simulated service takes, and the base path to join to
//     $MAKEFUN_ENV_URL when dialling it.
//   - ratings (free-model-aggregator-01/ratings/v1): every free model's public rating in
//     milli-points, its context length, whether it takes tools, images and JSON mode, and
//     whether it is free. This is the table the router ranks candidates by; it is the brief's
//     copy of the sealed quality table, and it only lists free models.
//   - workload (free-model-aggregator-01/workload/v1): the requests the env's client will send,
//     in arrival order, each with its API, whether it streams, its weight and the requirements
//     a model must meet to serve it.
//
// Nothing here reads the clock or any randomness: parsing is a pure function of the file.

import { readFileSync } from 'node:fs';

export type Wire = 'openai' | 'anthropic' | 'gemini';
export type AuthScheme = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';

export const PROVIDERS_SCHEMA = 'free-model-aggregator-01/providers/v1';
export const RATINGS_SCHEMA = 'free-model-aggregator-01/ratings/v1';
export const WORKLOAD_SCHEMA = 'free-model-aggregator-01/workload/v1';

/** Any JSON object. */
export type Json = Record<string, unknown>;

export function isRecord(value: unknown): value is Json {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

/** One provider as the brief's directory lists it. `base` is its path under the env origin. */
export interface Provider {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthScheme;
  base: string;
  keyEnv: string;
  key: string;
}

/** One row of the quality table. */
export interface RatingRow {
  model: string;
  provider: string;
  ratingMilli: number;
  contextLength: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

/** What one request needs from its model. */
export interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  minContext?: number;
}

/** One row of the workload manifest. */
export interface ManifestEntry {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: number;
  requires: Requirement;
}

/** The brief as this server reads it. `error` names the problem when it could not be read. */
export interface Brief {
  providers: Provider[];
  ratings: RatingRow[];
  best: { id: string; ratingMilli: number }[];
  manifest: ManifestEntry[];
  error: string | null;
}

function textOf(value: unknown): string {
  return typeof value === 'string' ? value : '';
}

function intOrNull(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

/** The index of the `}` that closes the object opening at `start`, or -1. Strings are skipped. */
function objectEnd(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

/** Every JSON object in `text` that carries a string `schema`, in the order they appear. */
function documentsIn(text: string): Json[] {
  const out: Json[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    const end = objectEnd(text, start);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') out.push(value);
    } catch {
      // Not a document on its own: keep scanning.
    }
  }
  return out;
}

function wireOf(value: unknown): Wire | null {
  return value === 'openai' || value === 'anthropic' || value === 'gemini' ? value : null;
}

function authOf(value: unknown): AuthScheme {
  if (value === 'x-api-key' || value === 'x-goog-api-key' || value === 'none') return value;
  return 'bearer';
}

function providerOf(value: unknown): Provider | null {
  if (!isRecord(value)) return null;
  const id = textOf(value.id);
  const wire = wireOf(value.wire);
  if (id === '' || wire === null) return null;
  // The brief writes `path` (/p/<id>); accept the absolute spellings too, `path` first.
  const base =
    textOf(value.path) !== ''
      ? textOf(value.path)
      : textOf(value.base_path) !== ''
        ? textOf(value.base_path)
        : textOf(value.base_url) !== ''
          ? textOf(value.base_url)
          : textOf(value.base);
  return {
    id,
    name: textOf(value.name) !== '' ? textOf(value.name) : id,
    wire,
    auth: authOf(value.auth),
    base,
    keyEnv: textOf(value.key_env),
    key: textOf(value.key),
  };
}

function ratingOf(value: unknown): RatingRow | null {
  if (!isRecord(value)) return null;
  const model = textOf(value.model);
  const provider = textOf(value.provider);
  if (model === '' || provider === '') return null;
  return {
    model,
    provider,
    ratingMilli: intOrNull(value.rating_milli) ?? 0,
    contextLength: intOrNull(value.context_length),
    tools: value.tools === true,
    vision: value.vision === true,
    json: value.json === true,
    free: value.free === true,
  };
}

function manifestEntryOf(value: unknown): ManifestEntry | null {
  if (!isRecord(value)) return null;
  const id = textOf(value.id);
  if (id === '') return null;
  const requires = isRecord(value.requires) ? value.requires : {};
  const entry: ManifestEntry = {
    id,
    api: value.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: value.stream === true,
    weight: value.weight === 2 ? 2 : 1,
    requires: {
      tools: requires.tools === true,
      vision: requires.vision === true,
      json: requires.json === true,
    },
  };
  const minContext = intOrNull(requires.min_context);
  if (minContext !== null) entry.requires.minContext = minContext;
  return entry;
}

const EMPTY_BRIEF: Brief = { providers: [], ratings: [], best: [], manifest: [], error: null };

/** The brief at `path`, its three documents found by schema. */
export function readBrief(path: string): Brief {
  if (path === '') return { ...EMPTY_BRIEF, error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSyncUtf8(path);
  } catch (err) {
    return { ...EMPTY_BRIEF, error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = documentsIn(text);
  const directory = docs.find((d) => d.schema === PROVIDERS_SCHEMA && Array.isArray(d.providers)) ?? null;
  if (directory === null || !Array.isArray(directory.providers)) {
    return { ...EMPTY_BRIEF, error: `the brief at ${path} holds no ${PROVIDERS_SCHEMA} document` };
  }
  const providers = directory.providers.flatMap((v) => {
    const p = providerOf(v);
    return p === null ? [] : [p];
  });
  const ratingsDoc = docs.find((d) => d.schema === RATINGS_SCHEMA && Array.isArray(d.ratings)) ?? null;
  const ratings =
    ratingsDoc !== null && Array.isArray(ratingsDoc.ratings)
      ? ratingsDoc.ratings.flatMap((v) => {
          const r = ratingOf(v);
          return r === null ? [] : [r];
        })
      : [];
  const best =
    ratingsDoc !== null && Array.isArray(ratingsDoc.best)
      ? ratingsDoc.best.flatMap((v) => {
          if (!isRecord(v) || textOf(v.id) === '') return [];
          return [{ id: textOf(v.id), ratingMilli: intOrNull(v.rating_milli) ?? 0 }];
        })
      : [];
  const workloadDoc = docs.find((d) => d.schema === WORKLOAD_SCHEMA && Array.isArray(d.manifest)) ?? null;
  const manifest =
    workloadDoc !== null && Array.isArray(workloadDoc.manifest)
      ? workloadDoc.manifest.flatMap((v) => {
          const m = manifestEntryOf(v);
          return m === null ? [] : [m];
        })
      : [];
  return { providers, ratings, best, manifest, error: null };
}

function readFileSyncUtf8(path: string): string {
  return readFileSync(path, 'utf8');
}

/** The quality-table row for one provider's model, when the table has one. */
export function ratingRowFor(brief: Brief, providerId: string, modelId: string): RatingRow | null {
  return brief.ratings.find((r) => r.provider === providerId && r.model === modelId) ?? null;
}

/** A base path joined to the env origin; an absolute base URL is used as it stands. */
export function joinBase(envUrl: string, base: string): string {
  const b = base.trim();
  if (/^https?:\/\//i.test(b)) return b.replace(/\/+$/, '');
  const prefix = envUrl.replace(/\/+$/, '');
  if (b === '') return prefix;
  return `${prefix}${b.startsWith('/') ? '' : '/'}${b.replace(/\/+$/, '')}`;
}

/** The headers that carry one provider's key, in its own scheme. */
export function authHeaders(auth: AuthScheme, key: string): Record<string, string> {
  if (key === '') return {};
  if (auth === 'bearer') return { authorization: `Bearer ${key}` };
  if (auth === 'x-api-key') return { 'x-api-key': key };
  if (auth === 'x-goog-api-key') return { 'x-goog-api-key': key };
  return {};
}
