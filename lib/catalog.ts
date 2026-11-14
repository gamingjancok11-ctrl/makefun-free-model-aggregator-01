// The catalog probe: ask every provider for its model list, in its own wire, and build the
// one catalog document that `--list` prints and the register body carries.
//
// Each provider's catalog route lists free and paid models, with each model's context length
// and capabilities. The probe speaks each wire's own listing:
//
//   openai     GET <base>/v1/models       Authorization: Bearer <key>
//   anthropic  GET <base>/v1/models       x-api-key: <key>, anthropic-version: 2023-06-01
//   gemini     GET <base>/v1beta/models   x-goog-api-key: <key>
//
// The wire does not decide the header: the auth scheme from the brief does (zai speaks the
// Anthropic wire but takes a Bearer token, for example). A probe that fails is retried once;
// a provider that still does not answer is left out of the catalog and gets no traffic.
//
// The catalog's `free` flag and context length come from the quality table's row where one
// exists (the table is the sealed truth for free models), and from the listing itself where
// it does not (the one paid model has no row in the brief's free-only table, so the listing's
// own `free: false` is what keeps it out of the free set). The catalog's `base_url` is the
// directory's own path (`/p/<id>`), never joined to the env origin: the origin differs
// between a live pass and its replay, and the catalog is part of the state export a replay
// must reproduce byte for byte.

import { authHeaders, joinBase, ratingRowFor, type Brief, type Json, type Provider, type RatingRow, type Wire } from './brief.ts';

/** One model as a provider's catalog route listed it; null members are what it did not say. */
export interface ListedModel {
  id: string;
  displayName: string;
  free: boolean | null;
  contextLength: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

/** One provider's catalog probe: what its route answered, and the models it listed. */
export interface Probe {
  provider: Provider;
  ok: boolean;
  status: number;
  models: ListedModel[];
  error: string | null;
}

/** One model in the catalog document. */
export interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

/** One provider in the catalog document. */
export interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

/** What `--list` prints and what the entry POSTs to /register: the same document. */
export interface Catalog {
  providers: CatalogProvider[];
}

function numberOrNull(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function boolOrNull(value: unknown): boolean | null {
  return typeof value === 'boolean' ? value : null;
}

function stringOf(value: unknown): string {
  return typeof value === 'string' ? value : '';
}

/** Whether a listing's pricing block prices the model at zero; null when it says nothing. */
function zeroPriced(value: Json): boolean | null {
  const pricing = value.pricing;
  if (pricing === null || typeof pricing !== 'object' || Array.isArray(pricing)) return null;
  const values = Object.values(pricing as Record<string, unknown>).flatMap((x) => {
    if (typeof x === 'number') return [x];
    if (typeof x === 'string' && x.trim() !== '') return [Number(x)];
    return [];
  });
  if (values.length === 0) return null;
  return values.every((n) => Number.isFinite(n) && n === 0);
}

/** One model as a listing wrote it, in whichever wire's spelling. */
function modelIn(wire: Wire, value: unknown): ListedModel | null {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) return null;
  const row = value as Json;
  const id = wire === 'gemini' ? stringOf(row.name).replace(/^models\//, '') : stringOf(row.id);
  if (id === '') return null;
  const displayName =
    stringOf(row.display_name) !== '' ? stringOf(row.display_name) : stringOf(row.displayName) !== '' ? stringOf(row.displayName) : id;
  return {
    id,
    displayName,
    free: boolOrNull(row.free) ?? zeroPriced(row),
    contextLength: numberOrNull(row.context_length) ?? numberOrNull(row.context_window) ?? numberOrNull(row.inputTokenLimit),
    tools: boolOrNull(row.tools),
    vision: boolOrNull(row.vision),
    json: boolOrNull(row.json),
  };
}

/** The models one catalog route's answer listed, in its wire's shape. */
function modelsIn(wire: Wire, body: unknown): ListedModel[] {
  if (body === null || typeof body !== 'object' || Array.isArray(body)) return [];
  const row = body as Json;
  const list = wire === 'gemini' ? row.models : row.data;
  if (!Array.isArray(list)) return [];
  return list.flatMap((v) => {
    const m = modelIn(wire, v);
    return m === null ? [] : [m];
  });
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** Probe one provider's catalog route, in its own wire, with one retry on failure. */
export async function probeProvider(envUrl: string, provider: Provider, timeoutMs = 5000): Promise<Probe> {
  const base = joinBase(envUrl, provider.base);
  const url = provider.wire === 'gemini' ? `${base}/v1beta/models` : `${base}/v1/models`;
  const headers: Record<string, string> = { accept: 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  let lastError = '';
  for (let attempt = 0; attempt < 2; attempt++) {
    try {
      const res = await fetch(url, { headers, signal: AbortSignal.timeout(timeoutMs) });
      if (!res.ok) {
        lastError = `GET ${url} answered HTTP ${res.status}`;
      } else {
        const body: unknown = await res.json();
        return { provider, ok: true, status: res.status, models: modelsIn(provider.wire, body), error: null };
      }
    } catch (err) {
      lastError = `GET ${url} failed: ${(err as Error).message}`;
    }
    if (attempt === 0) await sleep(250);
  }
  return { provider, ok: false, status: 0, models: [], error: lastError };
}

/**
 * Whether a listed model is free: the quality table's row where it has one, the listing's own
 * word where it does not, and paid when neither says. A paid model is never routed.
 */
export function modelIsFree(model: ListedModel, row: RatingRow | null): boolean {
  if (row !== null) return row.free;
  return model.free ?? false;
}

/** What a model can do: the quality table's row where it has one, the listing's fields otherwise. */
export function modelCapabilities(
  model: ListedModel,
  row: RatingRow | null,
): { contextLength: number | null; tools: boolean | null; vision: boolean | null; json: boolean | null } {
  return {
    contextLength: row?.contextLength ?? model.contextLength ?? null,
    tools: row?.tools ?? model.tools ?? null,
    vision: row?.vision ?? model.vision ?? null,
    json: row?.json ?? model.json ?? null,
  };
}

/**
 * The catalog the probes and the quality table make: every provider that answered, every model
 * it listed, models sorted by id within each provider, providers in the brief's order.
 */
export function buildCatalog(probes: readonly Probe[], brief: Brief): Catalog {
  const providers: CatalogProvider[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    const seen = new Set<string>();
    const models: CatalogModel[] = [];
    const sorted = [...probe.models].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
    for (const model of sorted) {
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      const row = ratingRowFor(brief, probe.provider.id, model.id);
      models.push({
        id: model.id,
        free: modelIsFree(model, row),
        context_length: row?.contextLength ?? model.contextLength ?? null,
      });
    }
    providers.push({
      id: probe.provider.id,
      base_url: probe.provider.base,
      key_env: probe.provider.keyEnv,
      wire: probe.provider.wire,
      models,
    });
  }
  return { providers };
}

/** One model this entry serves, as GET /v1/models lists it: each free id once, first-seen wins. */
export interface ServedModel {
  id: string;
  provider: string;
  displayName: string;
}

export function servedModels(probes: readonly Probe[], brief: Brief): ServedModel[] {
  const out: ServedModel[] = [];
  const seen = new Set<string>();
  for (const probe of probes) {
    if (!probe.ok) continue;
    const sorted = [...probe.models].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
    for (const model of sorted) {
      if (seen.has(model.id)) continue;
      const row = ratingRowFor(brief, probe.provider.id, model.id);
      if (!modelIsFree(model, row)) continue;
      seen.add(model.id);
      out.push({ id: model.id, provider: probe.provider.id, displayName: model.displayName });
    }
  }
  return out;
}
