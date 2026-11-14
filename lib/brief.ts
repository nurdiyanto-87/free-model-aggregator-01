// Brief parsing, provider catalog probing, and catalog assembly.
//
// The trial brief (MAKEFUN_PROMPT_FILE) is prose with three embedded JSON
// documents, selected by their `schema` field: the provider directory, the
// quality table, and the workload manifest. Nothing here reads the clock or
// any randomness source: the same brief always yields the same catalog.

export type Json = { [key: string]: unknown };
export type Wire = 'openai' | 'anthropic' | 'gemini';

export interface DirectoryProvider {
  id: string;
  name: string;
  wire: Wire;
  auth: 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';
  path: string;
  keyEnv: string;
  key: string;
}

export interface RatingRow {
  model: string;
  provider: string;
  ratingMilli: number;
  contextLength: number;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

export interface ManifestRow {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: number;
  requires: { tools: boolean; vision: boolean; json: boolean; minContext: number };
}

export interface ListedModel {
  id: string;
  free: boolean;
  contextLength: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

export interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

export interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

export interface Catalog {
  providers: CatalogProvider[];
}

export interface Brief {
  providers: DirectoryProvider[];
  ratings: RatingRow[];
  manifest: ManifestRow[];
}

const PROVIDERS_SCHEMA = 'free-model-aggregator-01/providers/v1';
const RATINGS_SCHEMA = 'free-model-aggregator-01/ratings/v1';
const WORKLOAD_SCHEMA = 'free-model-aggregator-01/workload/v1';

export function isRecord(value: unknown): value is Json {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function asString(value: unknown, fallback = ''): string {
  return typeof value === 'string' ? value : fallback;
}

function asNumber(value: unknown, fallback = 0): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function asBool(value: unknown, fallback = false): boolean {
  return typeof value === 'boolean' ? value : fallback;
}

/**
 * Scan prose for top-level JSON objects. Braces inside strings are skipped,
 * and only objects that parse cleanly are returned, in document order.
 */
export function scanJsonDocuments(text: string): Json[] {
  const docs: Json[] = [];
  let i = 0;
  while (i < text.length) {
    const start = text.indexOf('{', i);
    if (start === -1) break;
    let depth = 0;
    let inString = false;
    let escaped = false;
    let j = start;
    for (; j < text.length; j += 1) {
      const ch = text[j];
      if (inString) {
        if (escaped) escaped = false;
        else if (ch === '\\') escaped = true;
        else if (ch === '"') inString = false;
        continue;
      }
      if (ch === '"') inString = true;
      else if (ch === '{') depth += 1;
      else if (ch === '}') {
        depth -= 1;
        if (depth === 0) break;
      }
    }
    if (depth !== 0) break;
    try {
      const parsed: unknown = JSON.parse(text.slice(start, j + 1));
      if (isRecord(parsed)) docs.push(parsed);
    } catch {
      // Not a JSON object; keep scanning past it.
    }
    i = j + 1;
  }
  return docs;
}

function parseDirectory(doc: Json): DirectoryProvider[] {
  const raw = Array.isArray(doc.providers) ? doc.providers : [];
  const out: DirectoryProvider[] = [];
  for (const item of raw) {
    if (!isRecord(item)) continue;
    const id = asString(item.id);
    const wire = asString(item.wire);
    if (id === '' || (wire !== 'openai' && wire !== 'anthropic' && wire !== 'gemini')) continue;
    const auth = asString(item.auth);
    out.push({
      id,
      name: asString(item.name, id),
      wire,
      auth: auth === 'bearer' || auth === 'x-api-key' || auth === 'x-goog-api-key' ? auth : 'none',
      path: asString(item.path),
      keyEnv: asString(item.key_env),
      key: asString(item.key),
    });
  }
  return out;
}

function parseRatings(doc: Json): RatingRow[] {
  const raw = Array.isArray(doc.ratings) ? doc.ratings : [];
  const out: RatingRow[] = [];
  for (const item of raw) {
    if (!isRecord(item)) continue;
    const model = asString(item.model);
    if (model === '') continue;
    out.push({
      model,
      provider: asString(item.provider),
      ratingMilli: asNumber(item.rating_milli),
      contextLength: asNumber(item.context_length),
      tools: asBool(item.tools),
      vision: asBool(item.vision),
      json: asBool(item.json),
      free: asBool(item.free, true),
    });
  }
  return out;
}

function parseManifest(doc: Json): ManifestRow[] {
  const raw = Array.isArray(doc.manifest) ? doc.manifest : [];
  const out: ManifestRow[] = [];
  for (const item of raw) {
    if (!isRecord(item)) continue;
    const id = asString(item.id);
    if (id === '') continue;
    const req = isRecord(item.requires) ? item.requires : {};
    out.push({
      id,
      api: asString(item.api) === 'anthropic' ? 'anthropic' : 'openai',
      stream: asBool(item.stream),
      weight: asNumber(item.weight, 1),
      requires: {
        tools: asBool(req.tools),
        vision: asBool(req.vision),
        json: asBool(req.json),
        minContext: asNumber(req.min_context),
      },
    });
  }
  return out;
}

/** Parse the brief text into its three documents; missing docs yield empty lists. */
export function parseBrief(text: string): Brief {
  const brief: Brief = { providers: [], ratings: [], manifest: [] };
  for (const doc of scanJsonDocuments(text)) {
    if (doc.schema === PROVIDERS_SCHEMA) brief.providers = parseDirectory(doc);
    else if (doc.schema === RATINGS_SCHEMA) brief.ratings = parseRatings(doc);
    else if (doc.schema === WORKLOAD_SCHEMA) brief.manifest = parseManifest(doc);
  }
  return brief;
}

/**
 * Join a provider path to the environment origin. A directory entry whose
 * path is already absolute is used as-is; otherwise it hangs off the env URL.
 */
export function joinBase(envUrl: string, path: string): string {
  if (/^https?:\/\//.test(path)) return path.replace(/\/$/, '');
  const base = envUrl.replace(/\/$/, '');
  return `${base}${path.startsWith('/') ? path : `/${path}`}`;
}

/** The auth headers a provider expects, from its directory entry. */
export function authHeaders(provider: DirectoryProvider): Record<string, string> {
  if (provider.key === '') return {};
  if (provider.auth === 'bearer') return { authorization: `Bearer ${provider.key}` };
  if (provider.auth === 'x-api-key') return { 'x-api-key': provider.key };
  if (provider.auth === 'x-goog-api-key') return { 'x-goog-api-key': provider.key };
  return {};
}

function listedFromRecord(item: Json, id: string): ListedModel {
  const ctx = item.context_length;
  return {
    id,
    free: asBool(item.free, true),
    contextLength: typeof ctx === 'number' && Number.isFinite(ctx) ? ctx : null,
    tools: typeof item.tools === 'boolean' ? item.tools : null,
    vision: typeof item.vision === 'boolean' ? item.vision : null,
    json: typeof item.json === 'boolean' ? item.json : null,
  };
}

/**
 * Parse one provider's model listing into a uniform shape. The three wires
 * differ: OpenAI and Anthropic list under `data` with a bare `id`; Gemini
 * lists under `models` with `name: "models/<id>"`.
 */
export function parseListing(provider: DirectoryProvider, body: unknown): ListedModel[] {
  if (!isRecord(body)) return [];
  if (provider.wire === 'gemini') {
    const raw = Array.isArray(body.models) ? body.models : [];
    const out: ListedModel[] = [];
    for (const item of raw) {
      if (!isRecord(item)) continue;
      const name = asString(item.name);
      const id = name.startsWith('models/') ? name.slice('models/'.length) : name;
      if (id !== '') out.push(listedFromRecord(item, id));
    }
    return out;
  }
  const raw = Array.isArray(body.data) ? body.data : [];
  const out: ListedModel[] = [];
  for (const item of raw) {
    if (!isRecord(item)) continue;
    const id = asString(item.id);
    if (id !== '') out.push(listedFromRecord(item, id));
  }
  return out;
}

function catalogPath(provider: DirectoryProvider): string {
  return provider.wire === 'gemini' ? '/v1beta/models' : '/v1/models';
}

/**
 * Probe one provider's catalog route. Returns null when the provider does
 * not answer with a usable listing; callers retry or drop the provider.
 */
export async function probeProvider(
  provider: DirectoryProvider,
  envUrl: string,
  timeoutMs = 5000,
): Promise<ListedModel[] | null> {
  const headers: Record<string, string> = { ...authHeaders(provider) };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  try {
    const res = await fetch(`${joinBase(envUrl, provider.path)}${catalogPath(provider)}`, {
      method: 'GET',
      headers,
      signal: AbortSignal.timeout(timeoutMs),
    });
    if (!res.ok) return null;
    const body: unknown = await res.json();
    return parseListing(provider, body);
  } catch {
    return null;
  }
}

/**
 * Probe every provider, retrying a failed probe twice so a transient hiccup
 * does not drop a provider from the catalog. Probes run in parallel; the
 * result keeps brief order, so the output is deterministic.
 */
export async function probeAll(
  providers: readonly DirectoryProvider[],
  envUrl: string,
): Promise<Map<string, ListedModel[]>> {
  const results = await Promise.all(
    providers.map(async (provider) => {
      for (let attempt = 0; attempt < 3; attempt += 1) {
        const listed = await probeProvider(provider, envUrl);
        if (listed !== null) return [provider.id, listed] as const;
      }
      return [provider.id, null] as const;
    }),
  );
  const map = new Map<string, ListedModel[]>();
  for (const [id, listed] of results) {
    if (listed !== null) map.set(id, listed);
  }
  return map;
}

/**
 * Build the catalog document. The quality table is the authority for whether
 * a model is free and how much context it has; the probe listing fills gaps
 * (notably the paid model, which the quality table omits). Models are
 * deduplicated and sorted by id within each provider; providers keep brief
 * order. `base_url` is the directory path itself, never joined to the env
 * origin: the origin differs between a live trial and its replay.
 */
export function buildCatalog(
  brief: Brief,
  listings: Map<string, ListedModel[]>,
): Catalog {
  const ratingByKey = new Map<string, RatingRow>();
  for (const row of brief.ratings) ratingByKey.set(`${row.provider}\0${row.model}`, row);

  const providers: CatalogProvider[] = [];
  for (const provider of brief.providers) {
    const listed = listings.get(provider.id);
    if (listed === undefined) continue;
    const byId = new Map<string, CatalogModel>();
    for (const model of listed) {
      const row = ratingByKey.get(`${provider.id}\0${model.id}`);
      const free = row !== undefined ? row.free : model.free;
      const context = row !== undefined ? row.contextLength : model.contextLength;
      if (!byId.has(model.id)) byId.set(model.id, { id: model.id, free, context_length: context });
    }
    const models = [...byId.values()].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
    providers.push({ id: provider.id, base_url: provider.path, key_env: provider.keyEnv, wire: provider.wire, models });
  }
  return { providers };
}

/** The free model ids a provider serves, from the catalog, in catalog order. */
export function freeIdsOf(catalog: Catalog): string[] {
  const seen = new Set<string>();
  const ids: string[] = [];
  for (const provider of catalog.providers) {
    for (const model of provider.models) {
      if (model.free && !seen.has(model.id)) {
        seen.add(model.id);
        ids.push(model.id);
      }
    }
  }
  return ids;
}
