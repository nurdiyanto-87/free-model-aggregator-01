// Free-model aggregator entrypoint.
//
// Serves an OpenAI-compatible and an Anthropic-compatible API over the ten
// simulated free-tier providers named in the trial brief. Each request is
// routed to the best free model that meets its requirements, with failover
// across candidates when a provider refuses, and the provider's answer is
// returned whole in the client's wire, streamed or not as asked.
//
// Flow: read the brief, probe every provider's catalog route, build the
// catalog, listen, register once with the environment, serve the driver's
// requests, poll /work until done, exit 0. `--list` prints the catalog
// document and exits without serving.
//
// Determinism: no clock reads and no randomness anywhere in request handling
// or in anything sent. Routing order is a pure function of the brief, the
// manifest, and the outcomes of earlier calls in this trial.

import { readFileSync } from 'node:fs';
import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';

import {
  authHeaders,
  buildCatalog,
  freeIdsOf,
  isRecord,
  joinBase,
  parseBrief,
  probeAll,
  type Brief,
  type Catalog,
  type DirectoryProvider,
  type Json,
  type ListedModel,
  type RatingRow,
  type Wire,
} from './lib/brief.ts';
import {
  anthropicModelList,
  buildUpstreamBody,
  errorBody,
  extractAnswer,
  openAIModelList,
  parseAnthropicRequest,
  parseOpenAIRequest,
  renderAnthropicJson,
  renderAnthropicStream,
  renderOpenAIJson,
  renderOpenAIStream,
  upstreamHeaders,
  upstreamPath,
  type Canonical,
  type ClientWire,
  type ProviderAnswer,
  type ServedModel,
} from './lib/wires.ts';

const MAX_BODY_BYTES = 32 * 1024 * 1024;
const UPSTREAM_TIMEOUT_MS = 5000;
const POLL_INTERVAL_MS = 100;

interface Candidate {
  provider: DirectoryProvider;
  providerIndex: number;
  model: string;
  ratingMilli: number;
  contextLength: number;
  tools: boolean;
  vision: boolean;
  json: boolean;
}

interface ProviderState {
  calls: number;
  consecutiveRefusals: number;
}

interface Requirements {
  tools: boolean;
  vision: boolean;
  json: boolean;
  minContext: number;
}

/** Load the brief text, or '' when the prompt file is absent or unreadable. */
function loadBriefText(): string {
  const path = process.env.MAKEFUN_PROMPT_FILE;
  if (path === undefined || path === '') return '';
  try {
    return readFileSync(path, 'utf8');
  } catch {
    return '';
  }
}

/**
 * The candidate (provider, model) pairs: every free model a probe listed,
 * with capabilities from the quality table where one exists, else from the
 * listing. The paid model is excluded by its free flag.
 */
function buildCandidates(
  brief: Brief,
  listings: Map<string, ListedModel[]>,
): Candidate[] {
  const ratingByKey = new Map<string, RatingRow>();
  for (const row of brief.ratings) ratingByKey.set(`${row.provider}\0${row.model}`, row);

  const candidates: Candidate[] = [];
  brief.providers.forEach((provider, providerIndex) => {
    const listed = listings.get(provider.id);
    if (listed === undefined) return;
    const seen = new Set<string>();
    for (const model of listed) {
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      const row = ratingByKey.get(`${provider.id}\0${model.id}`);
      const free = row !== undefined ? row.free : model.free;
      if (!free) continue;
      candidates.push({
        provider,
        providerIndex,
        model: model.id,
        ratingMilli: row !== undefined ? row.ratingMilli : 0,
        contextLength: row !== undefined ? row.contextLength : (model.contextLength ?? 0),
        tools: row !== undefined ? row.tools : (model.tools ?? false),
        vision: row !== undefined ? row.vision : (model.vision ?? false),
        json: row !== undefined ? row.json : (model.json ?? false),
      });
    }
  });
  return candidates;
}

/** Requirements a request imposes, from the manifest row and the body itself. */
function requirementsFor(manifestRow: Brief['manifest'][number] | undefined, canonical: Canonical): Requirements {
  const bodyTools = canonical.tools.length > 0;
  const bodyVision = canonical.messages.some((m) => m.parts.some((p) => p.kind === 'image'));
  return {
    tools: (manifestRow?.requires.tools ?? false) || bodyTools,
    vision: (manifestRow?.requires.vision ?? false) || bodyVision,
    json: (manifestRow?.requires.json ?? false) || canonical.jsonMode,
    minContext: manifestRow?.requires.minContext ?? 0,
  };
}

/**
 * Rank the candidates for one request. Providers that refused twice in a
 * row are demoted (their budget is likely spent or they are down) but stay
 * available as a last resort. Within a tier, the highest-rated model wins;
 * ties break on brief order, then model id. An OpenAI JSON-mode request
 * cannot use an Anthropic-wire provider: that wire cannot echo JSON mode.
 */
function rankCandidates(
  candidates: readonly Candidate[],
  states: Map<string, ProviderState>,
  requirements: Requirements,
  jsonModeFromOpenAI: boolean,
): Candidate[] {
  const eligible = candidates.filter((c) => {
    if (requirements.tools && !c.tools) return false;
    if (requirements.vision && !c.vision) return false;
    if (requirements.json && !c.json) return false;
    if (c.contextLength < requirements.minContext) return false;
    if (jsonModeFromOpenAI && c.provider.wire === 'anthropic') return false;
    return true;
  });
  const tierOf = (c: Candidate): number => {
    const state = states.get(c.provider.id);
    return state !== undefined && state.consecutiveRefusals >= 2 ? 1 : 0;
  };
  return [...eligible].sort((a, b) => {
    const tier = tierOf(a) - tierOf(b);
    if (tier !== 0) return tier;
    if (a.ratingMilli !== b.ratingMilli) return b.ratingMilli - a.ratingMilli;
    if (a.providerIndex !== b.providerIndex) return a.providerIndex - b.providerIndex;
    return a.model < b.model ? -1 : a.model > b.model ? 1 : 0;
  });
}

interface UpstreamResult {
  ok: boolean;
  answer: ProviderAnswer | null;
  status: number;
}

/** One upstream call. Any failure shape is a plain failure: the caller fails over. */
async function callProvider(
  candidate: Candidate,
  canonical: Canonical,
  envUrl: string,
): Promise<UpstreamResult> {
  const { provider } = candidate;
  const url = `${joinBase(envUrl, provider.path)}${upstreamPath(provider.wire, candidate.model)}`;
  const body = buildUpstreamBody(canonical, provider.wire, candidate.model);
  try {
    const res = await fetch(url, {
      method: 'POST',
      headers: { ...authHeaders(provider), ...upstreamHeaders(provider.wire) },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
    });
    if (!res.ok) return { ok: false, answer: null, status: res.status };
    const parsed: unknown = await res.json();
    const answer = extractAnswer(provider.wire, parsed);
    if (answer === null) return { ok: false, answer: null, status: res.status };
    return { ok: true, answer, status: res.status };
  } catch {
    return { ok: false, answer: null, status: 0 };
  }
}

function readBody(req: IncomingMessage): Promise<{ text: string; tooLarge: boolean }> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let total = 0;
    let tooLarge = false;
    req.on('data', (chunk: Buffer) => {
      total += chunk.length;
      if (total > MAX_BODY_BYTES) {
        tooLarge = true;
        return;
      }
      chunks.push(chunk);
    });
    req.on('end', () => resolve({ text: Buffer.concat(chunks).toString('utf8'), tooLarge }));
    req.on('error', reject);
  });
}

function sendJson(res: ServerResponse, status: number, body: Json): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendStream(res: ServerResponse, text: string): void {
  res.writeHead(200, {
    'content-type': 'text/event-stream',
    'cache-control': 'no-cache',
    'content-length': Buffer.byteLength(text),
  });
  res.end(text);
}

class Aggregator {
  private readonly brief: Brief;
  private readonly catalog: Catalog;
  private readonly candidates: Candidate[];
  private readonly states = new Map<string, ProviderState>();
  private readonly served: ServedModel[];
  private manifestIndex = 0;
  /** Serializes request handling: the client sends one request at a time, and
   *  serial handling keeps provider call order deterministic and avoids the
   *  one provider with a concurrency limit ever seeing two calls at once. */
  private queue: Promise<void> = Promise.resolve();

  constructor(brief: Brief, catalog: Catalog, listings: Map<string, ListedModel[]>) {
    this.brief = brief;
    this.catalog = catalog;
    this.candidates = buildCandidates(brief, listings);
    const firstProvider = new Map<string, string>();
    for (const provider of catalog.providers) {
      for (const model of provider.models) {
        if (model.free && !firstProvider.has(model.id)) firstProvider.set(model.id, provider.id);
      }
    }
    this.served = freeIdsOf(catalog).map((id) => ({ id, provider: firstProvider.get(id) ?? '' }));
    for (const provider of brief.providers) {
      this.states.set(provider.id, { calls: 0, consecutiveRefusals: 0 });
    }
  }

  get catalogDocument(): Catalog {
    return this.catalog;
  }

  handle(req: IncomingMessage, res: ServerResponse): void {
    const run = this.queue.then(() => this.dispatch(req, res));
    this.queue = run.then(
      () => undefined,
      () => undefined,
    );
    run.catch((error: unknown) => {
      console.error('handler error', error instanceof Error ? error.message : String(error));
      if (!res.writableEnded) {
        try {
          sendJson(res, 500, errorBody('openai', 500, 'internal error'));
        } catch {
          // The socket is already gone.
        }
      }
    });
  }

  private async dispatch(req: IncomingMessage, res: ServerResponse): Promise<void> {
    const url = new URL(req.url ?? '/', 'http://localhost');
    if (req.method === 'GET' && url.pathname === '/v1/models') {
      const anthropic = req.headers['x-api-key'] !== undefined;
      sendJson(res, 200, anthropic ? anthropicModelList(this.served) : openAIModelList(this.served));
      return;
    }
    if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
      await this.infer(req, res, 'openai');
      return;
    }
    if (req.method === 'POST' && url.pathname === '/v1/messages') {
      await this.infer(req, res, 'anthropic');
      return;
    }
    sendJson(res, 404, errorBody('openai', 404, 'not found'));
  }

  private async infer(req: IncomingMessage, res: ServerResponse, wire: ClientWire): Promise<void> {
    const { text, tooLarge } = await readBody(req);
    if (tooLarge) {
      sendJson(res, 413, errorBody(wire, 413, 'request body too large'));
      return;
    }
    let parsed: unknown;
    try {
      parsed = JSON.parse(text);
    } catch {
      sendJson(res, 400, errorBody(wire, 400, 'request body is not valid JSON'));
      return;
    }
    if (!isRecord(parsed)) {
      sendJson(res, 400, errorBody(wire, 400, 'request body must be a JSON object'));
      return;
    }

    const canonical = wire === 'openai' ? parseOpenAIRequest(parsed) : parseAnthropicRequest(parsed);
    const manifestRow = this.brief.manifest[this.manifestIndex];
    this.manifestIndex += 1;
    const requirements = requirementsFor(manifestRow, canonical);
    const jsonModeFromOpenAI = wire === 'openai' && canonical.jsonMode;
    const ranked = rankCandidates(this.candidates, this.states, requirements, jsonModeFromOpenAI);

    const envUrl = process.env.MAKEFUN_ENV_URL ?? '';
    let served: { answer: ProviderAnswer; model: string } | null = null;
    const tried = new Set<string>();
    for (const candidate of ranked) {
      const key = `${candidate.provider.id}\0${candidate.model}`;
      if (tried.has(key)) continue;
      tried.add(key);
      const state = this.states.get(candidate.provider.id) ?? { calls: 0, consecutiveRefusals: 0 };
      state.calls += 1;
      this.states.set(candidate.provider.id, state);
      const result = await callProvider(candidate, canonical, envUrl);
      if (result.ok && result.answer !== null) {
        state.consecutiveRefusals = 0;
        served = { answer: result.answer, model: candidate.model };
        break;
      }
      state.consecutiveRefusals += 1;
    }

    if (served === null) {
      sendJson(res, 503, errorBody(wire, 503, 'all providers refused the request'));
      return;
    }

    const accept = typeof req.headers.accept === 'string' ? req.headers.accept : '';
    const stream = parsed.stream === true || accept.includes('text/event-stream');
    if (wire === 'openai') {
      if (stream) sendStream(res, renderOpenAIStream(served.answer, served.model));
      else sendJson(res, 200, renderOpenAIJson(served.answer, served.model));
    } else {
      if (stream) sendStream(res, renderAnthropicStream(served.answer, served.model));
      else sendJson(res, 200, renderAnthropicJson(served.answer, served.model));
    }
  }
}

async function register(envUrl: string, port: number, catalog: Catalog): Promise<boolean> {
  for (let attempt = 0; attempt < 5; attempt += 1) {
    try {
      const res = await fetch(`${envUrl.replace(/\/$/, '')}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: 1, port, catalog }),
        signal: AbortSignal.timeout(5000),
      });
      if (res.ok) {
        const body: unknown = await res.json();
        if (isRecord(body) && body.ok === true) return true;
      }
    } catch {
      // Retry below.
    }
    await new Promise((resolve) => setTimeout(resolve, 200));
  }
  return false;
}

async function pollWork(envUrl: string): Promise<boolean> {
  try {
    const res = await fetch(`${envUrl.replace(/\/$/, '')}/work`, { signal: AbortSignal.timeout(5000) });
    if (!res.ok) return false;
    const body: unknown = await res.json();
    return isRecord(body) && body.done === true;
  } catch {
    return false;
  }
}

async function main(): Promise<void> {
  const listOnly = process.argv.includes('--list');
  const envUrl = process.env.MAKEFUN_ENV_URL ?? '';
  const brief = parseBrief(loadBriefText());

  let catalog: Catalog = { providers: [] };
  let listings = new Map<string, ListedModel[]>();
  if (envUrl !== '' && brief.providers.length > 0) {
    listings = await probeAll(brief.providers, envUrl);
    catalog = buildCatalog(brief, listings);
  }

  if (listOnly) {
    process.stdout.write(`${JSON.stringify(catalog)}\n`);
    process.exit(0);
  }

  if (envUrl === '') {
    console.error('MAKEFUN_ENV_URL is not set; refusing to serve outside a trial.');
    process.exit(1);
  }

  const aggregator = new Aggregator(brief, catalog, listings);
  const server = createServer((req, res) => aggregator.handle(req, res));

  let port = 0;
  for (let attempt = 0; attempt < 3; attempt += 1) {
    port = await new Promise<number>((resolve, reject) => {
      server.once('error', reject);
      server.listen(0, '0.0.0.0', () => {
        server.removeListener('error', reject);
        const address = server.address();
        resolve(typeof address === 'object' && address !== null ? address.port : 0);
      });
    });
    if (port !== 8080 && port !== 0) break;
    await new Promise<void>((resolve) => server.close(() => resolve()));
    port = 0;
  }
  if (port === 0 || port === 8080) {
    console.error('could not bind a usable port');
    process.exit(1);
  }

  const stop = (): void => {
    process.exit(0);
  };
  process.on('SIGTERM', stop);
  process.on('SIGINT', stop);

  const registered = await register(envUrl, port, aggregator.catalogDocument);
  if (!registered) {
    console.error('registration with the environment failed');
    process.exit(1);
  }

  let polling = false;
  const timer = setInterval(() => {
    if (polling) return;
    polling = true;
    void pollWork(envUrl)
      .then((done) => {
        if (done) {
          clearInterval(timer);
          process.exit(0);
        }
      })
      .finally(() => {
        polling = false;
      });
  }, POLL_INTERVAL_MS);
  timer.unref();
}

await main();
