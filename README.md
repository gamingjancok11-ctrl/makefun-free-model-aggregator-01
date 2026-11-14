# Free AI model aggregator

One server that serves the OpenAI Chat Completions API and the Anthropic Messages
API over ten simulated free-tier providers, routing each request to the best free
model that can serve it, failing over when a provider refuses, and reporting the
catalog it read. An entry for make.fun's `free-model-aggregator-01` bounty.

Node 22, standard library only; nothing is installed before the server runs.

## Layout

- `main.ts` — the server: brief + probe + catalog, the router, failover, the two
  client APIs, register and the `/work` poll.
- `lib/brief.ts` — the trial brief's three JSON documents (provider directory,
  quality table, workload manifest), found by schema.
- `lib/catalog.ts` — the catalog probe: each provider's model list in its own wire,
  and the one catalog document `--list` prints and the register body carries.
- `lib/wire.ts` — the three wires (OpenAI, Anthropic, Gemini): requests normalised
  out of the client's wire and rendered into the provider's; answers parsed out of
  the provider's wire and rendered into the client's, in JSON and in server-sent
  events.

## Run

```sh
node main.ts --list   # print the catalog document and exit 0
node main.ts          # serve a trial (needs MAKEFUN_ENV_URL + MAKEFUN_PROMPT_FILE)
```

## How it routes

Every request is served by the highest-rated free model that meets its
requirements (tools, vision, JSON mode, minimum context), ranked from the brief's
quality table. The ranking charges a small penalty per call already made to a
provider, which spreads the load across the near-best providers instead of
draining the top one: free-tier budgets are drawn per trial and never published,
so a greedy router would burn the best provider early and fail over late. A
provider that fails twice in a row is deprioritised heavily but kept as a last
resort.

One wire constraint shapes the candidates: a JSON-mode request from an OpenAI
client goes only to an OpenAI- or Gemini-wire provider, because the Anthropic
wire cannot carry JSON mode. A paid model is never a legal pick, whatever its
rating.

## How it fails over

A provider refuses with 402 when its free allowance is gone (one provider) or
429 on the other nine, sometimes with a retry-after. A refusal is never the
answer: the top candidate is retried once after its retry-after (a scheduled
refusal spends that call number, so the next one usually answers) — but only
when it has never refused before, so the retry is never wasted on a provider
that already looks spent — then the request fails over to the next candidate
until one answers. A provider that refused twice is deprioritised heavily, so
later requests stop knocking on a spent provider's door. A 401 or 400 is not
retried on the same provider. Exactly one provider call answers each request,
and upstream calls are serialised, so the provider that serves one call at a
time never sees a concurrent second call.

## How it keeps the wires faithful

Every field the provider echoes — system prompt, user text, images, tool
definitions, max_tokens, stop sequences, JSON mode — passes through untouched,
translated between the three wires. Streaming answers are buffered, parsed and
re-emitted in the client's own event order (OpenAI chunks ending in `[DONE]`,
Anthropic `message_start` … `message_stop`), with the provider's usage counts
and stop reason mapped into the client's wire. Upstream OpenAI streaming calls
set `stream_options.include_usage`, without which the stream carries no usage.

Everything is deterministic given the brief, the requests and the providers'
answers: fixed ids and timestamps, no wall clock, no unseeded randomness, so a
replay leaves the same state export.
