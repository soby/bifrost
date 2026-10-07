# soby/bifrost fork

This fork carries the changes Pretxt's gateway (`api-llm-gateway`) runs on top of upstream
[maximhq/bifrost](https://github.com/maximhq/bifrost). Runtime branches are named
`runtime/*` and are never squashed, amended or force-pushed; a refresh is a new branch.

Current runtime branch: `runtime/request-scoped`.

- Base: upstream PR #2030 (`feat/provider-override`, 63344b104), which is upstream `dev`
  1bd65ae82 plus the request-scoped provider redesign.
- Previous runtime branch: `runtime/provider-override-plus-5277` (825e40e35). It is kept
  unchanged and is not an ancestor of this branch.

## Upstream PRs in this branch

| Upstream PR | What | Status | In this branch |
|---|---|---|---|
| #2030 | Request-scoped provider configuration: `BifrostRequest.UpdateProviderKey`, `UpdateProviderBaseURL`, `UpdateProviderAllowPrivateNetwork`, `ProviderOverrideFor`; `schemas.RequestScopedProvider`; inline execution bounded by a per-instance semaphore | Open (ours) | Base of the branch |
| #6630 | Honor `UseRawRequestBody` on native inference routes | Open (third party) | Carried |
| #7227 | Accept a scalar `stop` on chat completions | Open (third party) | Carried |
| #5277 | Preserve `reasoning_details` on OpenAI-compatible chat requests | Open (third party) | Carried, adapted: forwarded only to destinations without a curated OpenAI dialect (OpenRouter, custom providers); curated dialects keep it off the wire; Anthropic conversion drops unsigned text-only details |
| #6452 | Request-scoped routing and provider deadlines | Open (ours) | Its deadline and OpenRouter routing parts are carried as the fork commits below |

## Fork-only commits

Carried from `fork/fidelity-on-dev` (authored on the fork, not filed upstream):

| Commit | Change |
|---|---|
| 9cd2fecf7 | Exact numbers in extra params (integers above 2^53 and decimal literals kept as written); every typed chat parameter is a known field |
| cfb3ae58d | Extra-param passthrough is scoped to the attempt that enabled it; `x-bf-passthrough-extra-params` parsed as a boolean |
| d5eb1b894 | No caller-field rewrites on OpenAI-compatible conversions: 16-token floor only for OpenAI/Azure, `max_tokens` spelling kept for uncurated destinations, custom providers keep their custom identity |
| 8fd184eb6 | Unknown JSON Schema keywords in tool parameters are preserved; non-string `enum` kept verbatim |
| a1629fd4d | Chat audio voice objects and assistant audio replay references are carried |
| 510934489 | The compat reasoning-with-tools Responses reroute only fires for OpenAI/Azure requests that have tools and ask for reasoning, without a raw body |
| 21665fac9 | Inbound fasthttp server releases idle connection buffers (`reduce_memory_usage`, default true) and closes connections idle for `idle_timeout_seconds` (default 620) |
| 8c48cd8b7 | Model catalog provider indexes are read without copying |
| d7a812149 | Choice-level in-band errors on a 200 (`choices[i].error`, `finish_reason: "error"`) are failures, so fallbacks run |

Ported from the previous runtime branch:

| Commit | Previous | Change |
|---|---|---|
| 4a569be54, 1d3a4d95b | 2f68ec4d9, 0e9bd51ea | Stale pooled-connection retries share the caller's original deadline (PLATFORM-2422) |
| 3cfd1ecf8 | d414d9175 | OpenRouter's top-level `provider` routing object reaches OpenRouter (only OpenRouter) |
| 4f8f6320b | 14f5ade79 | `pricing.automatic_sync_enabled: false` starts the model catalog from stored or empty data without remote fetches |
| f7388d5b1 | 61a53ac78 | Access logs carry the exported W3C trace ID (PLATFORM-3033) |
| 56b9578b1 | 02c90d302 | `schemas.UpstreamWindow` records the first provider handoff, last socket wait end and provider-only wait total (PLATFORM-3980) |
| a698c1dc2 | 9536040aa | `BIFROST_ACCESS_LOG_QUIET_PATHS` skips access logs for successful requests on listed path prefixes (PLATFORM-4014) |
| 2e548b16a | part of 503e90ef9 | Chat streams follow `stream_options.include_usage`: usage omitted unless requested, otherwise sent as its own final chunk; `BifrostChatResponse.Usage` is `omitempty` |

New on this branch:

| Commit | Change |
|---|---|
| dff2db37c | `BatchCreateRequest`'s provider check runs after `PreRequestHook` and is skipped for a request carrying request-scoped configuration, so a request-scoped batch create initializes no provider |
| 1ecdac20d | Request-scoped Vertex with a caller-supplied OAuth access token (`VertexKeyConfig.AccessToken`) |
| 012bd19ab | Per-attempt request timeout from the context (`BifrostContextKeyAttemptRequestTimeout`); expiry is a retryable timeout, so fallbacks run |
| c64050487 | Fallbacks follow the request the primary attempt's `PreLLMHook` returned (fallback list, fallback decision and base request), as on the previous runtime branch |
| 4d0091cca | The transport interceptor middleware, the request handler, the LLM hooks and the HTTP transport post-hook share one `BifrostContext` per request (`lib.EnsureSharedBifrostContext`), as `cbcdf67b8` did on the previous runtime branch |

Dropped from the previous runtime branch: the earlier `ProviderOverride` / provider auto-init
implementation (503e90ef9, a42f4fb7a, 047ae6692, a9e7eb0c5, 16d994bca; replaced by #2030's
redesign), 76263338e (superseded by the adapted #5277 commit), the reasoning_details test
afd5a305e (superseded by the adapted #5277 tests), and the x/crypto upgrade and its revert
(cbe9dea30, 5dafa1fa3).

## What the gateway relies on

### Request-scoped provider configuration

- Set the credential, endpoint and private-network policy with
  `req.UpdateProviderKey(provider, key)`, `req.UpdateProviderBaseURL(provider, url)` and
  `req.UpdateProviderAllowPrivateNetwork(provider, allow)`; read them back with
  `req.ProviderOverrideFor(provider)`. Route with `SetProvider` / `SetModel`.
- Set them in `PreRequestHook`. A request that leaves `PreRequestHook` with no request-scoped
  configuration resolves its provider before `PreLLMHook`, so an unconfigured provider fails
  there; `PreLLMHook` can only change the configuration of a request that already carries some.
- Fallbacks come from the request the primary attempt ran with (fork-only): a `PreLLMHook`
  that returns a replacement request on the primary attempt sets the fallbacks, and every
  fallback attempt starts from a copy of that request.
- Key fields must be literal values; `env.` and `vault.` references are rejected. A key without
  an ID is reported as `request-scoped`.
- Request-scoped attempts run on an unregistered instance built with default network settings
  (`DefaultNetworkConfig`: 300 s request timeout, 0 retries). There is no per-request
  retry count or backoff; bound a request with its context deadline and an attempt with
  `BifrostContextKeyAttemptRequestTimeout` (below). Loopback and
  RFC 1918 destinations need `UpdateProviderAllowPrivateNetwork`; link-local is always refused.
- WebSocket and realtime routes do not honor request-scoped configuration.
- Supported: OpenAI, Anthropic, Gemini, Cohere, Cerebras, Groq, Mistral, Nebius, OpenRouter,
  Parasail, Perplexity, xAI, Replicate, Hugging Face, Ollama, SGL, vLLM (URL on the key),
  Azure (API key and endpoint on the key), Bedrock (API key or static keys, region on the key)
  and, fork-only, Vertex. Others fail closed.

### Vertex with a caller-supplied token (fork-only)

- Put a ready OAuth access token with the `cloud-platform` scope in
  `Key.VertexKeyConfig.AccessToken`, with `ProjectID` and `Region` on the same
  `VertexKeyConfig` (`ProjectNumber` for fine-tuned endpoints). The caller mints and refreshes
  the token; Bifrost never caches it or builds a token source.
- A request-scoped Vertex key must not carry `Key.Value`, `AuthCredentials` or
  `AWSWorkloadIdentity`, and the token must be non-empty with no whitespace or control
  characters. Application Default Credentials and AWS federation are never used.
- `AccessToken` is `json:"-"`: it is not serialized, persisted or returned by the API.

### Per-request context keys

- `schemas.BifrostContextKeyExtraHeaders` (`map[string][]string`): extra provider request
  headers. They are set after the provider's configured extra headers and replace a header of
  the same name, including a version header the adapter set before them; hop-by-hop headers are
  filtered. The key is not cleared between attempts: a plugin that varies it per provider sets
  the whole map in `PreLLMHook` for every attempt.
- `schemas.BifrostContextKeyAttemptRequestTimeout` (`time.Duration`, fork-only): bounds each
  provider HTTP round trip of the current attempt, the whole call for a unary request and the
  wait for response headers for a stream, like `default_request_timeout_in_seconds` for a
  configured provider. Expiry closes the socket and fails the attempt with a retryable timeout
  (504 `request_timed_out`), so fallbacks run; cancelling the request stays a cancellation.
  Cleared before each fallback, so set it in `PreLLMHook` for every attempt that needs it.
  Bedrock's net/http requests are not bounded by it.
- `schemas.BifrostContextKeyStreamIdleTimeout` (`time.Duration`): per-chunk idle timeout for
  streams. A positive value set before the attempt wins over the provider's configured
  `stream_idle_timeout_in_seconds` (default 120 s).
- `schemas.GetUpstreamWindow(ctx)`: the request's `UpstreamWindow` (fork-only).

### Transport

- One `BifrostContext` per request (fork-only): a value a plugin writes during the request is
  visible to its `HTTPTransportPostHook`, and the handler's cancellation of that context is
  visible there too. Upstream builds a separate context in the transport middleware.

- `BIFROST_ACCESS_LOG_QUIET_PATHS`: comma-separated path prefixes whose successful requests
  are not access-logged (fork-only).
- Access-log `trace_id` is the exported W3C trace ID (fork-only).
- Chat streams follow `stream_options.include_usage` (fork-only).

## Known non-preserved fields

These are accepted on input but not forwarded to providers. The fork documents them and does
not compensate:

- Explicit `null` values in request fields are dropped; a field sent as `null` is treated as
  absent.
- The legacy assistant `function_call` (pre-tools OpenAI format) on replayed assistant
  messages is dropped; use `tool_calls`.
- An empty assistant `reasoning` / `reasoning_content` string is treated as absent, and empty
  `reasoning_details` entries are pruned (upstream #7294).
