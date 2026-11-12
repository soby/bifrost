# Fork notes

This branch is upstream `main` plus one commit. The commit exists for one use case: a plugin
looks up the calling user and routes each request to a provider and model chosen for that
request, for example to meet a compliance or data-residency requirement. Nothing about a
provider is configured ahead of time; the plugin supplies it per request.

Everything else in the commit is either a small hook such a plugin needs from the HTTP
transport or a bug fix found while running that way. Each item below is independent and is
intended to go upstream on its own.

## Per-request provider configuration

A plugin gives a request its own provider credential, base URL and private-network policy:

| API (`core/schemas`) | Use |
|---|---|
| `BifrostRequest.UpdateProviderKey(provider, key)` | Credential for every attempt against `provider`. It must be a ready credential with literal values: `env.` and `vault.` references, `oauth_key_config` and Anthropic workload identity are refused, and an endpoint the key carries must be a plain URL. |
| `BifrostRequest.UpdateProviderBaseURL(provider, url)` | Endpoint for those attempts. |
| `BifrostRequest.UpdateProviderAllowPrivateNetwork(provider, allow)` | Whether those attempts may reach private and loopback addresses. |
| `BifrostRequest.ProviderOverrideFor(provider)` | Read back what the request carries. |

Call them from `PreRequestHook` for the whole request or from `PreLLMHook` for the attempt
about to run, so two fallbacks to the same provider type can use different credentials.

- The provider does not need to be configured, and nothing is registered, queued or kept for
  it: an attempt runs on its own one-request worker and uses the request's key as a direct
  key. The caller's enqueue, wait, cancel and reject paths are the ones every request takes.
- An attempt the hooks did not configure resolves its provider as any other request does, so
  an unconfigured provider fails closed.
- Session affinity neither reroutes nor learns from a request that carries its own provider
  configuration.
- The attempt's key is on the context only while the attempt runs; each fallback starts
  from its own copy of the configuration the primary attempt ran with.
- Shutdown waits for admitted attempts until they return to their caller, so the hooks that
  run there finish before plugins are cleaned up. A request configured in `PreRequestHook`
  is admitted before its LLM hooks; a stream handed back to the caller is not waited for.
- Providers that need no process state implement `schemas.RequestScopedProvider`
  (`ForRequest`). Vertex takes part with a caller-supplied OAuth access token
  (`VertexKeyConfig.AccessToken`); Bedrock takes a base URL for model invocation only.
- Instances serving these attempts treat loopback as private, leave the response wait to the
  request context and the attempt's own timeout, and have no per-host connection cap.
- The fallbacks of a request follow the request the primary attempt's `PreLLMHook` returned,
  so a plugin that chooses the route there also chooses its fallbacks.
- An MCP agent loop's later turn is refused for a request that ran with request-scoped
  configuration: the continuation carries none, and is not run against a configured
  provider instead.
- `BatchCreateRequest` checks its provider after `PreRequestHook`, so a request configured
  there does not initialize a provider.

## Context keys and observers

| Name | Use |
|---|---|
| `BifrostContextKeyAttemptRequestTimeout` | Bounds each provider round trip of the current attempt; expiry is a retryable timeout, so fallbacks run. Honored by the fasthttp and Bedrock net/http clients. |
| `BifrostContextKeyStreamFrameTap` | A function that receives the exact bytes of each stream frame sent to the client, for example for audit logging. |
| `BifrostContextKeyStreamAdmissionWait` | Lets a plugin that holds a streaming request before the provider call (for example while it waits on an external lookup) ask the transport to commit the event stream early; the stream heartbeat then keeps the connection alive. |
| `schemas.SetConnectionObserver` | Reports provider connections opened, reused, awaited and closed. |
| `schemas.GetUpstreamWindow` | First hand-off to a provider transport and last provider wait of a request, with the provider-only total. |

## HTTP transport

- The transport pre-hook, the LLM hooks and the transport post-hook of one request share one
  context.
- The transport post-hook runs once after a stream ends on integration routes as well as
  `/v1` routes, after the provider stream has finished. The stream hands it over by signal,
  so it is never skipped and a stream without transport plugins does not wait for it.
- After the handler returns, transport hooks run on a context derived from the request's:
  they read every value the earlier hooks set and see the same cancellation, and keep their
  own plugin logs.
- `/v1` chat streams follow the OpenAI `stream_options.include_usage` contract.
- Without a config store: model capabilities and model parameters are served from the
  in-memory catalog, and `/api` keeps its setup lock.
- The access log records the exported W3C trace ID.
- Inbound read timeout, idle timeout and idle-buffer release are configurable; unset means
  upstream's behavior.
- A compressed request body waits for the decompression budget instead of being refused.

## Request and response fidelity fixes

Each of these stops a silent rewrite or drop of what the caller or provider sent:

- a scalar `stop` on chat completion requests is accepted;
- `reasoning_details` survives on OpenAI-compatible chat requests;
- numbers in extra params keep their exact form;
- extra-param passthrough applies only to the attempt that enabled it;
- OpenAI-compatible conversions do not rewrite caller fields (including the `max_tokens`
  spelling);
- unknown JSON Schema keywords in tool parameters are kept;
- chat audio voice objects and assistant audio replay are carried;
- the compat plugin's reasoning-with-tools Responses reroute is scoped to the models it is for;
- a choice-level error on a 200 response is a failure, so fallbacks run;
- OpenRouter provider routing preferences are forwarded;
- a stale-connection retry keeps the original request deadline;
- Gemini and Cohere map `seed` and logprobs on their chat, Responses and native paths, and
  logprobs survive Chat/Responses conversions;
- an Azure deployment path is decoded before it is parsed;
- a Cursor stream keeps its terminal response ID on a final chunk without usage;
- a Bedrock invoke reply with no content renders as `content: []`, and string content is kept;
- parallel tool turns stay whole for Gemini Responses and Bedrock chat;
- a chat response without usage carries no `usage` key.
