package handlers

import (
	"context"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/fasthttp/router"
	bifrost "github.com/maximhq/bifrost/core"
	"github.com/maximhq/bifrost/core/schemas"
	"github.com/maximhq/bifrost/framework/configstore"
	"github.com/maximhq/bifrost/transports/v2/bifrost-http/integrations"
	"github.com/maximhq/bifrost/transports/v2/bifrost-http/lib"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/valyala/fasthttp"
)

// The transport pre-hook, the LLM hooks and the transport post-hook of one HTTP request
// share one BifrostContext: a value a plugin writes in one hook is readable in the hooks
// that run after it.

type sharedContextKey string

const (
	sharedContextFromTransportPre sharedContextKey = "from-transport-pre-hook"
	sharedContextFromPreLLM       sharedContextKey = "from-pre-llm-hook"
	sharedContextFromPostLLM      sharedContextKey = "from-post-llm-hook"
)

// sharedContextPlugin is one plugin that implements both the transport hooks and the LLM
// hooks. Each hook writes its own marker on the request context and records which of the
// earlier hooks' markers it could read.
type sharedContextPlugin struct {
	mu        sync.Mutex
	seen      map[string]map[sharedContextKey]string
	postCalls int
	postDone  chan struct{}
}

func newSharedContextPlugin() *sharedContextPlugin {
	return &sharedContextPlugin{seen: map[string]map[sharedContextKey]string{}, postDone: make(chan struct{})}
}

func (p *sharedContextPlugin) record(hook string, ctx *schemas.BifrostContext) {
	got := map[sharedContextKey]string{}
	for _, key := range []sharedContextKey{sharedContextFromTransportPre, sharedContextFromPreLLM, sharedContextFromPostLLM} {
		if v, ok := ctx.Value(key).(string); ok {
			got[key] = v
		}
	}
	p.mu.Lock()
	p.seen[hook] = got
	p.mu.Unlock()
}

func (p *sharedContextPlugin) seenBy(hook string) map[sharedContextKey]string {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.seen[hook]
}

func (p *sharedContextPlugin) GetName() string { return "shared-context" }
func (p *sharedContextPlugin) Cleanup() error  { return nil }

func (p *sharedContextPlugin) HTTPTransportPreAuthHook(*schemas.BifrostContext, *schemas.HTTPRequest) (*schemas.HTTPResponse, error) {
	return nil, nil
}

func (p *sharedContextPlugin) HTTPTransportPreHook(ctx *schemas.BifrostContext, _ *schemas.HTTPRequest) (*schemas.HTTPResponse, error) {
	ctx.SetValue(sharedContextFromTransportPre, "pre")
	return nil, nil
}

func (p *sharedContextPlugin) HTTPTransportResponseHeadersHook(*schemas.BifrostContext, *schemas.HTTPRequest, *schemas.HTTPResponseMetadata) error {
	return nil
}

func (p *sharedContextPlugin) HTTPTransportStreamChunkHook(_ *schemas.BifrostContext, _ *schemas.HTTPRequest, chunk *schemas.BifrostStreamChunk) (*schemas.BifrostStreamChunk, error) {
	return chunk, nil
}

func (p *sharedContextPlugin) HTTPTransportPostHook(ctx *schemas.BifrostContext, _ *schemas.HTTPRequest, _ *schemas.HTTPResponse) error {
	p.record("transport-post", ctx)
	p.mu.Lock()
	p.postCalls++
	first := p.postCalls == 1
	p.mu.Unlock()
	if first {
		close(p.postDone)
	}
	return nil
}

func (p *sharedContextPlugin) PreRequestHook(*schemas.BifrostContext, *schemas.BifrostRequest) error {
	return nil
}

func (p *sharedContextPlugin) PreLLMHook(ctx *schemas.BifrostContext, req *schemas.BifrostRequest) (*schemas.BifrostRequest, *schemas.LLMPluginShortCircuit, error) {
	p.record("pre-llm", ctx)
	ctx.SetValue(sharedContextFromPreLLM, "pre-llm")
	return req, nil, nil
}

func (p *sharedContextPlugin) PostLLMHook(ctx *schemas.BifrostContext, resp *schemas.BifrostResponse, bifrostErr *schemas.BifrostError) (*schemas.BifrostResponse, *schemas.BifrostError, error) {
	p.record("post-llm", ctx)
	ctx.SetValue(sharedContextFromPostLLM, "post-llm")
	return resp, bifrostErr, nil
}

// sharedContextAccount configures OpenAI against a local upstream.
type sharedContextAccount struct{ baseURL string }

func (a sharedContextAccount) GetConfiguredProviders() ([]schemas.ModelProvider, error) {
	return []schemas.ModelProvider{schemas.OpenAI}, nil
}

func (a sharedContextAccount) GetKeysForProvider(context.Context, schemas.ModelProvider) ([]schemas.Key, error) {
	return []schemas.Key{{ID: "shared-context-key", Value: *schemas.NewSecretVar("sk-test"), Models: schemas.WhiteList{"*"}, Weight: 1.0}}, nil
}

func (a sharedContextAccount) GetConfigForProvider(schemas.ModelProvider) (*schemas.ProviderConfig, error) {
	nc := schemas.DefaultNetworkConfig
	nc.BaseURL = a.baseURL
	return &schemas.ProviderConfig{NetworkConfig: nc, ConcurrencyAndBufferSize: schemas.DefaultConcurrencyAndBufferSize}, nil
}

// sharedContextUpstream answers chat completions, as JSON or as an SSE stream.
func sharedContextUpstream(t *testing.T) *httptest.Server {
	t.Helper()
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		if strings.Contains(string(body), `"stream":true`) {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = io.WriteString(w, `data: {"id":"c1","object":"chat.completion.chunk","model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","content":"hi"}}]}`+"\n\n")
			_, _ = io.WriteString(w, `data: {"id":"c1","object":"chat.completion.chunk","model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`+"\n\n")
			_, _ = io.WriteString(w, "data: [DONE]\n\n")
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"c1","object":"chat.completion","model":"gpt-4o","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
	}))
	t.Cleanup(upstream.Close)
	return upstream
}

// TestTransportInterceptorMiddleware_HooksShareOneContext drives a chat completion through
// the transport plugin middleware, the real inference handler and a real Bifrost client.
func TestTransportInterceptorMiddleware_HooksShareOneContext(t *testing.T) {
	SetLogger(&mockLogger{})
	for _, tc := range []struct {
		name string
		body string
	}{
		{"unary", `{"model":"openai/gpt-4o","messages":[{"role":"user","content":"hello"}]}`},
		{"streaming", `{"model":"openai/gpt-4o","stream":true,"messages":[{"role":"user","content":"hello"}]}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			upstream := sharedContextUpstream(t)
			plugin := newSharedContextPlugin()
			client, err := bifrost.Init(context.Background(), schemas.BifrostConfig{
				Account:    sharedContextAccount{baseURL: upstream.URL},
				LLMPlugins: []schemas.LLMPlugin{plugin},
				Logger:     bifrost.NewDefaultLogger(schemas.LogLevelError),
			})
			require.NoError(t, err)
			defer client.Shutdown()

			config := responseHeadersTestConfig(t, plugin)
			config.ClientConfig = &configstore.ClientConfig{}
			h := NewInferenceHandler(nil, client, config)
			ln, err := net.Listen("tcp", "127.0.0.1:0")
			require.NoError(t, err)
			server := &fasthttp.Server{Handler: TransportInterceptorMiddleware(config)(h.chatCompletion)}
			go server.Serve(ln)     //nolint:errcheck
			defer server.Shutdown() //nolint:errcheck

			resp, err := http.Post("http://"+ln.Addr().String()+"/v1/chat/completions", "application/json", strings.NewReader(tc.body))
			require.NoError(t, err)
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			require.NoError(t, err)
			require.Equal(t, http.StatusOK, resp.StatusCode, string(body))
			<-plugin.postDone

			assert.Equal(t, "pre", plugin.seenBy("pre-llm")[sharedContextFromTransportPre], "PreLLMHook reads what the transport pre-hook wrote")
			assert.Equal(t, "pre", plugin.seenBy("post-llm")[sharedContextFromTransportPre], "PostLLMHook reads what the transport pre-hook wrote")
			assert.Equal(t, "pre-llm", plugin.seenBy("post-llm")[sharedContextFromPreLLM], "PostLLMHook reads what PreLLMHook wrote")
			post := plugin.seenBy("transport-post")
			assert.Equal(t, "pre", post[sharedContextFromTransportPre], "the transport post-hook reads what the transport pre-hook wrote")
			assert.Equal(t, "pre-llm", post[sharedContextFromPreLLM], "the transport post-hook reads what PreLLMHook wrote")
			assert.Equal(t, "post-llm", post[sharedContextFromPostLLM], "the transport post-hook reads what PostLLMHook wrote")
		})
	}
}

// TestTransportInterceptorMiddleware_IntegrationStreamRunsPostHook streams a chat completion
// on an integration route: the transport post-hook runs once, after the stream, on the
// request's shared context.
func TestTransportInterceptorMiddleware_IntegrationStreamRunsPostHook(t *testing.T) {
	SetLogger(&mockLogger{})
	upstream := sharedContextUpstream(t)
	plugin := newSharedContextPlugin()
	client, err := bifrost.Init(context.Background(), schemas.BifrostConfig{
		Account:    sharedContextAccount{baseURL: upstream.URL},
		LLMPlugins: []schemas.LLMPlugin{plugin},
		Logger:     bifrost.NewDefaultLogger(schemas.LogLevelError),
	})
	require.NoError(t, err)
	defer client.Shutdown()

	config := responseHeadersTestConfig(t, plugin)
	config.ClientConfig = &configstore.ClientConfig{}
	// The trace completer is the last step of the stream's teardown, after the post-hooks.
	torndown := make(chan struct{})
	tracing := func(next fasthttp.RequestHandler) fasthttp.RequestHandler {
		return func(ctx *fasthttp.RequestCtx) {
			ctx.SetUserValue(schemas.BifrostContextKeyTraceCompleter, func([]schemas.PluginLogEntry) { close(torndown) })
			next(ctx)
		}
	}
	r := router.New()
	integrations.NewOpenAIRouter(client, config, nil, &mockLogger{}).RegisterRoutes(r, tracing, TransportInterceptorMiddleware(config))
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	server := &fasthttp.Server{Handler: r.Handler}
	go server.Serve(ln)     //nolint:errcheck
	defer server.Shutdown() //nolint:errcheck

	resp, err := http.Post("http://"+ln.Addr().String()+"/openai/v1/chat/completions", "application/json",
		strings.NewReader(`{"model":"openai/gpt-4o","stream":true,"messages":[{"role":"user","content":"hello"}]}`))
	require.NoError(t, err)
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	require.NoError(t, err)
	require.Equal(t, http.StatusOK, resp.StatusCode, string(body))
	require.Contains(t, string(body), "[DONE]")
	<-torndown

	plugin.mu.Lock()
	calls := plugin.postCalls
	plugin.mu.Unlock()
	require.Equal(t, 1, calls, "the transport post-hook runs exactly once for the stream")
	post := plugin.seenBy("transport-post")
	assert.Equal(t, "pre", post[sharedContextFromTransportPre])
	assert.Equal(t, "pre-llm", post[sharedContextFromPreLLM])
	assert.Equal(t, "post-llm", post[sharedContextFromPostLLM])
}

// TestTransportInterceptorMiddleware_HandlerAdoptsMiddlewareContext pins the mechanism: the
// handler's context is built on the one the middleware stored, and the post-hook runs on a
// context derived from the handler's, which reads through to it but is not it, so the
// transport's plugin logs never share a store with the logs core drains from the handler's.
func TestTransportInterceptorMiddleware_HandlerAdoptsMiddlewareContext(t *testing.T) {
	type key string
	var preHookCtx, handlerCtx, postHookCtx *schemas.BifrostContext
	var postHookSaw any
	plugin := &fakeResponseHeadersPlugin{
		name: "context-identity",
		hook: func(*schemas.BifrostContext, *schemas.HTTPRequest, *schemas.HTTPResponseMetadata) error { return nil },
		preHook: func(ctx *schemas.BifrostContext, _ *schemas.HTTPRequest) (*schemas.HTTPResponse, error) {
			preHookCtx = ctx.Root()
			return nil, nil
		},
		postHook: func(ctx *schemas.BifrostContext, _ *schemas.HTTPRequest, _ *schemas.HTTPResponse) error {
			postHookCtx = ctx.Root()
			postHookSaw = ctx.Value(key("written-by-handler"))
			ctx.Log(schemas.LogLevelInfo, "post-hook log")
			return nil
		},
	}
	handler := TransportInterceptorMiddleware(responseHeadersTestConfig(t, plugin))(func(ctx *fasthttp.RequestCtx) {
		stored, _ := ctx.UserValue(lib.FastHTTPUserValueBifrostContext).(*schemas.BifrostContext)
		require.Same(t, preHookCtx, stored, "the middleware stores the pre-hook's context for the handler")
		var cancel context.CancelFunc
		handlerCtx, cancel = lib.ConvertToBifrostContext(ctx, nil)
		defer cancel()
		handlerCtx.SetValue(key("written-by-handler"), "value")
	})
	ctx := preAuthTestCtx()
	handler(ctx)

	require.NotNil(t, handlerCtx)
	require.NotNil(t, postHookCtx)
	require.NotSame(t, handlerCtx, postHookCtx, "transport hooks must not log into the handler context's store")
	require.Equal(t, "value", postHookSaw, "the post-hook reads what was written on the handler's context")
	require.Empty(t, handlerCtx.GetPluginLogs(), "a transport hook's log never lands on the handler's context")
	logs, _ := ctx.UserValue(schemas.BifrostContextKeyTransportPluginLogs).([]schemas.PluginLogEntry)
	require.Len(t, logs, 1, "the transport hook's log is kept as a transport plugin log")
	require.Equal(t, "post-hook log", logs[0].Message)
}

// A unary request keeps its synchronous transport post-hook: exactly one call before the
// middleware returns. The marker that tells streaming handlers to allocate a completer
// slot never reaches a plugin as a path parameter.
func TestTransportInterceptorMiddleware_UnaryPostHookRunsOnceSynchronously(t *testing.T) {
	calls := 0
	var sawMarker bool
	plugin := &fakeResponseHeadersPlugin{
		name: "unary-post-hook",
		hook: func(*schemas.BifrostContext, *schemas.HTTPRequest, *schemas.HTTPResponseMetadata) error { return nil },
		preHook: func(_ *schemas.BifrostContext, req *schemas.HTTPRequest) (*schemas.HTTPResponse, error) {
			_, sawMarker = req.PathParams[string(schemas.BifrostContextKeyTransportPostHookCompleter)]
			return nil, nil
		},
		postHook: func(_ *schemas.BifrostContext, req *schemas.HTTPRequest, _ *schemas.HTTPResponse) error {
			calls++
			if _, ok := req.PathParams[string(schemas.BifrostContextKeyTransportPostHookCompleter)]; ok {
				sawMarker = true
			}
			return nil
		},
	}
	var markedForHandler bool
	handler := TransportInterceptorMiddleware(responseHeadersTestConfig(t, plugin))(func(ctx *fasthttp.RequestCtx) {
		markedForHandler, _ = ctx.UserValue(schemas.BifrostContextKeyTransportPostHookCompleter).(bool)
		ctx.SetStatusCode(fasthttp.StatusOK)
		ctx.SetBodyString(`{"ok":true}`)
	})
	ctx := preAuthTestCtx()
	ctx.Request.SetRequestURI("/anthropic/v1/messages")
	handler(ctx)
	if !markedForHandler {
		t.Fatal("the handler was not told that transport post-hooks will run")
	}
	if calls != 1 {
		t.Fatalf("unary post-hook calls = %d, want exactly 1 before the middleware returns", calls)
	}
	if sawMarker {
		t.Fatal("the completer-slot marker leaked into the request's path parameters")
	}
}
