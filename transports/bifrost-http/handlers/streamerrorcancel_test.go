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
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/valyala/fasthttp"
)

// An error a plugin's PostLLMHook puts in place of a stream chunk does not end the provider's
// stream: the provider goes on reading its upstream. An integration route ends the client's
// stream at that error, so it must also cancel the request. Otherwise the upstream keeps
// generating for nobody, and the stream's teardown, which waits for the provider before it
// runs the transport post-hooks and completes the trace, waits as long.

// errorOnSecondChunkPlugin lets the first stream chunk through and, when failSecondChunk is
// set, answers the second with an error. Its transport post-hook records how often it ran and
// whether its context was cancelled.
type errorOnSecondChunkPlugin struct {
	failSecondChunk bool

	mu            sync.Mutex
	postLLMCalls  int
	postHookCalls int
	postHookErr   error
	firstChunk    chan struct{}
}

func (p *errorOnSecondChunkPlugin) GetName() string { return "error-on-second-chunk" }
func (p *errorOnSecondChunkPlugin) Cleanup() error  { return nil }
func (p *errorOnSecondChunkPlugin) PreRequestHook(*schemas.BifrostContext, *schemas.BifrostRequest) error {
	return nil
}

func (p *errorOnSecondChunkPlugin) PreLLMHook(_ *schemas.BifrostContext, req *schemas.BifrostRequest) (*schemas.BifrostRequest, *schemas.LLMPluginShortCircuit, error) {
	return req, nil, nil
}

func (p *errorOnSecondChunkPlugin) PostLLMHook(_ *schemas.BifrostContext, resp *schemas.BifrostResponse, bifrostErr *schemas.BifrostError) (*schemas.BifrostResponse, *schemas.BifrostError, error) {
	p.mu.Lock()
	p.postLLMCalls++
	call := p.postLLMCalls
	p.mu.Unlock()
	switch {
	case call == 1:
		close(p.firstChunk)
	case call == 2 && p.failSecondChunk:
		return nil, &schemas.BifrostError{Error: &schemas.ErrorField{Message: "refused by plugin"}}, nil
	}
	return resp, bifrostErr, nil
}

func (p *errorOnSecondChunkPlugin) HTTPTransportPreAuthHook(*schemas.BifrostContext, *schemas.HTTPRequest) (*schemas.HTTPResponse, error) {
	return nil, nil
}

func (p *errorOnSecondChunkPlugin) HTTPTransportPreHook(*schemas.BifrostContext, *schemas.HTTPRequest) (*schemas.HTTPResponse, error) {
	return nil, nil
}

func (p *errorOnSecondChunkPlugin) HTTPTransportResponseHeadersHook(*schemas.BifrostContext, *schemas.HTTPRequest, *schemas.HTTPResponseMetadata) error {
	return nil
}

func (p *errorOnSecondChunkPlugin) HTTPTransportStreamChunkHook(_ *schemas.BifrostContext, _ *schemas.HTTPRequest, chunk *schemas.BifrostStreamChunk) (*schemas.BifrostStreamChunk, error) {
	return chunk, nil
}

func (p *errorOnSecondChunkPlugin) HTTPTransportPostHook(ctx *schemas.BifrostContext, _ *schemas.HTTPRequest, _ *schemas.HTTPResponse) error {
	p.mu.Lock()
	p.postHookCalls++
	p.postHookErr = ctx.Err()
	p.mu.Unlock()
	return nil
}

// streamErrorCancelAccount configures OpenAI and Anthropic against one local upstream.
type streamErrorCancelAccount struct{ baseURL string }

func (a streamErrorCancelAccount) GetConfiguredProviders() ([]schemas.ModelProvider, error) {
	return []schemas.ModelProvider{schemas.OpenAI, schemas.Anthropic}, nil
}

func (a streamErrorCancelAccount) GetKeysForProvider(context.Context, schemas.ModelProvider) ([]schemas.Key, error) {
	return []schemas.Key{{ID: "stream-error-cancel-key", Value: *schemas.NewSecretVar("sk-test"), Models: schemas.WhiteList{"*"}, Weight: 1.0}}, nil
}

func (a streamErrorCancelAccount) GetConfigForProvider(schemas.ModelProvider) (*schemas.ProviderConfig, error) {
	nc := schemas.DefaultNetworkConfig
	nc.BaseURL = a.baseURL
	return &schemas.ProviderConfig{NetworkConfig: nc, ConcurrencyAndBufferSize: schemas.DefaultConcurrencyAndBufferSize}, nil
}

// streamEndRoute is one served streaming route and the chunk its upstream sends.
type streamEndRoute struct {
	name, path, body, chunk string
	// endsAtErrorChunk: the route ends the client's stream at the first error chunk. The
	// /v1 routes do not; they forward what follows and leave out the [DONE] marker.
	endsAtErrorChunk bool
}

var streamEndRoutes = []streamEndRoute{
	{
		name: "v1 route", path: "/v1/chat/completions",
		body:  `{"model":"openai/gpt-4o","stream":true,"messages":[{"role":"user","content":"hello"}]}`,
		chunk: `data: {"id":"c1","object":"chat.completion.chunk","model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","content":"hi"}}]}` + "\n\n",
	},
	{
		name: "typed integration route", path: "/openai/v1/chat/completions", endsAtErrorChunk: true,
		body:  `{"model":"openai/gpt-4o","stream":true,"messages":[{"role":"user","content":"hello"}]}`,
		chunk: `data: {"id":"c1","object":"chat.completion.chunk","model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","content":"hi"}}]}` + "\n\n",
	},
	{
		name: "passthrough route", path: "/anthropic_passthrough/v1/messages", endsAtErrorChunk: true,
		body:  `{"model":"claude-test","stream":true,"messages":[{"role":"user","content":"hello"}]}`,
		chunk: "data: {\"n\":1}\n\n",
	},
}

// streamEndUpstream says what the upstream does after its second chunk.
type streamEndUpstream int

const (
	// upstreamEnds closes its stream normally.
	upstreamEnds streamEndUpstream = iota
	// upstreamStaysOpen sends nothing more and stays open until its request is cancelled.
	upstreamStaysOpen
	// upstreamKeepsSending writes chunk after chunk until its request is cancelled.
	upstreamKeepsSending
)

// streamEndCase is one served stream: a real Bifrost client behind the transport plugin
// middleware, a local upstream, and a plugin whose transport post-hook reports its context.
type streamEndCase struct {
	plugin            *errorOnSecondChunkPlugin
	resp              *http.Response
	torndown          chan struct{}
	upstreamCancelled chan struct{}
}

func startStreamEndCase(t *testing.T, route streamEndRoute, failSecondChunk bool, after streamEndUpstream) *streamEndCase {
	t.Helper()
	SetLogger(&mockLogger{})
	c := &streamEndCase{
		plugin:            &errorOnSecondChunkPlugin{failSecondChunk: failSecondChunk, firstChunk: make(chan struct{})},
		torndown:          make(chan struct{}),
		upstreamCancelled: make(chan struct{}),
	}
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, route.chunk)
		w.(http.Flusher).Flush()
		// The second chunk leaves only after the plugin has seen the first, so the two
		// are never read as one.
		select {
		case <-c.plugin.firstChunk:
		case <-r.Context().Done():
			close(c.upstreamCancelled)
			return
		}
		_, _ = io.WriteString(w, route.chunk)
		w.(http.Flusher).Flush()
		switch after {
		case upstreamEnds:
			return
		case upstreamKeepsSending:
			for r.Context().Err() == nil {
				_, _ = io.WriteString(w, route.chunk)
				w.(http.Flusher).Flush()
			}
		}
		<-r.Context().Done()
		close(c.upstreamCancelled)
	}))
	t.Cleanup(upstream.Close)

	client, err := bifrost.Init(context.Background(), schemas.BifrostConfig{
		Account:    streamErrorCancelAccount{baseURL: upstream.URL},
		LLMPlugins: []schemas.LLMPlugin{c.plugin},
		Logger:     bifrost.NewDefaultLogger(schemas.LogLevelError),
	})
	require.NoError(t, err)
	t.Cleanup(client.Shutdown)

	config := responseHeadersTestConfig(t, c.plugin)
	config.ClientConfig = &configstore.ClientConfig{}
	// The trace completer is the last step of the stream's teardown, after the transport
	// post-hooks.
	tracing := func(next fasthttp.RequestHandler) fasthttp.RequestHandler {
		return func(ctx *fasthttp.RequestCtx) {
			ctx.SetUserValue(schemas.BifrostContextKeyTraceCompleter, func([]schemas.PluginLogEntry) { close(c.torndown) })
			next(ctx)
		}
	}
	r := router.New()
	transport := TransportInterceptorMiddleware(config)
	r.POST("/v1/chat/completions", tracing(transport(NewInferenceHandler(nil, client, config).chatCompletion)))
	integrations.NewOpenAIRouter(client, config, nil, &mockLogger{}).RegisterRoutes(r, tracing, transport)
	integrations.NewAnthropicPassthroughRouter(client, config, nil, &mockLogger{}).RegisterRoutes(r, tracing, transport)
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	server := &fasthttp.Server{Handler: r.Handler}
	go server.Serve(ln)                  //nolint:errcheck
	t.Cleanup(func() { _ = ln.Close() }) //nolint:errcheck

	c.resp, err = http.Post("http://"+ln.Addr().String()+route.path, "application/json", strings.NewReader(route.body))
	require.NoError(t, err)
	t.Cleanup(func() { _ = c.resp.Body.Close() })
	require.Equal(t, http.StatusOK, c.resp.StatusCode)
	return c
}

// postHook waits for the stream's teardown and returns how often the transport post-hook ran
// and the error its context reported.
func (c *streamEndCase) postHook() (calls int, ctxErr error) {
	<-c.torndown
	c.plugin.mu.Lock()
	defer c.plugin.mu.Unlock()
	return c.plugin.postHookCalls, c.plugin.postHookErr
}

func TestIntegrationStreamCancelsUpstreamWhenAPluginErrorsAChunk(t *testing.T) {
	for _, route := range streamEndRoutes {
		if !route.endsAtErrorChunk {
			continue
		}
		t.Run(route.name, func(t *testing.T) {
			c := startStreamEndCase(t, route, true, upstreamStaysOpen)
			// The client's stream ends at the plugin's error although the upstream is still open.
			_, err := io.ReadAll(c.resp.Body)
			require.NoError(t, err)

			// Teardown finishes, and the upstream request is cancelled, without the
			// upstream ever ending its stream.
			calls, ctxErr := c.postHook()
			<-c.upstreamCancelled
			assert.Equal(t, 1, calls, "the transport post-hook runs once")
			assert.NoError(t, ctxErr, "the error was delivered: the post-hook's context is live although the upstream request was cancelled")
		})
	}
}

// The transport post-hook of a stream sees a cancelled context if and only if the request
// ended on its own: the client went away or a write to it failed. A cancel the transport
// issues itself, to stop the provider once it has nothing more to write, is not reported.
func TestStreamPostHookContextIsCancelledOnlyWhenTheRequestWas(t *testing.T) {
	for _, route := range streamEndRoutes {
		t.Run(route.name, func(t *testing.T) {
			t.Run("normal end", func(t *testing.T) {
				c := startStreamEndCase(t, route, false, upstreamEnds)
				_, err := io.ReadAll(c.resp.Body)
				require.NoError(t, err)
				calls, ctxErr := c.postHook()
				assert.Equal(t, 1, calls)
				assert.NoError(t, ctxErr, "a stream delivered to its end leaves the post-hook a live context")
			})

			t.Run("plugin error chunk delivered", func(t *testing.T) {
				// Routes that end at the error stop the upstream themselves; the /v1
				// routes forward on, so there the upstream ends its own stream.
				after := upstreamEnds
				if route.endsAtErrorChunk {
					after = upstreamStaysOpen
				}
				c := startStreamEndCase(t, route, true, after)
				_, err := io.ReadAll(c.resp.Body)
				require.NoError(t, err)
				calls, ctxErr := c.postHook()
				assert.Equal(t, 1, calls)
				assert.NoError(t, ctxErr, "an error that reached the client leaves the post-hook a live context")
			})

			t.Run("client disconnect", func(t *testing.T) {
				c := startStreamEndCase(t, route, false, upstreamKeepsSending)
				// The client leaves mid-stream, once the stream is under way.
				<-c.plugin.firstChunk
				require.NoError(t, c.resp.Body.Close())
				calls, ctxErr := c.postHook()
				<-c.upstreamCancelled
				assert.Equal(t, 1, calls)
				assert.ErrorIs(t, ctxErr, context.Canceled, "a client that went away is reported to the post-hook as a cancelled context")
			})
		})
	}
}
