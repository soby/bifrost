package integrations

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/fasthttp/router"
	bifrost "github.com/maximhq/bifrost/core"
	"github.com/maximhq/bifrost/core/schemas"
	"github.com/maximhq/bifrost/transports/v2/bifrost-http/lib"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/valyala/fasthttp"
)

// Streams on integration routes run the transport post-hooks exactly once after the stream
// ends, as streams on the /v1 routes do. The tests reproduce TransportInterceptorMiddleware's
// part: mark that post-hooks will run, call the handler, then store the completer in the slot
// the handler allocated.

// streamPostHookTestEvents is a minimal native Responses stream: two text
// deltas and the completed event.
func streamPostHookTestEvents() []*schemas.BifrostStreamChunk {
	completed := schemas.ResponsesStreamResponseTypeCompleted
	delta := schemas.ResponsesStreamResponseTypeOutputTextDelta
	return []*schemas.BifrostStreamChunk{
		{BifrostResponsesStreamResponse: &schemas.BifrostResponsesStreamResponse{Type: delta, SequenceNumber: 0, Delta: schemas.Ptr("first ")}},
		{BifrostResponsesStreamResponse: &schemas.BifrostResponsesStreamResponse{Type: delta, SequenceNumber: 1, Delta: schemas.Ptr("second")}},
		{BifrostResponsesStreamResponse: &schemas.BifrostResponsesStreamResponse{Type: completed, SequenceNumber: 2, Response: &schemas.BifrostResponsesResponse{
			ID: schemas.Ptr("resp_1"), Model: "m", Status: schemas.Ptr("completed"),
		}}},
	}
}

func streamPostHookRoute(t *testing.T, routes []RouteConfig, pathFragment string) RouteConfig {
	t.Helper()
	for _, route := range routes {
		if route.StreamConfig != nil && route.StreamConfig.ResponsesStreamResponseConverter != nil && strings.Contains(route.Path, pathFragment) {
			return route
		}
	}
	t.Fatalf("no responses streaming route containing %q", pathFragment)
	return RouteConfig{}
}

func TestHandleStreamingRunsTransportPostHookOnceAfterTheLastEvent(t *testing.T) {
	for _, test := range []struct {
		name   string
		routes []RouteConfig
		path   string
	}{
		{name: "anthropic", routes: CreateAnthropicRouteConfigs("/anthropic", bifrost.NewNoOpLogger()), path: "/messages"},
		{name: "genai", routes: CreateGenAIRouteConfigs("/genai"), path: "models"},
	} {
		t.Run(test.name, func(t *testing.T) {
			route := streamPostHookRoute(t, test.routes, test.path)
			events := streamPostHookTestEvents()
			var converted atomic.Int32
			convert := route.StreamConfig.ResponsesStreamResponseConverter
			streamConfig := *route.StreamConfig
			streamConfig.ResponsesStreamResponseConverter = func(ctx *schemas.BifrostContext, resp *schemas.BifrostResponsesStreamResponse) (string, interface{}, error) {
				converted.Add(1)
				return convert(ctx, resp)
			}
			route.StreamConfig = &streamConfig

			stream := make(chan *schemas.BifrostStreamChunk, len(events))
			for _, event := range events {
				stream <- event
			}

			ctx := &fasthttp.RequestCtx{}
			ctx.Request.Header.SetMethod(fasthttp.MethodPost)
			ctx.Request.SetRequestURI(route.Path)
			ctx.SetUserValue(schemas.BifrostContextKeyTransportPostHookCompleter, true)
			// The trace completer runs after the post-hook completer in the producer's
			// teardown and is the teardown's last step that calls out, so its call marks
			// the point after which no further post-hook call can happen.
			traced := make(chan []schemas.PluginLogEntry, 1)
			ctx.SetUserValue(schemas.BifrostContextKeyTraceCompleter, func(logs []schemas.PluginLogEntry) { traced <- logs })

			router := NewGenericRouter(nil, &mockHandlerStore{}, nil, nil, nil, bifrost.NewNoOpLogger())
			router.handleStreaming(ctx, schemas.NewBifrostContext(context.Background(), schemas.NoDeadline), route, schemas.ResponsesStreamRequest, stream, func() {})

			slot, ok := ctx.UserValue(schemas.BifrostContextKeyTransportPostHookCompleter).(*lib.TransportPostHookSlot)
			require.True(t, ok, "a stream marked for post-hooks must allocate the completer slot")
			wantLogs := []schemas.PluginLogEntry{{PluginName: "test-transport", Level: schemas.LogLevelInfo, Message: "post-hook ran"}}
			var calls, convertedAtPostHook atomic.Int32
			slot.Resolve(func() ([]schemas.PluginLogEntry, error) {
				calls.Add(1)
				convertedAtPostHook.Store(converted.Load())
				return wantLogs, nil
			})
			// The stream ends only now, after the completer is published, as it is in a
			// served request whose middleware returns before the provider finishes.
			close(stream)

			body, err := io.ReadAll(ctx.Response.BodyStream())
			require.NoError(t, err)
			assert.Contains(t, string(body), "second", "the client receives the last event")
			gotLogs := <-traced
			assert.Equal(t, int32(1), calls.Load(), "the transport post-hook runs exactly once")
			assert.Equal(t, int32(len(events)), convertedAtPostHook.Load(), "post-hooks run only after every event was converted")
			assert.Equal(t, wantLogs, gotLogs, "the post-hooks' logs reach the trace completer")
		})
	}
}

// A request that did not pass through the transport plugin middleware (no plugins loaded)
// carries no marker: the stream allocates no slot and its teardown waits for no completer.
func TestHandleStreamingWithoutTransportPluginsAllocatesNoCompleterSlot(t *testing.T) {
	route := streamPostHookRoute(t, CreateAnthropicRouteConfigs("/anthropic", bifrost.NewNoOpLogger()), "/messages")
	stream := make(chan *schemas.BifrostStreamChunk)
	close(stream)

	ctx := &fasthttp.RequestCtx{}
	ctx.Request.Header.SetMethod(fasthttp.MethodPost)
	ctx.Request.SetRequestURI(route.Path)
	traced := make(chan []schemas.PluginLogEntry, 1)
	ctx.SetUserValue(schemas.BifrostContextKeyTraceCompleter, func(logs []schemas.PluginLogEntry) { traced <- logs })

	router := NewGenericRouter(nil, &mockHandlerStore{}, nil, nil, nil, bifrost.NewNoOpLogger())
	router.handleStreaming(ctx, schemas.NewBifrostContext(context.Background(), schemas.NoDeadline), route, schemas.ResponsesStreamRequest, stream, func() {})

	assert.Nil(t, ctx.UserValue(schemas.BifrostContextKeyTransportPostHookCompleter), "no slot without the middleware's marker")
	_, err := io.ReadAll(ctx.Response.BodyStream())
	require.NoError(t, err)
	assert.Nil(t, <-traced)
}

// A streamed passthrough request runs the transport post-hooks exactly once after the
// upstream's last byte was relayed. Driven through the registered route with a real Bifrost
// client and a local SSE upstream.
func TestHandlePassthroughStreamRunsTransportPostHookOnce(t *testing.T) {
	// The upstream holds its last event until the middleware has published the completer,
	// as a served stream outlives the middleware's return.
	published := make(chan struct{})
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: one\n\n")
		w.(http.Flusher).Flush()
		<-published
		_, _ = io.WriteString(w, "data: two\n\n")
	}))
	defer upstream.Close()

	client, err := bifrost.Init(context.Background(), schemas.BifrostConfig{
		Account: passthroughTestAccount{baseURL: upstream.URL},
		Logger:  bifrost.NewDefaultLogger(schemas.LogLevelError),
	})
	require.NoError(t, err)
	defer client.Shutdown()

	wantLogs := []schemas.PluginLogEntry{{PluginName: "test-transport", Level: schemas.LogLevelInfo, Message: "post-hook ran"}}
	var calls atomic.Int32
	traced := make(chan []schemas.PluginLogEntry, 1)
	// transportMiddleware does what TransportInterceptorMiddleware does around a handler.
	transportMiddleware := func(next fasthttp.RequestHandler) fasthttp.RequestHandler {
		return func(ctx *fasthttp.RequestCtx) {
			ctx.SetUserValue(schemas.BifrostContextKeyTraceCompleter, func(logs []schemas.PluginLogEntry) { traced <- logs })
			ctx.SetUserValue(schemas.BifrostContextKeyTransportPostHookCompleter, true)
			next(ctx)
			defer close(published)
			slot, ok := ctx.UserValue(schemas.BifrostContextKeyTransportPostHookCompleter).(*lib.TransportPostHookSlot)
			if !ok {
				return
			}
			slot.Resolve(func() ([]schemas.PluginLogEntry, error) {
				calls.Add(1)
				return wantLogs, nil
			})
		}
	}
	r := router.New()
	NewAnthropicPassthroughRouter(client, passthroughTestStore(), nil, &testLogger{}).RegisterRoutes(r, transportMiddleware)

	var ctx fasthttp.RequestCtx
	ctx.Request.Header.SetMethod(fasthttp.MethodGet)
	ctx.Request.SetRequestURI("/anthropic_passthrough/v1/files/file_1?stream=true")
	ctx.Request.Header.Set("x-model-provider", "anthropic")
	r.Handler(&ctx)

	require.Equal(t, fasthttp.StatusOK, ctx.Response.StatusCode())
	require.NotNil(t, ctx.Response.BodyStream(), "the request is streamed")
	body, err := io.ReadAll(ctx.Response.BodyStream())
	require.NoError(t, err)
	assert.Contains(t, string(body), "data: two", "the client receives the upstream's last event")
	assert.Equal(t, wantLogs, <-traced, "the post-hooks' logs reach the trace completer")
	assert.Equal(t, int32(1), calls.Load(), "the transport post-hook runs exactly once")
}
