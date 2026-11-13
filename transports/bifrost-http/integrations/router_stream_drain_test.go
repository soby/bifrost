package integrations

import (
	"context"
	"io"
	"sync"
	"sync/atomic"
	"testing"

	bifrost "github.com/maximhq/bifrost/core"
	"github.com/maximhq/bifrost/core/providers/bedrock"
	"github.com/maximhq/bifrost/core/schemas"
	"github.com/maximhq/bifrost/transports/v2/bifrost-http/lib"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/valyala/fasthttp"
)

// An integration stream that stops before the provider's stream has ended (an error chunk, a
// failed client write) waits for the provider goroutine before it runs the transport
// post-hooks and completes the trace, as the streams on the /v1 routes do. The provider
// goroutine runs its last post-hooks on the request context as it winds down; the transport
// post-hooks must not run beside it.

// drainProbe stands in for the provider goroutine and for the middleware's completer. finished
// is set by the provider just before it closes its stream; the completer and the trace
// completer record whether they saw it set.
type drainProbe struct {
	stream               chan *schemas.BifrostStreamChunk
	finished             atomic.Bool
	finishedAtPostHook   atomic.Bool
	finishedAtTraceClose atomic.Bool
	postHookCalls        atomic.Int32
	traced               chan struct{}
}

// start runs handleStreaming for route as TransportInterceptorMiddleware would around it.
func (p *drainProbe) start(t *testing.T, route RouteConfig, requestType schemas.RequestType, cancel context.CancelFunc) *fasthttp.RequestCtx {
	t.Helper()
	p.stream = make(chan *schemas.BifrostStreamChunk)
	p.traced = make(chan struct{})
	ctx := &fasthttp.RequestCtx{}
	ctx.Request.Header.SetMethod(fasthttp.MethodPost)
	ctx.SetUserValue(schemas.BifrostContextKeyTransportPostHookCompleter, true)
	ctx.SetUserValue(schemas.BifrostContextKeyTraceCompleter, func([]schemas.PluginLogEntry) {
		p.finishedAtTraceClose.Store(p.finished.Load())
		close(p.traced)
	})
	router := NewGenericRouter(nil, &mockHandlerStore{}, nil, nil, nil, bifrost.NewNoOpLogger())
	router.handleStreaming(ctx, schemas.NewBifrostContext(context.Background(), schemas.NoDeadline), route, requestType, p.stream, cancel)
	slot, ok := ctx.UserValue(schemas.BifrostContextKeyTransportPostHookCompleter).(*lib.TransportPostHookSlot)
	require.True(t, ok)
	slot.Resolve(func() ([]schemas.PluginLogEntry, error) {
		p.postHookCalls.Add(1)
		p.finishedAtPostHook.Store(p.finished.Load())
		return nil, nil
	})
	return ctx
}

func (p *drainProbe) requireRanAfterProvider(t *testing.T) {
	t.Helper()
	<-p.traced
	assert.Equal(t, int32(1), p.postHookCalls.Load(), "the transport post-hook runs exactly once")
	assert.True(t, p.finishedAtPostHook.Load(), "the transport post-hook ran before the provider goroutine had finished")
	assert.True(t, p.finishedAtTraceClose.Load(), "the trace completed before the provider goroutine had finished")
}

// A mid-stream error ends the client's stream at once. The provider goroutine is still winding
// down then; the post-hooks and the trace wait for it.
func TestHandleStreamingMidStreamErrorWaitsForProviderBeforePostHook(t *testing.T) {
	route := streamPostHookRoute(t, CreateAnthropicRouteConfigs("/anthropic", bifrost.NewNoOpLogger()), "/messages")
	var probe drainProbe
	ctx := probe.start(t, route, schemas.ResponsesStreamRequest, func() {})

	clientStreamClosed := make(chan struct{})
	go func() {
		status := fasthttp.StatusBadGateway
		probe.stream <- &schemas.BifrostStreamChunk{BifrostError: &schemas.BifrostError{
			StatusCode: &status, Error: &schemas.ErrorField{Message: "upstream failed"},
		}}
		// The provider's own teardown outlasts the client's stream.
		<-clientStreamClosed
		probe.finished.Store(true)
		close(probe.stream)
	}()

	body, err := io.ReadAll(ctx.Response.BodyStream())
	require.NoError(t, err)
	assert.Contains(t, string(body), "upstream failed", "the client receives the error event")
	close(clientStreamClosed)
	probe.requireRanAfterProvider(t)
}

// A Bedrock EventStream client that disconnects is noticed by a failed write, which cancels
// the request. The provider goroutine ends in response to that cancel; the post-hooks and the
// trace wait for it.
func TestHandleStreamingBedrockDisconnectWaitsForProviderBeforePostHook(t *testing.T) {
	route := RouteConfig{
		Type: RouteConfigTypeBedrock,
		StreamConfig: &StreamConfig{
			ResponsesStreamResponseConverter: func(*schemas.BifrostContext, *schemas.BifrostResponsesStreamResponse) (string, interface{}, error) {
				return "", &bedrock.BedrockStreamEvent{Role: schemas.Ptr("assistant")}, nil
			},
		},
	}
	cancelled := make(chan struct{})
	var cancelOnce sync.Once
	var probe drainProbe
	ctx := probe.start(t, route, schemas.ResponsesStreamRequest, func() { cancelOnce.Do(func() { close(cancelled) }) })

	// The client goes away before the first event is written.
	require.NoError(t, ctx.Response.CloseBodyStream())
	go func() {
		probe.stream <- &schemas.BifrostStreamChunk{BifrostResponsesStreamResponse: &schemas.BifrostResponsesStreamResponse{
			Type: schemas.ResponsesStreamResponseTypeOutputTextDelta, Delta: schemas.Ptr("hello"),
		}}
		// The provider stops when the request is cancelled, and only then finishes.
		<-cancelled
		probe.finished.Store(true)
		close(probe.stream)
	}()
	probe.requireRanAfterProvider(t)
}

// streamEndStateFor feeds chunks through handleStreaming for route and returns what the slot
// recorded for the transport post-hooks: whether the request counts as ended.
func streamEndStateFor(t *testing.T, router *GenericRouter, route RouteConfig, chunks ...*schemas.BifrostStreamChunk) (ended, captured bool) {
	t.Helper()
	stream := make(chan *schemas.BifrostStreamChunk, len(chunks))
	for _, chunk := range chunks {
		stream <- chunk
	}
	close(stream)
	ctx := &fasthttp.RequestCtx{}
	ctx.Request.Header.SetMethod(fasthttp.MethodPost)
	ctx.SetUserValue(schemas.BifrostContextKeyTransportPostHookCompleter, true)
	traced := make(chan struct{})
	ctx.SetUserValue(schemas.BifrostContextKeyTraceCompleter, func([]schemas.PluginLogEntry) { close(traced) })
	router.handleStreaming(ctx, schemas.NewBifrostContext(context.Background(), schemas.NoDeadline), route, schemas.ResponsesStreamRequest, stream, func() {})
	slot, ok := ctx.UserValue(schemas.BifrostContextKeyTransportPostHookCompleter).(*lib.TransportPostHookSlot)
	require.True(t, ok)
	slot.Resolve(func() ([]schemas.PluginLogEntry, error) {
		ended, captured = slot.RequestEnded()
		return nil, nil
	})
	_, err := io.ReadAll(ctx.Response.BodyStream())
	require.NoError(t, err)
	<-traced
	return ended, captured
}

// A chunk that passed every hook and that the route has no converter for is answered with an
// error and ends the stream. The transport did not deliver what the hooks released, and no
// hook can see that, so the request is recorded as ended and the transport post-hooks run on
// a cancelled context.
func TestHandleStreamingMissingConverterIsReportedAsUndelivered(t *testing.T) {
	delta := &schemas.BifrostStreamChunk{BifrostResponsesStreamResponse: &schemas.BifrostResponsesStreamResponse{
		Type: schemas.ResponsesStreamResponseTypeOutputTextDelta, Delta: schemas.Ptr("hello"),
	}}
	router := NewGenericRouter(nil, &mockHandlerStore{}, nil, nil, nil, bifrost.NewNoOpLogger())

	// The route streams, but has no converter for Responses chunks.
	ended, captured := streamEndStateFor(t, router, RouteConfig{StreamConfig: &StreamConfig{}}, delta)
	assert.True(t, captured)
	assert.True(t, ended, "a released chunk the transport could not render must be reported to the post-hooks")

	// The same chunk on a route that can render it is delivered.
	route := streamPostHookRoute(t, CreateAnthropicRouteConfigs("/anthropic", bifrost.NewNoOpLogger()), "/messages")
	ended, captured = streamEndStateFor(t, router, route, delta)
	assert.True(t, captured)
	assert.False(t, ended, "a stream delivered to its end is not reported as ended")
}

// A stream chunk hook that ends the stream with an error ended it itself and can account for
// it: the error is written and the request is not reported as ended.
func TestHandleStreamingInterceptorErrorIsReportedAsDelivered(t *testing.T) {
	store := &interceptorHandlerStore{
		mockHandlerStore: &mockHandlerStore{},
		interceptor:      &stubChunkInterceptor{err: assert.AnError},
	}
	router := NewGenericRouter(nil, store, nil, nil, nil, bifrost.NewNoOpLogger())
	ended, captured := streamEndStateFor(t, router, RouteConfig{StreamConfig: &StreamConfig{}}, &schemas.BifrostStreamChunk{})
	assert.True(t, captured)
	assert.False(t, ended)
}
