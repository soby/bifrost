package integrations

import (
	"bufio"
	"context"
	"io"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	bifrost "github.com/maximhq/bifrost/core"
	"github.com/maximhq/bifrost/core/schemas"
	"github.com/maximhq/bifrost/transports/v2/bifrost-http/lib"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/valyala/fasthttp"
)

// Early SSE commitment on integration routes (lib.StreamSetup): the response was committed
// while a plugin held the request before the provider call, so the producer obtains the
// stream from the setup while the heartbeat keeps the connection alive, and writes a setup
// error in the route's own stream error shape.

func earlyCommitRoute(t *testing.T, routes []RouteConfig, pathFragment string) RouteConfig {
	t.Helper()
	for _, route := range routes {
		if route.StreamConfig != nil && strings.Contains(route.Path, pathFragment) {
			return route
		}
	}
	t.Fatalf("no streaming route containing %q", pathFragment)
	return RouteConfig{}
}

// committedSetup returns a setup that has already committed: its wait was signalled while
// release still gates setup's result.
func committedSetup(
	t *testing.T,
	release <-chan struct{},
	result func(ctx *schemas.BifrostContext) (chan *schemas.BifrostStreamChunk, *schemas.BifrostError),
) (*lib.StreamSetup, *schemas.BifrostContext) {
	t.Helper()
	bifrostCtx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	setup := lib.StartStreamSetup(bifrostCtx, true, func() (chan *schemas.BifrostStreamChunk, *schemas.BifrostError) {
		schemas.StreamAdmissionWaitFrom(bifrostCtx).Begin()
		<-release
		return result(bifrostCtx)
	})
	_, _, committed := setup.Await()
	require.True(t, committed, "the signalled wait must commit before setup ends")
	return setup, bifrostCtx
}

// readCommittedBody reads the committed stream up to its first heartbeat, releases setup,
// and returns what came before the heartbeat and what came after.
func readCommittedBody(t *testing.T, ctx *fasthttp.RequestCtx, release chan struct{}) (before, after string) {
	t.Helper()
	br := bufio.NewReader(ctx.Response.BodyStream())
	var head strings.Builder
	for {
		line, err := br.ReadString('\n')
		require.NoError(t, err, "stream ended before a heartbeat; read %q", head.String())
		if strings.HasPrefix(line, ": heartbeat") {
			break
		}
		head.WriteString(line)
	}
	close(release)
	rest, err := io.ReadAll(br)
	require.NoError(t, err)
	return head.String(), string(rest)
}

func TestEarlyCommitOpenAIRouteWaitsThenStreams(t *testing.T) {
	route := earlyCommitRoute(t, CreateOpenAIRouteConfigs("/openai", &mockHandlerStore{}), "/chat/completions")
	release := make(chan struct{})
	setup, bifrostCtx := committedSetup(t, release, func(*schemas.BifrostContext) (chan *schemas.BifrostStreamChunk, *schemas.BifrostError) {
		stream := make(chan *schemas.BifrostStreamChunk, 1)
		stream <- &schemas.BifrostStreamChunk{BifrostChatResponse: &schemas.BifrostChatResponse{
			ID: "chatcmpl-1", Object: "chat.completion.chunk", Model: "m",
			Choices: []schemas.BifrostResponseChoice{{ChatStreamResponseChoice: &schemas.ChatStreamResponseChoice{
				Delta: &schemas.ChatStreamResponseChoiceDelta{Content: schemas.Ptr("hello")},
			}}},
		}}
		close(stream)
		return stream, nil
	})
	router := NewGenericRouter(nil, &mockHandlerStore{}, nil, nil, nil, bifrost.NewNoOpLogger())
	ctx := &fasthttp.RequestCtx{}
	router.handleCommittedStreaming(ctx, bifrostCtx, route, schemas.ChatCompletionStreamRequest, nil, func() {}, setup)

	before, body := readCommittedBody(t, ctx, release)
	assert.Empty(t, before, "nothing but heartbeats is sent while setup is held")
	assert.Contains(t, body, `"hello"`)
	assert.True(t, strings.HasSuffix(body, "data: [DONE]\n\n"), "a clean stream ends with [DONE]:\n%s", body)
}

func TestEarlyCommitAnthropicRouteErrorUsesRouteErrorShape(t *testing.T) {
	route := earlyCommitRoute(t, CreateAnthropicRouteConfigs("/anthropic", bifrost.NewNoOpLogger()), "/messages")
	release := make(chan struct{})
	setup, bifrostCtx := committedSetup(t, release, func(*schemas.BifrostContext) (chan *schemas.BifrostStreamChunk, *schemas.BifrostError) {
		return nil, newBifrostErrorWithCode(nil, "lookup service unavailable", fasthttp.StatusServiceUnavailable)
	})
	router := NewGenericRouter(nil, &mockHandlerStore{}, nil, nil, nil, bifrost.NewNoOpLogger())
	ctx := &fasthttp.RequestCtx{}
	router.handleCommittedStreaming(ctx, bifrostCtx, route, schemas.ChatCompletionStreamRequest, nil, func() {}, setup)

	_, body := readCommittedBody(t, ctx, release)
	assert.Contains(t, body, "event: error\n", "Anthropic streams carry errors as their own error event:\n%s", body)
	assert.Contains(t, body, "lookup service unavailable")
	assert.NotContains(t, body, "[DONE]")
}

func TestEarlyCommitRefusesLargeResponsePassthrough(t *testing.T) {
	route := earlyCommitRoute(t, CreateOpenAIRouteConfigs("/openai", &mockHandlerStore{}), "/chat/completions")
	release := make(chan struct{})
	setup, bifrostCtx := committedSetup(t, release, func(ctx *schemas.BifrostContext) (chan *schemas.BifrostStreamChunk, *schemas.BifrostError) {
		ctx.SetValue(schemas.BifrostContextKeyLargeResponseMode, true)
		stream := make(chan *schemas.BifrostStreamChunk)
		close(stream)
		return stream, nil
	})
	router := NewGenericRouter(nil, &mockHandlerStore{}, nil, nil, nil, bifrost.NewNoOpLogger())
	ctx := &fasthttp.RequestCtx{}
	router.handleCommittedStreaming(ctx, bifrostCtx, route, schemas.ChatCompletionStreamRequest, nil, func() {}, setup)

	_, body := readCommittedBody(t, ctx, release)
	assert.Contains(t, body, "too large to stream", "a committed stream cannot switch to raw passthrough:\n%s", body)
	assert.NotContains(t, body, "[DONE]")
}

// earlyCommitUpstreamReader stands in for the large-response passthrough
// reader, which owns the upstream response once setup hands it over.
type earlyCommitUpstreamReader struct {
	io.Reader
	closed          atomic.Int32
	cancelledBefore atomic.Bool
	ctx             *schemas.BifrostContext
}

func (reader *earlyCommitUpstreamReader) Close() error {
	reader.cancelledBefore.Store(reader.ctx.Err() != nil)
	reader.closed.Add(1)
	return nil
}

// A committed stream that refuses large-response passthrough must still release the
// passthrough reader, once, after cancelling the request so Close cannot wait on a live
// upstream.
func TestEarlyCommitClosesRejectedLargeResponseReaderOnce(t *testing.T) {
	route := earlyCommitRoute(t, CreateOpenAIRouteConfigs("/openai", &mockHandlerStore{}), "/chat/completions")
	release := make(chan struct{})
	raw := &earlyCommitUpstreamReader{Reader: strings.NewReader("raw upstream")}
	setup, bifrostCtx := committedSetup(t, release, func(ctx *schemas.BifrostContext) (chan *schemas.BifrostStreamChunk, *schemas.BifrostError) {
		raw.ctx = ctx
		ctx.SetValue(schemas.BifrostContextKeyLargeResponseMode, true)
		ctx.SetValue(schemas.BifrostContextKeyLargeResponseReader, io.ReadCloser(raw))
		stream := make(chan *schemas.BifrostStreamChunk)
		close(stream)
		return stream, nil
	})
	router := NewGenericRouter(nil, &mockHandlerStore{}, nil, nil, nil, bifrost.NewNoOpLogger())
	ctx := &fasthttp.RequestCtx{}
	// The trace completer is the producer teardown's last step that calls out.
	torndown := make(chan struct{})
	ctx.SetUserValue(schemas.BifrostContextKeyTraceCompleter, func([]schemas.PluginLogEntry) { close(torndown) })
	var once sync.Once
	router.handleCommittedStreaming(ctx, bifrostCtx, route, schemas.ChatCompletionStreamRequest, nil, func() {
		once.Do(bifrostCtx.Cancel)
	}, setup)

	_, body := readCommittedBody(t, ctx, release)
	assert.Contains(t, body, "too large to stream")
	<-torndown
	assert.Equal(t, int32(1), raw.closed.Load(), "the rejected passthrough reader is closed exactly once")
	assert.True(t, raw.cancelledBefore.Load(), "the request is cancelled before the reader is closed")
	assert.Nil(t, bifrostCtx.Value(schemas.BifrostContextKeyLargeResponseReader), "no later owner may find the closed reader")
}
