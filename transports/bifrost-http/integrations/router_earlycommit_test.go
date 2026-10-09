package integrations

import (
	"context"
	"io"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	bifrost "github.com/maximhq/bifrost/core"
	"github.com/maximhq/bifrost/core/schemas"
	"github.com/maximhq/bifrost/transports/bifrost-http/lib"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/valyala/fasthttp"
)

// Early SSE commitment on integration routes (fork-only, lib.StreamSetup): the
// response was committed while a plugin held the request before admission, so
// the producer obtains the stream from the setup, writes ": waiting" comments
// meanwhile, and writes a setup error in the route's own stream error shape.

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

// committedSetup returns a setup that has already committed: its wait was
// signalled and outlasted CommitAfter, while release still gates setup's result.
func committedSetup(
	t *testing.T,
	release <-chan struct{},
	result func() (chan *schemas.BifrostStreamChunk, *schemas.BifrostError),
) (*lib.StreamSetup, *schemas.BifrostContext) {
	t.Helper()
	bifrostCtx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	setup := lib.StartStreamSetup(bifrostCtx, true, func() (chan *schemas.BifrostStreamChunk, *schemas.BifrostError) {
		schemas.StreamAdmissionWaitFrom(bifrostCtx).Begin(schemas.StreamAdmissionWaitOptions{
			CommitAfter: 10 * time.Millisecond, CommentInterval: 10 * time.Millisecond,
		})
		<-release
		return result()
	})
	_, _, committed := setup.Await()
	require.True(t, committed, "the signalled wait must commit before setup ends")
	return setup, bifrostCtx
}

func readCommittedBody(t *testing.T, ctx *fasthttp.RequestCtx, release chan struct{}) string {
	t.Helper()
	body := make(chan string, 1)
	go func() {
		b, _ := io.ReadAll(ctx.Response.BodyStream())
		body <- string(b)
	}()
	time.Sleep(60 * time.Millisecond)
	close(release)
	select {
	case got := <-body:
		return got
	case <-time.After(10 * time.Second):
		t.Fatal("committed stream did not finish")
		return ""
	}
}

func TestEarlyCommitOpenAIRouteWaitsThenStreams(t *testing.T) {
	route := earlyCommitRoute(t, CreateOpenAIRouteConfigs("/openai", &mockHandlerStore{}), "/chat/completions")
	release := make(chan struct{})
	setup, bifrostCtx := committedSetup(t, release, func() (chan *schemas.BifrostStreamChunk, *schemas.BifrostError) {
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
	router.handleCommittedStreaming(ctx, bifrostCtx, route, nil, func() {}, setup, schemas.ChatCompletionStreamRequest)

	body := readCommittedBody(t, ctx, release)
	waiting := strings.Index(body, ": waiting\n")
	content := strings.Index(body, `"hello"`)
	assert.True(t, waiting >= 0 && content > waiting, "waiting comments precede the content:\n%s", body)
	assert.True(t, strings.HasSuffix(body, "data: [DONE]\n\n"), "a clean stream ends with [DONE]:\n%s", body)
}

func TestEarlyCommitAnthropicRouteErrorUsesRouteErrorShape(t *testing.T) {
	route := earlyCommitRoute(t, CreateAnthropicRouteConfigs("/anthropic", bifrost.NewNoOpLogger()), "/messages")
	release := make(chan struct{})
	setup, bifrostCtx := committedSetup(t, release, func() (chan *schemas.BifrostStreamChunk, *schemas.BifrostError) {
		return nil, newBifrostErrorWithCode(nil, "session authority unavailable", fasthttp.StatusServiceUnavailable)
	})
	router := NewGenericRouter(nil, &mockHandlerStore{}, nil, nil, nil, bifrost.NewNoOpLogger())
	ctx := &fasthttp.RequestCtx{}
	router.handleCommittedStreaming(ctx, bifrostCtx, route, nil, func() {}, setup, schemas.ChatCompletionStreamRequest)

	body := readCommittedBody(t, ctx, release)
	assert.Contains(t, body, ": waiting\n")
	assert.Contains(t, body, "event: error\n", "Anthropic streams carry errors as their own error event:\n%s", body)
	assert.Contains(t, body, "session authority unavailable")
	assert.NotContains(t, body, "[DONE]")
}

func TestEarlyCommitRefusesLargeResponsePassthrough(t *testing.T) {
	route := earlyCommitRoute(t, CreateOpenAIRouteConfigs("/openai", &mockHandlerStore{}), "/chat/completions")
	release := make(chan struct{})
	var bifrostCtx *schemas.BifrostContext
	setup, bifrostCtx := committedSetup(t, release, func() (chan *schemas.BifrostStreamChunk, *schemas.BifrostError) {
		bifrostCtx.SetValue(schemas.BifrostContextKeyLargeResponseMode, true)
		stream := make(chan *schemas.BifrostStreamChunk)
		close(stream)
		return stream, nil
	})
	router := NewGenericRouter(nil, &mockHandlerStore{}, nil, nil, nil, bifrost.NewNoOpLogger())
	ctx := &fasthttp.RequestCtx{}
	router.handleCommittedStreaming(ctx, bifrostCtx, route, nil, func() {}, setup, schemas.ChatCompletionStreamRequest)

	body := readCommittedBody(t, ctx, release)
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

// Regression for review of soby/bifrost#19: a committed stream that refuses
// large-response passthrough must still release the passthrough reader, once,
// after cancelling the request so Close cannot wait on a live upstream.
func TestEarlyCommitClosesRejectedLargeResponseReaderOnce(t *testing.T) {
	route := earlyCommitRoute(t, CreateOpenAIRouteConfigs("/openai", &mockHandlerStore{}), "/chat/completions")
	release := make(chan struct{})
	raw := &earlyCommitUpstreamReader{Reader: strings.NewReader("raw upstream")}
	var bifrostCtx *schemas.BifrostContext
	setup, bifrostCtx := committedSetup(t, release, func() (chan *schemas.BifrostStreamChunk, *schemas.BifrostError) {
		raw.ctx = bifrostCtx
		bifrostCtx.SetValue(schemas.BifrostContextKeyLargeResponseMode, true)
		bifrostCtx.SetValue(schemas.BifrostContextKeyLargeResponseReader, io.ReadCloser(raw))
		stream := make(chan *schemas.BifrostStreamChunk)
		close(stream)
		return stream, nil
	})
	router := NewGenericRouter(nil, &mockHandlerStore{}, nil, nil, nil, bifrost.NewNoOpLogger())
	ctx := &fasthttp.RequestCtx{}
	finished := make(chan struct{})
	var once sync.Once
	router.handleCommittedStreaming(ctx, bifrostCtx, route, nil, func() {
		once.Do(func() {
			bifrostCtx.Cancel()
			close(finished)
		})
	}, setup, schemas.ChatCompletionStreamRequest)

	body := readCommittedBody(t, ctx, release)
	assert.Contains(t, body, "too large to stream")
	select {
	case <-finished:
	case <-time.After(5 * time.Second):
		t.Fatal("cleanup did not complete")
	}
	assert.Equal(t, int32(1), raw.closed.Load(), "the rejected passthrough reader is closed exactly once")
	assert.True(t, raw.cancelledBefore.Load(), "the request is cancelled before the reader is closed")
	assert.Nil(t, bifrostCtx.Value(schemas.BifrostContextKeyLargeResponseReader), "no later owner may find the closed reader")
}
