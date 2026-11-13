package integrations

import (
	"bytes"
	"context"
	"errors"
	"io"
	"sync"
	"testing"

	"github.com/bytedance/sonic"
	bifrost "github.com/maximhq/bifrost/core"
	"github.com/maximhq/bifrost/core/schemas"
	"github.com/maximhq/bifrost/transports/v2/bifrost-http/lib"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/valyala/fasthttp"
)

// A plugin that rewrites a streamed response may need to send the client more than one
// chunk for one provider chunk, in order: for example content it passes through followed
// by a replacement it produced. Its stream chunk hook writes the additional chunks with
// schemas.WriteStreamChunk; these tests run that through an integration route's stream.

// chunkWriterPlugin is a transport plugin whose stream chunk hook is supplied by the test.
type chunkWriterPlugin struct {
	name    string
	onChunk chunkHook
}

type chunkHook = func(ctx *schemas.BifrostContext, chunk *schemas.BifrostStreamChunk) (*schemas.BifrostStreamChunk, error)

func (p *chunkWriterPlugin) GetName() string {
	if p.name == "" {
		return "chunk-writer"
	}
	return p.name
}
func (p *chunkWriterPlugin) Cleanup() error { return nil }
func (p *chunkWriterPlugin) HTTPTransportPreAuthHook(*schemas.BifrostContext, *schemas.HTTPRequest) (*schemas.HTTPResponse, error) {
	return nil, nil
}

func (p *chunkWriterPlugin) HTTPTransportPreHook(*schemas.BifrostContext, *schemas.HTTPRequest) (*schemas.HTTPResponse, error) {
	return nil, nil
}

func (p *chunkWriterPlugin) HTTPTransportPostHook(*schemas.BifrostContext, *schemas.HTTPRequest, *schemas.HTTPResponse) error {
	return nil
}

func (p *chunkWriterPlugin) HTTPTransportResponseHeadersHook(*schemas.BifrostContext, *schemas.HTTPRequest, *schemas.HTTPResponseMetadata) error {
	return nil
}

func (p *chunkWriterPlugin) HTTPTransportStreamChunkHook(ctx *schemas.BifrostContext, _ *schemas.HTTPRequest, chunk *schemas.BifrostStreamChunk) (*schemas.BifrostStreamChunk, error) {
	return p.onChunk(ctx, chunk)
}

// frameRecorder is a stream frame tap that keeps each Write as one frame.
type frameRecorder struct {
	// onFrame, when set, runs for each frame on the stream's send path.
	onFrame func(frame string)

	mu     sync.Mutex
	frames []string
}

func (r *frameRecorder) Write(p []byte) (int, error) {
	r.mu.Lock()
	r.frames = append(r.frames, string(p))
	r.mu.Unlock()
	if r.onFrame != nil {
		r.onFrame(string(p))
	}
	return len(p), nil
}

func (r *frameRecorder) snapshot() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]string(nil), r.frames...)
}

const heartbeatFrame = ": heartbeat\n"

// withoutHeartbeats drops heartbeat comments, which follow the clock, not the stream.
func withoutHeartbeats(frames []string) []string {
	kept := make([]string, 0, len(frames))
	for _, frame := range frames {
		if frame != heartbeatFrame {
			kept = append(kept, frame)
		}
	}
	return kept
}

func chatDeltaChunk(text string) *schemas.BifrostStreamChunk {
	return &schemas.BifrostStreamChunk{
		BifrostChatResponse: &schemas.BifrostChatResponse{
			ID:     "chatcmpl-1",
			Object: "chat.completion.chunk",
			Model:  "test-model",
			Choices: []schemas.BifrostResponseChoice{{
				Index: 0,
				ChatStreamResponseChoice: &schemas.ChatStreamResponseChoice{
					Delta: &schemas.ChatStreamResponseChoiceDelta{Content: schemas.Ptr(text)},
				},
			}},
		},
	}
}

func chatFinishChunk() *schemas.BifrostStreamChunk {
	chunk := chatDeltaChunk("")
	chunk.BifrostChatResponse.Choices[0].FinishReason = schemas.Ptr("stop")
	return chunk
}

func chunkText(chunk *schemas.BifrostStreamChunk) string {
	if chunk == nil || chunk.BifrostChatResponse == nil || len(chunk.BifrostChatResponse.Choices) == 0 {
		return ""
	}
	choice := chunk.BifrostChatResponse.Choices[0]
	if choice.ChatStreamResponseChoice == nil || choice.Delta == nil || choice.Delta.Content == nil {
		return ""
	}
	return *choice.Delta.Content
}

func isFinishChunk(chunk *schemas.BifrostStreamChunk) bool {
	return chunk != nil && chunk.BifrostChatResponse != nil && len(chunk.BifrostChatResponse.Choices) > 0 &&
		chunk.BifrostChatResponse.Choices[0].FinishReason != nil
}

// chunkWriterRoute is one route format and the exact frame it writes for a chat chunk.
type chunkWriterRoute struct {
	name   string
	config RouteConfig
	frame  func(t *testing.T, chunk *schemas.BifrostStreamChunk) string
	done   string
}

var chunkWriterRoutes = []chunkWriterRoute{
	{
		// The OpenAI route: a JSON event per chunk and a [DONE] marker.
		name: "openai chat completions",
		config: RouteConfig{
			Type:         RouteConfigTypeOpenAI,
			Path:         "/openai/v1/chat/completions",
			StreamConfig: &StreamConfig{ChatStreamResponseConverter: openAIChatStreamWireConverter},
		},
		frame: func(t *testing.T, chunk *schemas.BifrostStreamChunk) string {
			t.Helper()
			_, converted, err := openAIChatStreamWireConverter(nil, chunk.BifrostChatResponse)
			require.NoError(t, err)
			body, err := sonic.Marshal(converted)
			require.NoError(t, err)
			return "data: " + string(body) + "\n\n"
		},
		done: "data: [DONE]\n\n",
	},
	{
		// A route whose converter names the event and returns a pre-formatted frame, and
		// whose stream ends without a marker.
		name: "named pre-formatted events",
		config: RouteConfig{
			Type: RouteConfigTypeAnthropic,
			Path: "/anthropic/v1/messages",
			StreamConfig: &StreamConfig{
				ChatStreamResponseConverter: func(_ *schemas.BifrostContext, resp *schemas.BifrostChatResponse) (string, interface{}, error) {
					return "content_block_delta", namedEventData(&schemas.BifrostStreamChunk{BifrostChatResponse: resp}), nil
				},
			},
		},
		frame: func(_ *testing.T, chunk *schemas.BifrostStreamChunk) string {
			return "event: content_block_delta\n" + namedEventData(chunk)
		},
	},
}

func namedEventData(chunk *schemas.BifrostStreamChunk) string {
	if isFinishChunk(chunk) {
		return "data: {\"stop\":true}\n\n"
	}
	return "data: {\"text\":\"" + chunkText(chunk) + "\"}\n\n"
}

// chunkWriterStream is one stream served by handleStreaming with the plugin's hook on it.
type chunkWriterStream struct {
	body   io.Reader
	closer io.Closer
	tap    *frameRecorder
	cancel *cancelRecorder
	ctx    *schemas.BifrostContext
}

// startChunkWriterStream serves chunks on route. hook is the plugin's stream chunk hook;
// nil serves the stream without any transport plugin.
func startChunkWriterStream(t *testing.T, route chunkWriterRoute, chunks []*schemas.BifrostStreamChunk, hook func(*schemas.BifrostContext, *schemas.BifrostStreamChunk) (*schemas.BifrostStreamChunk, error)) *chunkWriterStream {
	t.Helper()
	return startTappedChunkWriterStream(t, route, chunks, hook, nil)
}

// startTappedChunkWriterStream is startChunkWriterStream with onFrame run for each frame
// on the stream's send path.
func startTappedChunkWriterStream(t *testing.T, route chunkWriterRoute, chunks []*schemas.BifrostStreamChunk, hook chunkHook, onFrame func(string)) *chunkWriterStream {
	t.Helper()
	var plugins []*chunkWriterPlugin
	if hook != nil {
		plugins = []*chunkWriterPlugin{{onChunk: hook}}
	}
	return startPluginChunkWriterStream(t, route, chunks, plugins, onFrame)
}

// startPluginChunkWriterStream serves chunks on route with plugins loaded in the order
// given; their stream chunk hooks run in the reverse order.
func startPluginChunkWriterStream(t *testing.T, route chunkWriterRoute, chunks []*schemas.BifrostStreamChunk, plugins []*chunkWriterPlugin, onFrame func(string)) *chunkWriterStream {
	t.Helper()
	stream := make(chan *schemas.BifrostStreamChunk, len(chunks))
	for _, chunk := range chunks {
		stream <- chunk
	}
	close(stream)

	var store lib.HandlerStore = &mockHandlerStore{}
	if len(plugins) > 0 {
		config := &lib.Config{}
		for _, plugin := range plugins {
			require.NoError(t, config.ReloadPlugin(plugin))
		}
		interceptor := config.GetStreamChunkInterceptor()
		require.NotNil(t, interceptor)
		store = &interceptorHandlerStore{mockHandlerStore: &mockHandlerStore{}, interceptor: interceptor}
	}

	s := &chunkWriterStream{tap: &frameRecorder{onFrame: onFrame}, cancel: newCancelRecorder()}
	s.ctx = schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	s.ctx.SetValue(schemas.BifrostContextKeyStreamFrameTap, io.Writer(s.tap))
	reqCtx := &fasthttp.RequestCtx{}
	NewGenericRouter(nil, store, nil, nil, nil, bifrost.NewNoOpLogger()).handleStreaming(reqCtx, s.ctx, route.config, "", stream, s.cancel.cancel)
	s.body = reqCtx.Response.BodyStream()
	s.closer = reqCtx.Response.BodyStream().(io.Closer)
	return s
}

// nextFrame reads the next frame the client receives, skipping heartbeats. The stream
// reader hands over one frame per Read.
func (s *chunkWriterStream) nextFrame(t *testing.T) (string, bool) {
	t.Helper()
	buf := make([]byte, 64<<10)
	for {
		n, err := s.body.Read(buf)
		if err == io.EOF {
			return "", false
		}
		require.NoError(t, err)
		if frame := string(buf[:n]); frame != heartbeatFrame {
			return frame, true
		}
	}
}

// frames reads the stream to its end.
func (s *chunkWriterStream) frames(t *testing.T) []string {
	t.Helper()
	var frames []string
	for {
		frame, ok := s.nextFrame(t)
		if !ok {
			return frames
		}
		frames = append(frames, frame)
	}
}

func framesOf(t *testing.T, route chunkWriterRoute, done bool, chunks ...*schemas.BifrostStreamChunk) []string {
	t.Helper()
	frames := make([]string, 0, len(chunks)+1)
	for _, chunk := range chunks {
		frames = append(frames, route.frame(t, chunk))
	}
	if done && route.done != "" {
		frames = append(frames, route.done)
	}
	return frames
}

// Chunks the hook writes reach the client in call order, ahead of the chunk the hook
// returns, each as its own frame in the route's format, and each before the hook goes on:
// the hook here does not continue until the client has read the first one.
func TestStreamChunkWriterWritesAheadOfTheReturnedChunk(t *testing.T) {
	for _, route := range chunkWriterRoutes {
		t.Run(route.name, func(t *testing.T) {
			clientReadFirst := make(chan struct{})
			var wrote []bool
			s := startChunkWriterStream(t, route,
				[]*schemas.BifrostStreamChunk{chatDeltaChunk("one"), chatDeltaChunk("two"), chatFinishChunk()},
				func(ctx *schemas.BifrostContext, chunk *schemas.BifrostStreamChunk) (*schemas.BifrostStreamChunk, error) {
					if chunkText(chunk) != "two" {
						return chunk, nil
					}
					wrote = append(wrote, schemas.WriteStreamChunk(ctx, chatDeltaChunk("two-a")))
					<-clientReadFirst
					wrote = append(wrote, schemas.WriteStreamChunk(ctx, chatDeltaChunk("two-b")))
					return chunk, nil
				})

			var wire []string
			for range 2 {
				frame, ok := s.nextFrame(t)
				require.True(t, ok)
				wire = append(wire, frame)
			}
			require.Equal(t, framesOf(t, route, false, chatDeltaChunk("one"), chatDeltaChunk("two-a")), wire,
				"the first written chunk reaches the client while the hook is still running")
			close(clientReadFirst)
			wire = append(wire, s.frames(t)...)

			want := framesOf(t, route, true, chatDeltaChunk("one"), chatDeltaChunk("two-a"), chatDeltaChunk("two-b"), chatDeltaChunk("two"), chatFinishChunk())
			assert.Equal(t, want, wire)
			assert.Equal(t, wire, withoutHeartbeats(s.tap.snapshot()), "the frame tap sees every frame as written, in order")
			assert.Equal(t, []bool{true, true}, wrote)
		})
	}
}

// A hook that returns no chunk still gets the chunks it wrote to the client.
func TestStreamChunkWriterWithNoReturnedChunk(t *testing.T) {
	for _, route := range chunkWriterRoutes {
		t.Run(route.name, func(t *testing.T) {
			s := startChunkWriterStream(t, route,
				[]*schemas.BifrostStreamChunk{chatDeltaChunk("one"), chatDeltaChunk("two"), chatFinishChunk()},
				func(ctx *schemas.BifrostContext, chunk *schemas.BifrostStreamChunk) (*schemas.BifrostStreamChunk, error) {
					if chunkText(chunk) != "two" {
						return chunk, nil
					}
					schemas.WriteStreamChunk(ctx, chatDeltaChunk("replacement"))
					return nil, nil
				})

			wire := s.frames(t)
			assert.Equal(t, framesOf(t, route, true, chatDeltaChunk("one"), chatDeltaChunk("replacement"), chatFinishChunk()), wire)
			assert.Equal(t, wire, withoutHeartbeats(s.tap.snapshot()))
		})
	}
}

// A chunk written ahead of the stream's last chunk precedes it.
func TestStreamChunkWriterAheadOfTheTerminalChunk(t *testing.T) {
	for _, route := range chunkWriterRoutes {
		t.Run(route.name, func(t *testing.T) {
			s := startChunkWriterStream(t, route,
				[]*schemas.BifrostStreamChunk{chatDeltaChunk("one"), chatFinishChunk()},
				func(ctx *schemas.BifrostContext, chunk *schemas.BifrostStreamChunk) (*schemas.BifrostStreamChunk, error) {
					if isFinishChunk(chunk) {
						schemas.WriteStreamChunk(ctx, chatDeltaChunk("tail"))
					}
					return chunk, nil
				})

			wire := s.frames(t)
			assert.Equal(t, framesOf(t, route, true, chatDeltaChunk("one"), chatDeltaChunk("tail"), chatFinishChunk()), wire)
			assert.Equal(t, wire, withoutHeartbeats(s.tap.snapshot()))
		})
	}
}

// A chunk written before the hook returns an error stays written. The error event follows
// it and ends the stream: the hook is not called again and nothing else is written.
func TestStreamChunkWriterThenHookError(t *testing.T) {
	for _, route := range chunkWriterRoutes {
		t.Run(route.name, func(t *testing.T) {
			calls := 0
			s := startChunkWriterStream(t, route,
				[]*schemas.BifrostStreamChunk{chatDeltaChunk("one"), chatDeltaChunk("two"), chatFinishChunk()},
				func(ctx *schemas.BifrostContext, chunk *schemas.BifrostStreamChunk) (*schemas.BifrostStreamChunk, error) {
					calls++
					if chunkText(chunk) != "two" {
						return chunk, nil
					}
					schemas.WriteStreamChunk(ctx, chatDeltaChunk("kept"))
					return nil, errors.New("stop here")
				})

			wire := s.frames(t)
			want := append(framesOf(t, route, false, chatDeltaChunk("one"), chatDeltaChunk("kept")),
				"event: error\ndata: {\"error\":\"failed to intercept chunk with plugin chunk-writer: stop here\"}\n\n")
			assert.Equal(t, want, wire)
			assert.Equal(t, wire, withoutHeartbeats(s.tap.snapshot()))
			assert.Equal(t, 2, calls)
			s.cancel.requireCancelled(t, "the stream ends at the hook's error")
		})
	}
}

// The writer writes response chunks during a hook call and nothing else: it refuses a nil
// chunk and an error chunk, and it is inert once the hook has returned.
func TestStreamChunkWriterRefusals(t *testing.T) {
	route := chunkWriterRoutes[0]
	var refusedNil, refusedError bool
	s := startChunkWriterStream(t, route, []*schemas.BifrostStreamChunk{chatDeltaChunk("one")},
		func(ctx *schemas.BifrostContext, chunk *schemas.BifrostStreamChunk) (*schemas.BifrostStreamChunk, error) {
			refusedNil = !schemas.WriteStreamChunk(ctx, nil)
			refusedError = !schemas.WriteStreamChunk(ctx, &schemas.BifrostStreamChunk{BifrostError: &schemas.BifrostError{Error: &schemas.ErrorField{Message: "no"}}})
			return chunk, nil
		})

	wire := s.frames(t)
	assert.Equal(t, framesOf(t, route, true, chatDeltaChunk("one")), wire)
	assert.True(t, refusedNil, "a nil chunk is not written")
	assert.True(t, refusedError, "an error chunk is not written")
	assert.False(t, schemas.WriteStreamChunk(s.ctx, chatDeltaChunk("late")), "the request context carries no writer")
	assert.Equal(t, wire, withoutHeartbeats(s.tap.snapshot()))
}

// A write that finds the client gone reports false, and the stream ends when the hook
// returns: the returned chunk is not written and the hook is not called again.
func TestStreamChunkWriterClientGone(t *testing.T) {
	for _, route := range chunkWriterRoutes {
		t.Run(route.name, func(t *testing.T) {
			calls := 0
			var wrote, wroteAgain bool
			var s *chunkWriterStream
			started := make(chan struct{})
			hook := func(ctx *schemas.BifrostContext, chunk *schemas.BifrostStreamChunk) (*schemas.BifrostStreamChunk, error) {
				<-started
				calls++
				_ = s.closer.Close() // the client goes away
				wrote = schemas.WriteStreamChunk(ctx, chatDeltaChunk("unsent"))
				wroteAgain = schemas.WriteStreamChunk(ctx, chatDeltaChunk("unsent"))
				return chunk, nil
			}
			s = startChunkWriterStream(t, route, []*schemas.BifrostStreamChunk{chatDeltaChunk("one"), chatDeltaChunk("two"), chatFinishChunk()}, hook)
			close(started)

			assert.Empty(t, s.frames(t))
			s.cancel.requireCancelled(t, "a failed write cancels the request")
			assert.False(t, wrote)
			assert.False(t, wroteAgain)
			assert.Equal(t, 1, calls)
			assert.Empty(t, withoutHeartbeats(s.tap.snapshot()))
		})
	}
}

// A hook that never writes a chunk leaves the stream exactly as it is without any plugin.
func TestStreamChunkWriterUnusedLeavesTheStreamUnchanged(t *testing.T) {
	for _, route := range chunkWriterRoutes {
		t.Run(route.name, func(t *testing.T) {
			chunks := func() []*schemas.BifrostStreamChunk {
				return []*schemas.BifrostStreamChunk{chatDeltaChunk("one"), chatDeltaChunk("two"), chatFinishChunk()}
			}
			plain := startChunkWriterStream(t, route, chunks(), nil)
			hooked := startChunkWriterStream(t, route, chunks(),
				func(_ *schemas.BifrostContext, chunk *schemas.BifrostStreamChunk) (*schemas.BifrostStreamChunk, error) {
					return chunk, nil
				})

			want := plain.frames(t)
			assert.Equal(t, framesOf(t, route, true, chunks()...), want)
			assert.Equal(t, want, hooked.frames(t))
			assert.Nil(t, plain.ctx.Value(schemas.BifrostContextKeyStreamChunkWriter), "no writer is offered to a stream no hook runs on")
			assert.True(t, bytes.Equal([]byte(joinFrames(want)), []byte(joinFrames(withoutHeartbeats(hooked.tap.snapshot())))))
		})
	}
}

func joinFrames(frames []string) string {
	var joined bytes.Buffer
	for _, frame := range frames {
		joined.WriteString(frame)
	}
	return joined.String()
}

// noHeartbeatRoute is a route whose stream carries no heartbeat, so nothing but the
// producer ever closes its stream reader.
var noHeartbeatRoute = chunkWriterRoute{
	name: "no heartbeat",
	config: RouteConfig{
		Type: RouteConfigTypeOpenAI,
		Path: "/openai/v1/chat/completions",
		StreamConfig: &StreamConfig{
			ChatStreamResponseConverter: openAIChatStreamWireConverter,
			HeartbeatFraming:            lib.SSEHeartbeatNone,
		},
	},
	frame: chunkWriterRoutes[0].frame,
	done:  "data: [DONE]\n\n",
}

// A hook that panics leaves no usable writer behind: once the stream has ended, a write
// through the context that hook call received, or through the request context the
// post-hooks derive theirs from, reports false and touches nothing.
func TestStreamChunkWriterIsClosedAfterAHookPanic(t *testing.T) {
	for _, route := range []chunkWriterRoute{chunkWriterRoutes[0], noHeartbeatRoute} {
		t.Run(route.name, func(t *testing.T) {
			var hookCtx *schemas.BifrostContext
			var wroteBeforePanic bool
			s := startChunkWriterStream(t, route,
				[]*schemas.BifrostStreamChunk{chatDeltaChunk("one"), chatDeltaChunk("two")},
				func(ctx *schemas.BifrostContext, chunk *schemas.BifrostStreamChunk) (*schemas.BifrostStreamChunk, error) {
					hookCtx = ctx
					wroteBeforePanic = schemas.WriteStreamChunk(ctx, chatDeltaChunk("before"))
					panic("hook bug")
				})

			wire := s.frames(t)
			require.Len(t, wire, 2, "the written chunk, then the error frame the panic ends the stream with")
			assert.Equal(t, route.frame(t, chatDeltaChunk("before")), wire[0])
			assert.True(t, wroteBeforePanic)
			s.cancel.requireCancelled(t, "a panic ends the stream")

			// The stream is closed now. Neither context can write to it.
			assert.False(t, schemas.WriteStreamChunk(hookCtx, chatDeltaChunk("late")))
			assert.False(t, schemas.WriteStreamChunk(s.ctx, chatDeltaChunk("late")))
			assert.Equal(t, wire, withoutHeartbeats(s.tap.snapshot()))
		})
	}
}

// writeElsewhere writes a chunk through ctx from another goroutine and waits for the result.
func writeElsewhere(ctx *schemas.BifrostContext, text string) bool {
	result := make(chan bool)
	go func() { result <- schemas.WriteStreamChunk(ctx, chatDeltaChunk(text)) }()
	return <-result
}

// Only the hook call writes. Another goroutine holding the request context has no writer,
// during the call or after it. One the hook hands its own context to is part of the call:
// its write waits for the write in progress and follows it. After the call has returned
// that context writes nothing.
func TestStreamChunkWriterBelongsToTheHookCall(t *testing.T) {
	for _, route := range []chunkWriterRoute{chunkWriterRoutes[0], noHeartbeatRoute} {
		t.Run(route.name, func(t *testing.T) {
			var hookCtx *schemas.BifrostContext
			var duringFromRequestCtx, wrote, wroteBeside bool
			beside := make(chan bool, 1)
			midWrite := route.frame(t, chatDeltaChunk("two-a"))
			s := startTappedChunkWriterStream(t, route,
				[]*schemas.BifrostStreamChunk{chatDeltaChunk("one"), chatDeltaChunk("two"), chatFinishChunk()},
				func(ctx *schemas.BifrostContext, chunk *schemas.BifrostStreamChunk) (*schemas.BifrostStreamChunk, error) {
					if chunkText(chunk) != "two" {
						return chunk, nil
					}
					hookCtx = ctx
					duringFromRequestCtx = writeElsewhere(ctx.Root(), "elsewhere")
					wrote = schemas.WriteStreamChunk(ctx, chatDeltaChunk("two-a"))
					wroteBeside = <-beside
					return chunk, nil
				},
				func(frame string) {
					// On the send path of the hook's own write: a second write through
					// the call's context starts while the first is in progress.
					if frame == midWrite {
						go func() { beside <- schemas.WriteStreamChunk(hookCtx, chatDeltaChunk("two-b")) }()
					}
				})

			wire := s.frames(t)
			want := framesOf(t, route, true, chatDeltaChunk("one"), chatDeltaChunk("two-a"), chatDeltaChunk("two-b"), chatDeltaChunk("two"), chatFinishChunk())
			assert.Equal(t, want, wire)
			assert.True(t, wrote)
			assert.True(t, wroteBeside, "a write through the call's context waits for the one in progress")
			assert.False(t, duringFromRequestCtx, "the request context carries no writer during the call")
			assert.False(t, writeElsewhere(hookCtx, "late"), "the call's context is closed once the call has returned")
			assert.False(t, writeElsewhere(s.ctx, "late"), "the request context carries no writer after the call")
			assert.Equal(t, wire, withoutHeartbeats(s.tap.snapshot()))
		})
	}
}

// A context kept from one hook call never writes again: not while a later chunk's hook
// call is running, from any goroutine, and not while that call writes. The later call's
// own writes are unaffected by it.
func TestStreamChunkWriterOfAFinishedCallStaysClosedDuringLaterCalls(t *testing.T) {
	for _, route := range []chunkWriterRoute{chunkWriterRoutes[0], noHeartbeatRoute} {
		t.Run(route.name, func(t *testing.T) {
			var firstCallCtx *schemas.BifrostContext
			var wroteInFirstCall, staleDuringSecondCall, staleOnHookGoroutine, wrote bool
			var staleAttempts, staleWrites int
			s := startChunkWriterStream(t, route,
				[]*schemas.BifrostStreamChunk{chatDeltaChunk("one"), chatDeltaChunk("two"), chatFinishChunk()},
				func(ctx *schemas.BifrostContext, chunk *schemas.BifrostStreamChunk) (*schemas.BifrostStreamChunk, error) {
					switch chunkText(chunk) {
					case "one":
						firstCallCtx = ctx
						wroteInFirstCall = schemas.WriteStreamChunk(ctx, chatDeltaChunk("one-a"))
					case "two":
						staleDuringSecondCall = writeElsewhere(firstCallCtx, "stale")
						staleOnHookGoroutine = schemas.WriteStreamChunk(firstCallCtx, chatDeltaChunk("stale"))

						// The first call's context is used over and over on another
						// goroutine for as long as this call's write takes.
						stop, stopped := make(chan struct{}), make(chan struct{})
						started := make(chan struct{})
						go func() {
							defer close(stopped)
							for first := true; ; first = false {
								staleAttempts++
								if schemas.WriteStreamChunk(firstCallCtx, chatDeltaChunk("stale")) {
									staleWrites++
								}
								if first {
									close(started)
								}
								select {
								case <-stop:
									return
								default:
								}
							}
						}()
						<-started
						wrote = schemas.WriteStreamChunk(ctx, chatDeltaChunk("two-a"))
						close(stop)
						<-stopped
					}
					return chunk, nil
				})

			wire := s.frames(t)
			want := framesOf(t, route, true, chatDeltaChunk("one-a"), chatDeltaChunk("one"), chatDeltaChunk("two-a"), chatDeltaChunk("two"), chatFinishChunk())
			assert.Equal(t, want, wire, "nothing written through the finished call's context reaches the client")
			assert.True(t, wroteInFirstCall)
			assert.False(t, staleDuringSecondCall, "a finished call's context does not write during a later call")
			assert.False(t, staleOnHookGoroutine, "not even on the goroutine running the later call")
			assert.True(t, wrote, "the running call's own write succeeds while a finished call's context is in use")
			assert.Positive(t, staleAttempts)
			assert.Zero(t, staleWrites)
			assert.Equal(t, wire, withoutHeartbeats(s.tap.snapshot()))
		})
	}
}

// Each plugin's hook call has its own writer. Once one plugin's call has returned, its
// context cannot write while the next plugin's hook runs on the same chunk, and the next
// plugin's own write is unaffected.
func TestStreamChunkWriterIsPerPluginHookCall(t *testing.T) {
	route := chunkWriterRoutes[0]
	var firstPluginCtx *schemas.BifrostContext
	var wroteFirst, staleElsewhere, staleOnHookGoroutine, wroteSecond bool
	// Stream chunk hooks run in reverse load order, so "first" is loaded last.
	second := &chunkWriterPlugin{name: "second", onChunk: func(ctx *schemas.BifrostContext, chunk *schemas.BifrostStreamChunk) (*schemas.BifrostStreamChunk, error) {
		if chunkText(chunk) == "one" {
			staleElsewhere = writeElsewhere(firstPluginCtx, "stale")
			staleOnHookGoroutine = schemas.WriteStreamChunk(firstPluginCtx, chatDeltaChunk("stale"))
			wroteSecond = schemas.WriteStreamChunk(ctx, chatDeltaChunk("from-second"))
		}
		return chunk, nil
	}}
	first := &chunkWriterPlugin{name: "first", onChunk: func(ctx *schemas.BifrostContext, chunk *schemas.BifrostStreamChunk) (*schemas.BifrostStreamChunk, error) {
		if chunkText(chunk) == "one" {
			firstPluginCtx = ctx
			wroteFirst = schemas.WriteStreamChunk(ctx, chatDeltaChunk("from-first"))
		}
		return chunk, nil
	}}
	s := startPluginChunkWriterStream(t, route, []*schemas.BifrostStreamChunk{chatDeltaChunk("one"), chatFinishChunk()},
		[]*chunkWriterPlugin{second, first}, nil)

	wire := s.frames(t)
	want := framesOf(t, route, true, chatDeltaChunk("from-first"), chatDeltaChunk("from-second"), chatDeltaChunk("one"), chatFinishChunk())
	assert.Equal(t, want, wire)
	assert.True(t, wroteFirst)
	assert.True(t, wroteSecond)
	assert.False(t, staleElsewhere, "the first plugin's context is closed once its call has returned")
	assert.False(t, staleOnHookGoroutine)
	assert.Equal(t, wire, withoutHeartbeats(s.tap.snapshot()))
}
