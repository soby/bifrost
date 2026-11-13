package handlers

import (
	"context"
	"errors"
	"io"
	"sync"
	"testing"

	"github.com/maximhq/bifrost/core/schemas"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/valyala/fasthttp"
)

// A plugin that rewrites a streamed response may need to send the client more than one
// chunk for one provider chunk, in order: for example content it passes through followed
// by a replacement it produced. Its stream chunk hook writes the additional chunks with
// schemas.WriteStreamChunk; these tests run that through the /v1 chat completions stream.

// chunkWriterPlugin is a transport plugin whose stream chunk hook is supplied by the test.
type chunkWriterPlugin struct {
	fakeResponseHeadersPlugin
	onChunk func(ctx *schemas.BifrostContext, chunk *schemas.BifrostStreamChunk) (*schemas.BifrostStreamChunk, error)
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

const (
	heartbeatFrame = ": heartbeat\n"
	doneFrame      = "data: [DONE]\n\n"
)

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

func finishChunk() *schemas.BifrostStreamChunk {
	chunk := contentChunk("")
	chunk.BifrostChatResponse.Choices[0].FinishReason = schemas.Ptr("stop")
	return chunk
}

func usageChunk(text string) *schemas.BifrostStreamChunk {
	chunk := contentChunk(text)
	chunk.BifrostChatResponse.Usage = &schemas.BifrostLLMUsage{PromptTokens: 3, CompletionTokens: 4, TotalTokens: 7}
	return chunk
}

func chunkContent(chunk *schemas.BifrostStreamChunk) string {
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

// chatFrames is the exact frames the /v1 chat stream writes for chunks: one event per
// chunk in the Chat Completions wire contract, and its usage event when requested.
func chatFrames(t *testing.T, includeUsage bool, chunks ...*schemas.BifrostStreamChunk) []string {
	t.Helper()
	var frames []string
	for _, chunk := range chunks {
		primary, usage, err := marshalChatCompletionStreamEvents(chunk.BifrostChatResponse, includeUsage)
		require.NoError(t, err)
		if len(primary) > 0 {
			frames = append(frames, "data: "+string(primary)+"\n\n")
		}
		if len(usage) > 0 {
			frames = append(frames, "data: "+string(usage)+"\n\n")
		}
	}
	return frames
}

// chunkWriterStream is one /v1 chat stream with the plugin's hook on it.
type chunkWriterStream struct {
	body      io.Reader
	closer    io.Closer
	tap       *frameRecorder
	ctx       *schemas.BifrostContext
	cancelled <-chan struct{}
}

// startChunkWriterStream serves chunks as a /v1 chat completions stream. hook is the
// plugin's stream chunk hook; nil serves the stream without any transport plugin.
func startChunkWriterStream(t *testing.T, includeUsage bool, chunks []*schemas.BifrostStreamChunk, hook func(*schemas.BifrostContext, *schemas.BifrostStreamChunk) (*schemas.BifrostStreamChunk, error)) *chunkWriterStream {
	t.Helper()
	return startTappedChunkWriterStream(t, includeUsage, chunks, hook, nil)
}

// startTappedChunkWriterStream is startChunkWriterStream with onFrame run for each frame
// on the stream's send path.
func startTappedChunkWriterStream(t *testing.T, includeUsage bool, chunks []*schemas.BifrostStreamChunk, hook func(*schemas.BifrostContext, *schemas.BifrostStreamChunk) (*schemas.BifrostStreamChunk, error), onFrame func(string)) *chunkWriterStream {
	t.Helper()
	SetLogger(&mockLogger{})
	config := responseHeadersTestConfig(t)
	if hook != nil {
		config = responseHeadersTestConfig(t, &chunkWriterPlugin{fakeResponseHeadersPlugin: fakeResponseHeadersPlugin{name: "chunk-writer"}, onChunk: hook})
	}

	base, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	s := &chunkWriterStream{tap: &frameRecorder{onFrame: onFrame}, cancelled: base.Done()}
	s.ctx = schemas.NewBifrostContext(base, schemas.NoDeadline)
	s.ctx.SetValue(schemas.BifrostContextKeyStreamFrameTap, io.Writer(s.tap))
	getStream := func() (chan *schemas.BifrostStreamChunk, *schemas.BifrostError) {
		stream := make(chan *schemas.BifrostStreamChunk, len(chunks))
		for _, chunk := range chunks {
			stream <- chunk
		}
		close(stream)
		return stream, nil
	}
	reqCtx := &fasthttp.RequestCtx{}
	(&CompletionHandler{config: config}).handleStreamingResponseWithChatUsage(reqCtx, s.ctx, schemas.ChatCompletionStreamRequest, includeUsage, getStream, cancel)
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

// Chunks the hook writes reach the client in call order, ahead of the chunk the hook
// returns, each as its own event, and each before the hook goes on: the hook here does
// not continue until the client has read the first one.
func TestChatStreamChunkWriterWritesAheadOfTheReturnedChunk(t *testing.T) {
	clientReadFirst := make(chan struct{})
	var wrote []bool
	s := startChunkWriterStream(t, false,
		[]*schemas.BifrostStreamChunk{contentChunk("one"), contentChunk("two"), finishChunk()},
		func(ctx *schemas.BifrostContext, chunk *schemas.BifrostStreamChunk) (*schemas.BifrostStreamChunk, error) {
			if chunkContent(chunk) != "two" {
				return chunk, nil
			}
			wrote = append(wrote, schemas.WriteStreamChunk(ctx, contentChunk("two-a")))
			<-clientReadFirst
			wrote = append(wrote, schemas.WriteStreamChunk(ctx, contentChunk("two-b")))
			return chunk, nil
		})

	var wire []string
	for range 2 {
		frame, ok := s.nextFrame(t)
		require.True(t, ok)
		wire = append(wire, frame)
	}
	require.Equal(t, chatFrames(t, false, contentChunk("one"), contentChunk("two-a")), wire,
		"the first written chunk reaches the client while the hook is still running")
	close(clientReadFirst)
	wire = append(wire, s.frames(t)...)

	want := append(chatFrames(t, false, contentChunk("one"), contentChunk("two-a"), contentChunk("two-b"), contentChunk("two"), finishChunk()), doneFrame)
	assert.Equal(t, want, wire)
	assert.Equal(t, wire, withoutHeartbeats(s.tap.snapshot()), "the frame tap sees every frame as written, in order")
	assert.Equal(t, []bool{true, true}, wrote)
}

// A written chunk gets the route's own conversion. On the /v1 chat stream that is the
// stream_options.include_usage contract: usage a chunk carries is left out unless the
// caller asked for it, and is then its own event after the chunk.
func TestChatStreamChunkWriterAppliesTheChatUsageContract(t *testing.T) {
	for _, includeUsage := range []bool{false, true} {
		name := "usage not requested"
		if includeUsage {
			name = "usage requested"
		}
		t.Run(name, func(t *testing.T) {
			s := startChunkWriterStream(t, includeUsage,
				[]*schemas.BifrostStreamChunk{contentChunk("one"), finishChunk()},
				func(ctx *schemas.BifrostContext, chunk *schemas.BifrostStreamChunk) (*schemas.BifrostStreamChunk, error) {
					if chunkContent(chunk) == "one" {
						schemas.WriteStreamChunk(ctx, usageChunk("written"))
					}
					return chunk, nil
				})

			wire := s.frames(t)
			written := chatFrames(t, includeUsage, usageChunk("written"))
			if includeUsage {
				require.Len(t, written, 2, "the chunk and its usage are separate events")
				assert.Contains(t, written[0], `"usage":null`)
				assert.Contains(t, written[1], `"total_tokens":7`)
			} else {
				require.Len(t, written, 1)
				assert.NotContains(t, written[0], "usage")
			}
			want := append(append(written, chatFrames(t, includeUsage, contentChunk("one"), finishChunk())...), doneFrame)
			assert.Equal(t, want, wire)
			assert.Equal(t, wire, withoutHeartbeats(s.tap.snapshot()))
		})
	}
}

// A hook that returns no chunk still gets the chunks it wrote to the client.
func TestChatStreamChunkWriterWithNoReturnedChunk(t *testing.T) {
	s := startChunkWriterStream(t, false,
		[]*schemas.BifrostStreamChunk{contentChunk("one"), contentChunk("two"), finishChunk()},
		func(ctx *schemas.BifrostContext, chunk *schemas.BifrostStreamChunk) (*schemas.BifrostStreamChunk, error) {
			if chunkContent(chunk) != "two" {
				return chunk, nil
			}
			schemas.WriteStreamChunk(ctx, contentChunk("replacement"))
			return nil, nil
		})

	wire := s.frames(t)
	assert.Equal(t, append(chatFrames(t, false, contentChunk("one"), contentChunk("replacement"), finishChunk()), doneFrame), wire)
	assert.Equal(t, wire, withoutHeartbeats(s.tap.snapshot()))
}

// A chunk written ahead of the stream's last chunk precedes it and the [DONE] marker.
func TestChatStreamChunkWriterAheadOfTheTerminalChunk(t *testing.T) {
	s := startChunkWriterStream(t, false,
		[]*schemas.BifrostStreamChunk{contentChunk("one"), finishChunk()},
		func(ctx *schemas.BifrostContext, chunk *schemas.BifrostStreamChunk) (*schemas.BifrostStreamChunk, error) {
			if isFinishChunk(chunk) {
				schemas.WriteStreamChunk(ctx, contentChunk("tail"))
			}
			return chunk, nil
		})

	wire := s.frames(t)
	assert.Equal(t, append(chatFrames(t, false, contentChunk("one"), contentChunk("tail"), finishChunk()), doneFrame), wire)
	assert.Equal(t, wire, withoutHeartbeats(s.tap.snapshot()))
}

// A chunk written before the hook returns an error stays written. The error event follows
// it and ends the stream: the hook is not called again and nothing else is written, the
// [DONE] marker included.
func TestChatStreamChunkWriterThenHookError(t *testing.T) {
	calls := 0
	s := startChunkWriterStream(t, false,
		[]*schemas.BifrostStreamChunk{contentChunk("one"), contentChunk("two"), finishChunk()},
		func(ctx *schemas.BifrostContext, chunk *schemas.BifrostStreamChunk) (*schemas.BifrostStreamChunk, error) {
			calls++
			if chunkContent(chunk) != "two" {
				return chunk, nil
			}
			schemas.WriteStreamChunk(ctx, contentChunk("kept"))
			return nil, errors.New("stop here")
		})

	wire := s.frames(t)
	want := append(chatFrames(t, false, contentChunk("one"), contentChunk("kept")),
		"event: error\ndata: {\"error\":\"failed to intercept chunk with plugin chunk-writer: stop here\"}\n\n")
	assert.Equal(t, want, wire)
	assert.Equal(t, wire, withoutHeartbeats(s.tap.snapshot()))
	assert.Equal(t, 2, calls)
	<-s.cancelled
}

// The writer writes response chunks during a hook call and nothing else: it refuses a nil
// chunk and an error chunk, and it is inert once the hook has returned.
func TestChatStreamChunkWriterRefusals(t *testing.T) {
	var refusedNil, refusedError bool
	s := startChunkWriterStream(t, false, []*schemas.BifrostStreamChunk{contentChunk("one")},
		func(ctx *schemas.BifrostContext, chunk *schemas.BifrostStreamChunk) (*schemas.BifrostStreamChunk, error) {
			refusedNil = !schemas.WriteStreamChunk(ctx, nil)
			refusedError = !schemas.WriteStreamChunk(ctx, truncationErrorChunk())
			return chunk, nil
		})

	wire := s.frames(t)
	assert.Equal(t, append(chatFrames(t, false, contentChunk("one")), doneFrame), wire)
	assert.True(t, refusedNil, "a nil chunk is not written")
	assert.True(t, refusedError, "an error chunk is not written")
	assert.False(t, schemas.WriteStreamChunk(s.ctx, contentChunk("late")), "the request context carries no writer")
	assert.Equal(t, wire, withoutHeartbeats(s.tap.snapshot()))
}

// A write that finds the client gone reports false, and the stream ends when the hook
// returns: the returned chunk is not written and the hook is not called again.
func TestChatStreamChunkWriterClientGone(t *testing.T) {
	calls := 0
	var wrote, wroteAgain bool
	var s *chunkWriterStream
	started := make(chan struct{})
	hook := func(ctx *schemas.BifrostContext, chunk *schemas.BifrostStreamChunk) (*schemas.BifrostStreamChunk, error) {
		<-started
		calls++
		_ = s.closer.Close() // the client goes away
		wrote = schemas.WriteStreamChunk(ctx, contentChunk("unsent"))
		wroteAgain = schemas.WriteStreamChunk(ctx, contentChunk("unsent"))
		return chunk, nil
	}
	s = startChunkWriterStream(t, false, []*schemas.BifrostStreamChunk{contentChunk("one"), contentChunk("two"), finishChunk()}, hook)
	close(started)

	assert.Empty(t, s.frames(t))
	<-s.cancelled
	assert.False(t, wrote)
	assert.False(t, wroteAgain)
	assert.Equal(t, 1, calls)
	assert.Empty(t, withoutHeartbeats(s.tap.snapshot()))
}

// A hook that never writes a chunk leaves the stream exactly as it is without any plugin.
func TestChatStreamChunkWriterUnusedLeavesTheStreamUnchanged(t *testing.T) {
	chunks := func() []*schemas.BifrostStreamChunk {
		return []*schemas.BifrostStreamChunk{contentChunk("one"), contentChunk("two"), finishChunk()}
	}
	plain := startChunkWriterStream(t, false, chunks(), nil)
	hooked := startChunkWriterStream(t, false, chunks(),
		func(_ *schemas.BifrostContext, chunk *schemas.BifrostStreamChunk) (*schemas.BifrostStreamChunk, error) {
			return chunk, nil
		})

	want := plain.frames(t)
	assert.Equal(t, append(chatFrames(t, false, chunks()...), doneFrame), want)
	assert.Equal(t, want, hooked.frames(t))
	assert.Equal(t, want, withoutHeartbeats(hooked.tap.snapshot()))
	assert.Nil(t, plain.ctx.Value(schemas.BifrostContextKeyStreamChunkWriter), "no writer is offered to a stream no hook runs on")
}

// usageOnlyChunk holds usage and no choice: the /v1 chat stream has no event for it unless
// the caller asked for usage.
func usageOnlyChunk() *schemas.BifrostStreamChunk {
	chunk := usageChunk("")
	chunk.BifrostChatResponse.Choices = nil
	return chunk
}

// A chunk the route has no event for is not written: the writer reports false, with the
// client present or gone, and nothing reaches the stream.
func TestChatStreamChunkWriterReportsFalseWhenTheRouteHasNoEvent(t *testing.T) {
	primary, usage, err := marshalChatCompletionStreamEvents(usageOnlyChunk().BifrostChatResponse, false)
	require.NoError(t, err)
	require.Empty(t, primary)
	require.Empty(t, usage)

	t.Run("client present", func(t *testing.T) {
		var wroteNoEvent, wroteNext bool
		s := startChunkWriterStream(t, false, []*schemas.BifrostStreamChunk{contentChunk("one"), finishChunk()},
			func(ctx *schemas.BifrostContext, chunk *schemas.BifrostStreamChunk) (*schemas.BifrostStreamChunk, error) {
				if chunkContent(chunk) == "one" {
					wroteNoEvent = schemas.WriteStreamChunk(ctx, usageOnlyChunk())
					wroteNext = schemas.WriteStreamChunk(ctx, contentChunk("next"))
				}
				return chunk, nil
			})

		wire := s.frames(t)
		assert.Equal(t, append(chatFrames(t, false, contentChunk("next"), contentChunk("one"), finishChunk()), doneFrame), wire)
		assert.False(t, wroteNoEvent)
		assert.True(t, wroteNext, "the stream goes on after a chunk with no event")
		assert.Equal(t, wire, withoutHeartbeats(s.tap.snapshot()))
	})

	t.Run("client gone", func(t *testing.T) {
		var wroteNoEvent bool
		var s *chunkWriterStream
		started := make(chan struct{})
		hook := func(ctx *schemas.BifrostContext, chunk *schemas.BifrostStreamChunk) (*schemas.BifrostStreamChunk, error) {
			<-started
			_ = s.closer.Close() // the client goes away
			wroteNoEvent = schemas.WriteStreamChunk(ctx, usageOnlyChunk())
			return chunk, nil
		}
		s = startChunkWriterStream(t, false, []*schemas.BifrostStreamChunk{contentChunk("one")}, hook)
		close(started)

		assert.Empty(t, s.frames(t))
		assert.False(t, wroteNoEvent)
	})
}

// A hook that panics leaves no usable writer behind: once the stream has ended, a write
// through the context that hook call received, or through the request context the
// post-hooks derive theirs from, reports false and touches nothing.
func TestChatStreamChunkWriterIsClosedAfterAHookPanic(t *testing.T) {
	var hookCtx *schemas.BifrostContext
	var wroteBeforePanic bool
	s := startChunkWriterStream(t, false, []*schemas.BifrostStreamChunk{contentChunk("one"), contentChunk("two")},
		func(ctx *schemas.BifrostContext, chunk *schemas.BifrostStreamChunk) (*schemas.BifrostStreamChunk, error) {
			hookCtx = ctx
			wroteBeforePanic = schemas.WriteStreamChunk(ctx, contentChunk("before"))
			panic("hook bug")
		})

	wire := s.frames(t)
	require.Len(t, wire, 2, "the written chunk, then the error frame the panic ends the stream with")
	assert.Equal(t, chatFrames(t, false, contentChunk("before"))[0], wire[0])
	assert.Contains(t, wire[1], "event: error\n")
	assert.True(t, wroteBeforePanic)
	<-s.cancelled

	// The stream is closed now. Neither context can write to it.
	assert.False(t, schemas.WriteStreamChunk(hookCtx, contentChunk("late")))
	assert.False(t, schemas.WriteStreamChunk(s.ctx, contentChunk("late")))
	assert.Equal(t, wire, withoutHeartbeats(s.tap.snapshot()))
}

// writeElsewhere writes a chunk through ctx from another goroutine and waits for the result.
func writeElsewhere(ctx *schemas.BifrostContext, text string) bool {
	result := make(chan bool)
	go func() { result <- schemas.WriteStreamChunk(ctx, contentChunk(text)) }()
	return <-result
}

// Only the hook call writes. Another goroutine holding the request context has no writer,
// during the call or after it. One the hook hands its own context to is part of the call:
// its write waits for the write in progress and follows it. After the call has returned
// that context writes nothing.
func TestChatStreamChunkWriterBelongsToTheHookCall(t *testing.T) {
	var hookCtx *schemas.BifrostContext
	var duringFromRequestCtx, wrote, wroteBeside bool
	beside := make(chan bool, 1)
	midWrite := chatFrames(t, false, contentChunk("two-a"))[0]
	s := startTappedChunkWriterStream(t, false,
		[]*schemas.BifrostStreamChunk{contentChunk("one"), contentChunk("two"), finishChunk()},
		func(ctx *schemas.BifrostContext, chunk *schemas.BifrostStreamChunk) (*schemas.BifrostStreamChunk, error) {
			if chunkContent(chunk) != "two" {
				return chunk, nil
			}
			hookCtx = ctx
			duringFromRequestCtx = writeElsewhere(ctx.Root(), "elsewhere")
			wrote = schemas.WriteStreamChunk(ctx, contentChunk("two-a"))
			wroteBeside = <-beside
			return chunk, nil
		},
		func(frame string) {
			// On the send path of the hook's own write: a second write through the
			// call's context starts while the first is in progress.
			if frame == midWrite {
				go func() { beside <- schemas.WriteStreamChunk(hookCtx, contentChunk("two-b")) }()
			}
		})

	wire := s.frames(t)
	want := append(chatFrames(t, false, contentChunk("one"), contentChunk("two-a"), contentChunk("two-b"), contentChunk("two"), finishChunk()), doneFrame)
	assert.Equal(t, want, wire)
	assert.True(t, wrote)
	assert.True(t, wroteBeside, "a write through the call's context waits for the one in progress")
	assert.False(t, duringFromRequestCtx, "the request context carries no writer during the call")
	assert.False(t, writeElsewhere(hookCtx, "late"), "the call's context is closed once the call has returned")
	assert.False(t, writeElsewhere(s.ctx, "late"), "the request context carries no writer after the call")
	assert.Equal(t, wire, withoutHeartbeats(s.tap.snapshot()))
}

// A context kept from one hook call never writes again: not while a later chunk's hook
// call is running, from any goroutine, and not while that call writes. The later call's
// own writes are unaffected by it.
func TestChatStreamChunkWriterOfAFinishedCallStaysClosedDuringLaterCalls(t *testing.T) {
	var firstCallCtx *schemas.BifrostContext
	var wroteInFirstCall, staleDuringSecondCall, staleOnHookGoroutine, wrote bool
	var staleAttempts, staleWrites int
	s := startChunkWriterStream(t, false,
		[]*schemas.BifrostStreamChunk{contentChunk("one"), contentChunk("two"), finishChunk()},
		func(ctx *schemas.BifrostContext, chunk *schemas.BifrostStreamChunk) (*schemas.BifrostStreamChunk, error) {
			switch chunkContent(chunk) {
			case "one":
				firstCallCtx = ctx
				wroteInFirstCall = schemas.WriteStreamChunk(ctx, contentChunk("one-a"))
			case "two":
				staleDuringSecondCall = writeElsewhere(firstCallCtx, "stale")
				staleOnHookGoroutine = schemas.WriteStreamChunk(firstCallCtx, contentChunk("stale"))

				// The first call's context is used over and over on another goroutine
				// for as long as this call's write takes.
				stop, stopped := make(chan struct{}), make(chan struct{})
				started := make(chan struct{})
				go func() {
					defer close(stopped)
					for first := true; ; first = false {
						staleAttempts++
						if schemas.WriteStreamChunk(firstCallCtx, contentChunk("stale")) {
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
				wrote = schemas.WriteStreamChunk(ctx, contentChunk("two-a"))
				close(stop)
				<-stopped
			}
			return chunk, nil
		})

	wire := s.frames(t)
	want := append(chatFrames(t, false, contentChunk("one-a"), contentChunk("one"), contentChunk("two-a"), contentChunk("two"), finishChunk()), doneFrame)
	assert.Equal(t, want, wire, "nothing written through the finished call's context reaches the client")
	assert.True(t, wroteInFirstCall)
	assert.False(t, staleDuringSecondCall, "a finished call's context does not write during a later call")
	assert.False(t, staleOnHookGoroutine, "not even on the goroutine running the later call")
	assert.True(t, wrote, "the running call's own write succeeds while a finished call's context is in use")
	assert.Positive(t, staleAttempts)
	assert.Zero(t, staleWrites)
	assert.Equal(t, wire, withoutHeartbeats(s.tap.snapshot()))
}
