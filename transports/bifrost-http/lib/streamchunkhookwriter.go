package lib

import (
	"sync"

	"github.com/maximhq/bifrost/core/schemas"
)

// InterceptStreamChunk runs a stream's chunk interceptor on chunk for the stream's
// producer. write is the producer's own "convert, frame and send one chunk" step; an
// interceptor that is a StreamChunkWriterInterceptor offers it to its hooks as their
// schemas.StreamChunkWriter, one hook call at a time (see streamChunkHookCall). Any other
// interceptor is simply run, with no writer offered.
//
// write shares the producer's state. That is safe because the two never overlap: write
// runs only inside a hook call, while the producer is inside this function, and a write
// still in progress when a hook call ends is waited for before the call is left.
func InterceptStreamChunk(interceptor StreamChunkInterceptor, ctx *schemas.BifrostContext, req *schemas.HTTPRequest, chunk *schemas.BifrostStreamChunk, write func(chunk *schemas.BifrostStreamChunk) bool) (*schemas.BifrostStreamChunk, error) {
	if withWriter, ok := interceptor.(StreamChunkWriterInterceptor); ok && write != nil {
		return withWriter.InterceptChunkWithWriter(ctx, req, chunk, write)
	}
	return interceptor.InterceptChunk(ctx, req, chunk)
}

// streamChunkHookCall is the writer of one stream chunk hook call: open for exactly that
// call and closed for good when it returns or panics. Each call gets its own, so a context
// kept from a finished call reaches only a closed one: it can never write again, and it
// shares nothing with a later call, so it cannot delay or fail that call's writes.
//
// mu serializes the writes of this one call and is held across a write, which can wait on
// the client. Only this call's own context reaches it. close takes it once, when the hook
// call ends, so it waits for a write the hook left running on another goroutine; that
// write ends when the client reads the frame or the stream reader is closed, exactly as a
// write made by the producer itself would, so it cannot outlast the stream.
type streamChunkHookCall struct {
	write func(chunk *schemas.BifrostStreamChunk) bool

	mu     sync.Mutex
	closed bool
}

// writeChunk is the schemas.StreamChunkWriter the hook call receives.
func (c *streamChunkHookCall) writeChunk(chunk *schemas.BifrostStreamChunk) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return false
	}
	return c.write(chunk)
}

func (c *streamChunkHookCall) close() {
	c.mu.Lock()
	c.closed = true
	c.mu.Unlock()
}
