package schemas

import "context"

// StreamChunkWriter writes one additional chunk to the client's stream, now. The HTTP
// transport hands one to each HTTPTransportStreamChunkHook call, under
// BifrostContextKeyStreamChunkWriter on the context that call receives, so a plugin that
// rewrites a streamed response can send the client more than one chunk for one provider
// chunk, in order: for example content it passes through followed by a replacement it
// produced.
//
// The chunk takes the same path as a chunk the hook returns: the route's own conversion
// and framing, its own stream event, flushed on its own, and seen by the stream frame tap
// (BifrostContextKeyStreamFrameTap). It is written before the call returns, so chunks
// reach the client in call order, ahead of the chunk the hook returns, and a hook can
// release a chunk first and do slow work afterwards. It does not pass through the stream
// chunk hook of any other plugin.
//
// It reports whether the chunk was written. It writes nothing and reports false when the
// chunk is nil or carries a BifrostError (to end the stream with an error, return it from
// the hook), when the route produces no event for the chunk, and when the client has
// gone. Once the client has gone the stream ends when the hook returns, whatever the hook
// returns.
//
// A chunk written before the hook returns an error stays written: the error event follows
// it and ends the stream.
//
// The writer belongs to one hook call: every call of every plugin's hook gets its own.
// Only the context passed to that call carries it: the request's other contexts,
// ctx.Root() included, do not, and WriteStreamChunk reports false on them. Once that call
// has returned or panicked its writer writes nothing and reports false, from any
// goroutine and for the rest of the request, later hook calls included; it has no effect
// on the writer of a later call. Writes through one call's context are serialized: one
// that arrives while another is in progress waits for it.
type StreamChunkWriter func(chunk *BifrostStreamChunk) bool

// WriteStreamChunk writes chunk to the client's stream through the request's
// StreamChunkWriter and reports whether it was written. Without a writer on ctx (the
// request is not a streaming response of the HTTP transport) it reports false.
func WriteStreamChunk(ctx context.Context, chunk *BifrostStreamChunk) bool {
	if ctx == nil {
		return false
	}
	write, _ := ctx.Value(BifrostContextKeyStreamChunkWriter).(StreamChunkWriter)
	if write == nil {
		return false
	}
	return write(chunk)
}
