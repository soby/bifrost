package lib

import (
	"sync"

	"github.com/maximhq/bifrost/core/schemas"
	"github.com/valyala/fasthttp"
)

// TransportPostHookSlot hands the deferred transport post-hooks of a streamed response
// from TransportInterceptorMiddleware, which builds them on the handler goroutine once the
// handler has returned, to the stream's producer goroutine, which runs them when the stream
// ends. The middleware resolves the slot exactly once on every path out of it: with the
// completer, or with nil when there are no post-hooks to run. The producer waits for that
// resolution, so the post-hooks are never skipped because a short stream ended first.
type TransportPostHookSlot struct {
	once      sync.Once
	resolved  chan struct{}
	completer func() ([]schemas.PluginLogEntry, error)

	// How the request stood when the producer stopped writing to the client; see
	// CaptureRequestState. Written and read on the producer goroutine only.
	stateCaptured bool
	requestEnded  bool
}

// NewTransportPostHookCompleterSlot allocates the slot for a streamed response and stores it
// on ctx for the middleware to resolve. It returns nil when the middleware did not mark the
// request, which means no transport plugin is loaded: there is nothing to run and nothing to
// wait for. The stream's producer keeps the returned pointer, since ctx is recycled once the
// handler returns.
func NewTransportPostHookCompleterSlot(ctx *fasthttp.RequestCtx) *TransportPostHookSlot {
	if expected, _ := ctx.UserValue(schemas.BifrostContextKeyTransportPostHookCompleter).(bool); !expected {
		return nil
	}
	slot := &TransportPostHookSlot{resolved: make(chan struct{})}
	ctx.SetUserValue(schemas.BifrostContextKeyTransportPostHookCompleter, slot)
	return slot
}

// Resolve publishes the completer, or nil for "no post-hooks". Only the first call has an
// effect. Safe on a nil slot.
func (s *TransportPostHookSlot) Resolve(completer func() ([]schemas.PluginLogEntry, error)) {
	if s == nil {
		return
	}
	s.once.Do(func() {
		s.completer = completer
		close(s.resolved)
	})
}

// Wait blocks until the middleware has resolved the slot and returns the completer, nil when
// there are no post-hooks to run. A nil slot returns nil at once.
func (s *TransportPostHookSlot) Wait() func() ([]schemas.PluginLogEntry, error) {
	if s == nil {
		return nil
	}
	<-s.resolved
	return s.completer
}

// CaptureRequestState records, the first time it is called, whether the request had already
// ended on its own: its context was done (the client disconnected, its deadline passed, a
// caller cancelled it) or the client's stream was closed under the producer by a failed
// write. The producer calls it before it cancels the request itself, which it does to stop
// the provider once it has nothing more to write, and again when its teardown starts. The
// post-hooks are then told about the request's own end, not about that cancel: a stream
// whose last frame was written is not reported to them as cancelled. Safe on a nil slot.
func (s *TransportPostHookSlot) CaptureRequestState(ctx *schemas.BifrostContext, reader *SSEStreamReader) {
	if s == nil || s.stateCaptured {
		return
	}
	s.stateCaptured = true
	s.requestEnded = (ctx != nil && ctx.Err() != nil) || (reader != nil && reader.Closed())
}

// MarkUndelivered records that the transport could not deliver what the hooks released: it
// failed to render or write a chunk that had passed every hook, for a reason other than the
// client going away. No hook can see that happen, so the post-hooks are told the same way
// they are told about a client that left: their context is cancelled. It overrides a state
// captured earlier. Safe on a nil slot.
func (s *TransportPostHookSlot) MarkUndelivered() {
	if s == nil {
		return
	}
	s.stateCaptured = true
	s.requestEnded = true
}

// RequestEnded reports what CaptureRequestState or MarkUndelivered recorded. captured is false when the
// producer never called it; the request's context then speaks for itself.
func (s *TransportPostHookSlot) RequestEnded() (ended, captured bool) {
	if s == nil {
		return false, false
	}
	return s.requestEnded, s.stateCaptured
}

// RunTransportPostHookCompleter runs the deferred transport post-hooks once a stream ends and
// returns their logs. It waits for the middleware to resolve the slot. A post-hook error
// cannot reach the caller any more; it is logged. A nil slot runs nothing.
func RunTransportPostHookCompleter(slot *TransportPostHookSlot, logger schemas.Logger) []schemas.PluginLogEntry {
	completer := slot.Wait()
	if completer == nil {
		return nil
	}
	logs, err := completer()
	if err != nil && logger != nil {
		logger.Warn("transport post-hook failed after the stream ended: %v", err)
	}
	return logs
}
