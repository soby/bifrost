package schemas

import (
	"context"
	"sync"
	"time"
)

// StreamAdmissionWaitOptions tells the HTTP transport how to hold a streaming
// response open while a plugin deliberately delays the request before it reaches
// a provider (for example, while the previous step of the same conversation is
// still running).
type StreamAdmissionWaitOptions struct {
	// CommitAfter is how long the transport keeps the response uncommitted after
	// the wait begins; zero commits at once. While uncommitted, a failure is still
	// an ordinary HTTP error status. Once it elapses the transport commits a 200
	// event stream and any later failure is delivered as an in-stream error event.
	CommitAfter time.Duration
	// CommentInterval is how often the transport writes a ": waiting" comment
	// line on a committed stream while the wait continues.
	CommentInterval time.Duration
}

// StreamAdmissionWait is the per-request signal a plugin uses to tell the HTTP
// transport that a streaming request is waiting before admission. The transport
// stores one under BifrostContextKeyStreamAdmissionWait only for streaming routes
// that can carry SSE comment lines; on other routes and for non-streaming
// requests the key is absent and the wait is silent.
type StreamAdmissionWait struct {
	once      sync.Once
	requested chan StreamAdmissionWaitOptions
}

// NewStreamAdmissionWait returns an unsignalled wait.
func NewStreamAdmissionWait() *StreamAdmissionWait {
	return &StreamAdmissionWait{requested: make(chan StreamAdmissionWaitOptions, 1)}
}

// Begin reports that the request has started waiting. Only the first call has an
// effect; later calls, and calls on a nil wait, do nothing. Options with a
// negative CommitAfter or a non-positive CommentInterval are ignored, so the wait
// stays silent.
func (w *StreamAdmissionWait) Begin(options StreamAdmissionWaitOptions) {
	if w == nil || options.CommitAfter < 0 || options.CommentInterval <= 0 {
		return
	}
	w.once.Do(func() { w.requested <- options })
}

// Requested delivers the options of the first Begin call. It never closes.
func (w *StreamAdmissionWait) Requested() <-chan StreamAdmissionWaitOptions {
	if w == nil {
		return nil
	}
	return w.requested
}

// StreamAdmissionWaitFrom returns the request's wait signal, or nil when the
// transport offers none.
func StreamAdmissionWaitFrom(ctx context.Context) *StreamAdmissionWait {
	if ctx == nil {
		return nil
	}
	wait, _ := ctx.Value(BifrostContextKeyStreamAdmissionWait).(*StreamAdmissionWait)
	return wait
}
