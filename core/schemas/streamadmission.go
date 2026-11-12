package schemas

import (
	"context"
	"sync"
)

// StreamAdmissionWait is the per-request signal a plugin uses to tell the HTTP
// transport that a streaming request is being held before it reaches a provider,
// for example while the plugin waits on an external lookup. The transport stores
// one under BifrostContextKeyStreamAdmissionWait only for streaming routes whose
// wire format can carry SSE comment lines; on other routes and for non-streaming
// requests the key is absent and the wait is silent.
type StreamAdmissionWait struct {
	once      sync.Once
	requested chan struct{}
}

// NewStreamAdmissionWait returns an unsignalled wait.
func NewStreamAdmissionWait() *StreamAdmissionWait {
	return &StreamAdmissionWait{requested: make(chan struct{})}
}

// Begin reports that the request has started waiting. The transport then commits
// a 200 event stream at once, so its stream heartbeat keeps the client connection
// alive during the wait, and delivers any later failure as an in-stream error
// event instead of an HTTP error status. Only the first call has an effect; later
// calls, and calls on a nil wait, do nothing.
func (w *StreamAdmissionWait) Begin() {
	if w == nil {
		return
	}
	w.once.Do(func() { close(w.requested) })
}

// Requested is closed by the first Begin call. On a nil wait it never fires.
func (w *StreamAdmissionWait) Requested() <-chan struct{} {
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
