package lib

import (
	"fmt"
	"runtime/debug"
	"time"

	"github.com/maximhq/bifrost/core/schemas"
)

// sseWaitingFrame is the comment line sent on a committed stream while a plugin
// still holds the request before admission. Like the heartbeat it is a bare
// comment line, so decoders that dispatch an empty event for a comment block
// (see SSEHeartbeatBareCommentLine) are unaffected.
var sseWaitingFrame = []byte(": waiting\n")

// SendWaitingComment writes one ": waiting" comment line. It follows
// SendHeartbeat's rules: nothing is written mid-line, and a closed reader (the
// client left) reports false.
func (r *SSEStreamReader) SendWaitingComment() bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	select {
	case <-r.closeCh:
		return false
	default:
	}
	if !r.atLineBoundary {
		return true
	}
	return r.sendLocked(sseWaitingFrame)
}

type streamSetupResult struct {
	stream   chan *schemas.BifrostStreamChunk
	err      *schemas.BifrostError
	panicked any
	stack    []byte
}

// StreamSetup runs a streaming request's setup (core's pre-hooks and the
// provider's stream start) while the handler decides whether to commit the
// response early.
//
// Ordinarily the handler waits for setup, exactly as it would calling it
// inline: a setup error is still an HTTP status with a JSON body, and the
// routing and provider response headers are installed after setup. Only when a
// plugin signals a deliberate pre-admission wait (schemas.StreamAdmissionWait)
// and the wait outlasts its CommitAfter does the handler commit a 200 event
// stream before setup finishes. Such a stream omits the provider response
// headers and routed-identity headers, which do not exist yet; it carries
// ": waiting" comments until setup ends, and a setup error then becomes the
// route's ordinary in-stream error event.
//
// StreamSetup never touches the fasthttp RequestCtx: after an early commit the
// handler returns and fasthttp may recycle the RequestCtx while setup is still
// running.
type StreamSetup struct {
	wait   *schemas.StreamAdmissionWait
	result chan streamSetupResult
	// options are those of the wait that caused an early commit; zero otherwise.
	options schemas.StreamAdmissionWaitOptions
}

// StartStreamSetup starts getStream on its own goroutine. When canCommitEarly
// is true it offers the request a schemas.StreamAdmissionWait under
// schemas.BifrostContextKeyStreamAdmissionWait before getStream runs; when it is
// false (a route whose clients reject SSE comments, or a non-SSE wire format) no
// signal is offered and Await always waits for setup.
//
// A panic in getStream is captured and re-raised by Await on the handler
// goroutine, so the transport's recovery middleware still answers it, unless the
// response was already committed; then it is logged and the stream ends with an
// error event.
func StartStreamSetup(
	bifrostCtx *schemas.BifrostContext,
	canCommitEarly bool,
	getStream func() (chan *schemas.BifrostStreamChunk, *schemas.BifrostError),
) *StreamSetup {
	setup := &StreamSetup{result: make(chan streamSetupResult, 1)}
	if canCommitEarly && bifrostCtx != nil {
		setup.wait = schemas.NewStreamAdmissionWait()
		bifrostCtx.SetValue(schemas.BifrostContextKeyStreamAdmissionWait, setup.wait)
	}
	go func() {
		var result streamSetupResult
		defer func() {
			if recovered := recover(); recovered != nil {
				result = streamSetupResult{panicked: recovered, stack: debug.Stack()}
			}
			setup.result <- result
		}()
		result.stream, result.err = getStream()
	}()
	return setup
}

// Await blocks until setup finishes or the response should be committed early.
// committed=false returns setup's own result. committed=true means setup is still
// running; the caller commits the event stream and, on its producer goroutine,
// obtains the stream with AwaitCommitted.
func (setup *StreamSetup) Await() (stream chan *schemas.BifrostStreamChunk, bifrostErr *schemas.BifrostError, committed bool) {
	var commit <-chan time.Time
	requested := setup.wait.Requested()
	for {
		select {
		case result := <-setup.result:
			if result.panicked != nil {
				panic(result.panicked)
			}
			return result.stream, result.err, false
		case options := <-requested:
			requested = nil
			setup.options = options
			timer := time.NewTimer(options.CommitAfter)
			defer timer.Stop()
			commit = timer.C
		case <-commit:
			return nil, nil, true
		}
	}
}

// AwaitCommitted waits for setup after an early commit, sending a waiting
// comment through send every CommentInterval. A send that reports false means the
// client left: onDisconnect runs once (it cancels the request, which ends the
// plugin's wait) and no further comments are sent, but AwaitCommitted still waits
// for setup so the caller can drain and clean up.
//
// A setup error is returned as a one-chunk closed stream holding that error,
// tagged with requestType when it carries none, so the caller's ordinary chunk
// loop writes it in the route's own error-event shape and suppresses the
// success marker.
func (setup *StreamSetup) AwaitCommitted(
	send func() bool,
	onDisconnect func(),
	requestType schemas.RequestType,
) chan *schemas.BifrostStreamChunk {
	ticker := time.NewTicker(setup.options.CommentInterval)
	defer ticker.Stop()
	ticks := ticker.C
	for {
		select {
		case result := <-setup.result:
			if result.panicked != nil {
				logger.Error(fmt.Sprintf("recovered from panic in stream setup after early commit: %T\n%s", result.panicked, result.stack))
				result.err = &schemas.BifrostError{
					IsBifrostError: true,
					Error:          &schemas.ErrorField{Message: ClientSafeInternalErrorMessage},
				}
			}
			if result.err != nil {
				return ErrorStream(result.err, requestType)
			}
			if result.stream == nil {
				return ErrorStream(&schemas.BifrostError{
					IsBifrostError: true,
					Error:          &schemas.ErrorField{Message: "streaming is not supported for this request type"},
				}, requestType)
			}
			return result.stream
		case <-ticks:
			if !send() {
				ticks = nil
				onDisconnect()
			}
		}
	}
}

// ErrorStream returns a closed one-chunk stream holding bifrostErr. It sanitizes the error exactly as the pre-commit HTTP error path
// does (SanitizeBifrostErrorForClient), so committing early never reveals detail
// that the uncommitted response would have hidden.
func ErrorStream(bifrostErr *schemas.BifrostError, requestType schemas.RequestType) chan *schemas.BifrostStreamChunk {
	bifrostErr = SanitizeBifrostErrorForClient(bifrostErr)
	if bifrostErr.ExtraFields.RequestType == "" {
		bifrostErr.ExtraFields.RequestType = requestType
	}
	stream := make(chan *schemas.BifrostStreamChunk, 1)
	stream <- &schemas.BifrostStreamChunk{BifrostError: bifrostErr}
	close(stream)
	return stream
}
