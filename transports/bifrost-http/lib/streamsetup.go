package lib

import (
	"fmt"
	"runtime/debug"

	"github.com/maximhq/bifrost/core/schemas"
)

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
// plugin signals that it is holding the request (schemas.StreamAdmissionWait)
// does the handler commit a 200 event stream before setup finishes. Such a
// stream omits the provider response headers and routed-identity headers, which
// do not exist yet; the stream's heartbeat keeps the connection alive and
// detects a client that leaves, and a setup error then becomes the route's
// ordinary in-stream error event.
//
// StreamSetup never touches the fasthttp RequestCtx: after an early commit the
// handler returns and fasthttp may recycle the RequestCtx while setup is still
// running.
type StreamSetup struct {
	wait   *schemas.StreamAdmissionWait
	result chan streamSetupResult
}

// StartStreamSetup runs getStream. When canCommitEarly is true it offers the
// request a schemas.StreamAdmissionWait under
// schemas.BifrostContextKeyStreamAdmissionWait and runs getStream on its own
// goroutine, so the handler can return while a plugin still holds the request.
// When it is false (a route whose clients reject SSE comments, or a non-SSE wire
// format) no signal is offered, getStream runs on the caller's goroutine and
// Await returns its result.
//
// A panic in getStream on the setup goroutine is captured and re-raised by Await
// on the handler goroutine, so the transport's recovery middleware still answers
// it, unless the response was already committed; then it is logged and the
// stream ends with an error event.
func StartStreamSetup(
	bifrostCtx *schemas.BifrostContext,
	canCommitEarly bool,
	getStream func() (chan *schemas.BifrostStreamChunk, *schemas.BifrostError),
) *StreamSetup {
	setup := &StreamSetup{result: make(chan streamSetupResult, 1)}
	if !canCommitEarly || bifrostCtx == nil {
		var result streamSetupResult
		result.stream, result.err = getStream()
		setup.result <- result
		return setup
	}
	setup.wait = schemas.NewStreamAdmissionWait()
	bifrostCtx.SetValue(schemas.BifrostContextKeyStreamAdmissionWait, setup.wait)
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

// Await blocks until setup finishes or a plugin asks for the response to be
// committed early. committed=false returns setup's own result. committed=true
// means setup is still running; the caller commits the event stream and, on its
// producer goroutine, obtains the stream with AwaitCommitted.
func (setup *StreamSetup) Await() (stream chan *schemas.BifrostStreamChunk, bifrostErr *schemas.BifrostError, committed bool) {
	finished := func(result streamSetupResult) (chan *schemas.BifrostStreamChunk, *schemas.BifrostError, bool) {
		if result.panicked != nil {
			panic(result.panicked)
		}
		return result.stream, result.err, false
	}
	select {
	case result := <-setup.result:
		return finished(result)
	case <-setup.wait.Requested():
		// Setup that has already finished keeps its ordinary response.
		select {
		case result := <-setup.result:
			return finished(result)
		default:
			return nil, nil, true
		}
	}
}

// AwaitCommitted waits for setup after an early commit. A setup error is
// returned as a one-chunk closed stream holding that error, tagged with
// requestType when it carries none, so the caller's ordinary chunk loop writes
// it in the route's own error-event shape and suppresses the success marker.
func (setup *StreamSetup) AwaitCommitted(requestType schemas.RequestType) chan *schemas.BifrostStreamChunk {
	result := <-setup.result
	if result.panicked != nil {
		if logger != nil {
			logger.Error(fmt.Sprintf("recovered from panic in stream setup after early commit: %s\n%s", PanicSummary(result.panicked), result.stack))
		}
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
}

// ErrorStream returns a closed one-chunk stream holding bifrostErr. It sanitizes
// the error exactly as the pre-commit HTTP error path does
// (SanitizeBifrostErrorForClient), so committing early never reveals detail that
// the uncommitted response would have hidden.
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
