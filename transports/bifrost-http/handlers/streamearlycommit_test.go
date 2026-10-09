package handlers

import (
	"bufio"
	"context"
	"fmt"
	"net"
	"strings"
	"testing"
	"time"

	"github.com/maximhq/bifrost/core/schemas"
	"github.com/maximhq/bifrost/transports/bifrost-http/lib"
	"github.com/valyala/fasthttp"
)

// Early SSE commitment (fork-only, lib.StreamSetup): a plugin that holds a
// streaming request before admission signals schemas.StreamAdmissionWait. Until
// the wait outlasts CommitAfter the response stays uncommitted, so a failure is
// an ordinary HTTP status. After that the transport commits a 200 event stream,
// writes ": waiting" comments, and delivers a later failure as an error event.

type earlyCommitResponse struct {
	status  int
	headers string
	body    string
}

const earlyCommitTestTimeout = 10 * time.Second

// serveEarlyCommit runs handleStreamingResponse with the given setup function
// and returns what the client received. setup runs as core would: it receives
// the request's BifrostContext, which carries the transport's wait signal.
func serveEarlyCommit(
	t *testing.T,
	requestType schemas.RequestType,
	setup func(*schemas.BifrostContext) (chan *schemas.BifrostStreamChunk, *schemas.BifrostError),
	onCancel func(),
) earlyCommitResponse {
	t.Helper()
	handler := func(reqCtx *fasthttp.RequestCtx) {
		h := &CompletionHandler{config: &lib.Config{}}
		base, cancelBase := context.WithCancel(context.Background())
		bifrostCtx := schemas.NewBifrostContext(base, schemas.NoDeadline)
		cancel := func() {
			if onCancel != nil {
				onCancel()
			}
			cancelBase()
		}
		getStream := func() (chan *schemas.BifrostStreamChunk, *schemas.BifrostError) {
			return setup(bifrostCtx)
		}
		h.handleStreamingResponse(reqCtx, bifrostCtx, requestType, false, getStream, cancel)
	}
	serverConn, clientConn := net.Pipe()
	defer clientConn.Close()
	go func() { _ = fasthttp.ServeConn(serverConn, handler) }()
	if err := clientConn.SetDeadline(time.Now().Add(earlyCommitTestTimeout)); err != nil {
		t.Fatalf("set deadline: %v", err)
	}
	if _, err := clientConn.Write([]byte("POST /v1/chat/completions HTTP/1.1\r\nHost: test\r\n\r\n")); err != nil {
		t.Fatalf("write request: %v", err)
	}
	return readEarlyCommitResponse(t, bufio.NewReader(clientConn))
}

func readEarlyCommitResponse(t *testing.T, br *bufio.Reader) earlyCommitResponse {
	t.Helper()
	var response earlyCommitResponse
	statusLine, err := br.ReadString('\n')
	if err != nil {
		t.Fatalf("read status: %v", err)
	}
	if _, err := fmt.Sscanf(statusLine, "HTTP/1.1 %d", &response.status); err != nil {
		t.Fatalf("parse status %q: %v", statusLine, err)
	}
	var headers strings.Builder
	chunked := false
	contentLength := -1
	for {
		line, err := br.ReadString('\n')
		if err != nil {
			t.Fatalf("read header: %v", err)
		}
		if strings.TrimSpace(line) == "" {
			break
		}
		headers.WriteString(line)
		lower := strings.ToLower(line)
		if strings.HasPrefix(lower, "transfer-encoding:") && strings.Contains(lower, "chunked") {
			chunked = true
		}
		if strings.HasPrefix(lower, "content-length:") {
			_, _ = fmt.Sscanf(strings.TrimSpace(line[len("content-length:"):]), "%d", &contentLength)
		}
	}
	response.headers = headers.String()
	var body strings.Builder
	switch {
	case chunked:
		for {
			sizeLine, err := br.ReadString('\n')
			if err != nil {
				t.Fatalf("read chunk size: %v", err)
			}
			var size int
			if _, err := fmt.Sscanf(strings.TrimSpace(sizeLine), "%x", &size); err != nil {
				t.Fatalf("parse chunk size %q: %v", sizeLine, err)
			}
			if size == 0 {
				break
			}
			data := make([]byte, size+2)
			for n := 0; n < len(data); {
				nn, err := br.Read(data[n:])
				if err != nil {
					t.Fatalf("read chunk: %v", err)
				}
				n += nn
			}
			body.Write(data[:size])
		}
	case contentLength >= 0:
		data := make([]byte, contentLength)
		for n := 0; n < len(data); {
			nn, err := br.Read(data[n:])
			if err != nil {
				t.Fatalf("read body: %v", err)
			}
			n += nn
		}
		body.Write(data)
	}
	response.body = body.String()
	return response
}

func waitOptions() schemas.StreamAdmissionWaitOptions {
	return schemas.StreamAdmissionWaitOptions{CommitAfter: 50 * time.Millisecond, CommentInterval: 20 * time.Millisecond}
}

func readyStream() chan *schemas.BifrostStreamChunk {
	stream := make(chan *schemas.BifrostStreamChunk, 1)
	terminal := contentChunk("hello")
	terminal.BifrostChatResponse.Choices[0].FinishReason = schemas.Ptr("stop")
	stream <- terminal
	close(stream)
	return stream
}

func unavailableError() *schemas.BifrostError {
	status := fasthttp.StatusServiceUnavailable
	return &schemas.BifrostError{
		IsBifrostError: false,
		StatusCode:     &status,
		Error:          &schemas.ErrorField{Message: "session authority unavailable", Code: schemas.Ptr("SESSION_AUTHORITY_UNAVAILABLE")},
	}
}

func TestEarlyCommitReadyBeforeCommitDelayKeepsOrdinaryResponse(t *testing.T) {
	response := serveEarlyCommit(t, schemas.ChatCompletionStreamRequest, func(ctx *schemas.BifrostContext) (chan *schemas.BifrostStreamChunk, *schemas.BifrostError) {
		schemas.StreamAdmissionWaitFrom(ctx).Begin(schemas.StreamAdmissionWaitOptions{CommitAfter: time.Second, CommentInterval: 20 * time.Millisecond})
		time.Sleep(30 * time.Millisecond)
		return readyStream(), nil
	}, nil)
	if response.status != fasthttp.StatusOK || strings.Contains(response.body, ": waiting") {
		t.Fatalf("status=%d body=%q, want 200 without waiting comments", response.status, response.body)
	}
	if !strings.Contains(response.body, `"hello"`) || !strings.HasSuffix(response.body, "data: [DONE]\n\n") {
		t.Fatalf("stream content missing:\n%s", response.body)
	}
}

func TestEarlyCommitErrorBeforeCommitIsHTTPStatus(t *testing.T) {
	response := serveEarlyCommit(t, schemas.ChatCompletionStreamRequest, func(ctx *schemas.BifrostContext) (chan *schemas.BifrostStreamChunk, *schemas.BifrostError) {
		schemas.StreamAdmissionWaitFrom(ctx).Begin(schemas.StreamAdmissionWaitOptions{CommitAfter: time.Second, CommentInterval: 20 * time.Millisecond})
		return nil, unavailableError()
	}, nil)
	if response.status != fasthttp.StatusServiceUnavailable {
		t.Fatalf("status=%d, want 503 before commitment", response.status)
	}
	if strings.Contains(strings.ToLower(response.headers), "text/event-stream") || !strings.Contains(response.body, "SESSION_AUTHORITY_UNAVAILABLE") {
		t.Fatalf("want a JSON error body, got headers=%q body=%q", response.headers, response.body)
	}
}

func TestEarlyCommitSlowSetupWithoutSignalNeverCommits(t *testing.T) {
	response := serveEarlyCommit(t, schemas.ChatCompletionStreamRequest, func(*schemas.BifrostContext) (chan *schemas.BifrostStreamChunk, *schemas.BifrostError) {
		time.Sleep(150 * time.Millisecond)
		return nil, unavailableError()
	}, nil)
	if response.status != fasthttp.StatusServiceUnavailable {
		t.Fatalf("status=%d, want the setup error's own status", response.status)
	}
}

func TestEarlyCommitSendsWaitingCommentsThenStream(t *testing.T) {
	response := serveEarlyCommit(t, schemas.ChatCompletionStreamRequest, func(ctx *schemas.BifrostContext) (chan *schemas.BifrostStreamChunk, *schemas.BifrostError) {
		schemas.StreamAdmissionWaitFrom(ctx).Begin(waitOptions())
		time.Sleep(200 * time.Millisecond)
		return readyStream(), nil
	}, nil)
	if response.status != fasthttp.StatusOK || !strings.Contains(strings.ToLower(response.headers), "text/event-stream") {
		t.Fatalf("status=%d headers=%q, want a committed event stream", response.status, response.headers)
	}
	waiting := strings.Index(response.body, ": waiting\n")
	content := strings.Index(response.body, `"hello"`)
	if waiting < 0 || content < 0 || waiting > content {
		t.Fatalf("want waiting comments before the content:\n%s", response.body)
	}
	if !strings.HasSuffix(response.body, "data: [DONE]\n\n") {
		t.Fatalf("a clean stream ends with [DONE]:\n%s", response.body)
	}
}

func TestEarlyCommitErrorAfterCommitIsInStreamErrorWithoutDone(t *testing.T) {
	response := serveEarlyCommit(t, schemas.ChatCompletionStreamRequest, func(ctx *schemas.BifrostContext) (chan *schemas.BifrostStreamChunk, *schemas.BifrostError) {
		schemas.StreamAdmissionWaitFrom(ctx).Begin(waitOptions())
		time.Sleep(150 * time.Millisecond)
		return nil, unavailableError()
	}, nil)
	if response.status != fasthttp.StatusOK {
		t.Fatalf("status=%d, a committed stream keeps 200", response.status)
	}
	if !strings.Contains(response.body, ": waiting\n") || !strings.Contains(response.body, "SESSION_AUTHORITY_UNAVAILABLE") {
		t.Fatalf("want waiting comments then the error event:\n%s", response.body)
	}
	if strings.Contains(response.body, "[DONE]") {
		t.Fatalf("an error-terminated stream must not end with [DONE]:\n%s", response.body)
	}
}

func TestEarlyCommitResponsesErrorUsesErrorEvent(t *testing.T) {
	response := serveEarlyCommit(t, schemas.ResponsesStreamRequest, func(ctx *schemas.BifrostContext) (chan *schemas.BifrostStreamChunk, *schemas.BifrostError) {
		schemas.StreamAdmissionWaitFrom(ctx).Begin(waitOptions())
		time.Sleep(150 * time.Millisecond)
		return nil, unavailableError()
	}, nil)
	if !strings.Contains(response.body, "event: error\n") {
		t.Fatalf("Responses streams carry errors as an error event:\n%s", response.body)
	}
}

func TestEarlyCommitSanitizesInternalErrors(t *testing.T) {
	response := serveEarlyCommit(t, schemas.ChatCompletionStreamRequest, func(ctx *schemas.BifrostContext) (chan *schemas.BifrostStreamChunk, *schemas.BifrostError) {
		schemas.StreamAdmissionWaitFrom(ctx).Begin(waitOptions())
		time.Sleep(150 * time.Millisecond)
		return nil, &schemas.BifrostError{IsBifrostError: true, Error: &schemas.ErrorField{Message: "panic: secret detail at handler.go:42"}}
	}, nil)
	if strings.Contains(response.body, "secret detail") || !strings.Contains(response.body, lib.ClientSafeInternalErrorMessage) {
		t.Fatalf("internal detail must be sanitized as on the pre-commit path:\n%s", response.body)
	}
}

func TestEarlyCommitClientDisconnectCancelsTheWait(t *testing.T) {
	cancelled := make(chan struct{})
	setupDone := make(chan struct{})
	handler := func(reqCtx *fasthttp.RequestCtx) {
		h := &CompletionHandler{config: &lib.Config{}}
		base, cancelBase := context.WithCancel(context.Background())
		bifrostCtx := schemas.NewBifrostContext(base, schemas.NoDeadline)
		var once bool
		cancel := func() {
			if !once {
				once = true
				close(cancelled)
			}
			cancelBase()
		}
		getStream := func() (chan *schemas.BifrostStreamChunk, *schemas.BifrostError) {
			defer close(setupDone)
			schemas.StreamAdmissionWaitFrom(bifrostCtx).Begin(waitOptions())
			<-bifrostCtx.Done()
			return nil, &schemas.BifrostError{IsBifrostError: true, Error: &schemas.ErrorField{Message: "request cancelled"}}
		}
		h.handleStreamingResponse(reqCtx, bifrostCtx, schemas.ChatCompletionStreamRequest, false, getStream, cancel)
	}
	serverConn, clientConn := net.Pipe()
	go func() { _ = fasthttp.ServeConn(serverConn, handler) }()
	_ = clientConn.SetDeadline(time.Now().Add(earlyCommitTestTimeout))
	if _, err := clientConn.Write([]byte("POST /v1/chat/completions HTTP/1.1\r\nHost: test\r\n\r\n")); err != nil {
		t.Fatalf("write request: %v", err)
	}
	br := bufio.NewReader(clientConn)
	for {
		line, err := br.ReadString('\n')
		if err != nil {
			t.Fatalf("read: %v", err)
		}
		if strings.Contains(line, ": waiting") {
			break
		}
	}
	_ = clientConn.Close()
	select {
	case <-cancelled:
	case <-time.After(earlyCommitTestTimeout):
		t.Fatal("a client that leaves during the wait must cancel the request")
	}
	select {
	case <-setupDone:
	case <-time.After(earlyCommitTestTimeout):
		t.Fatal("the plugin's wait must end once the request is cancelled")
	}
}

func TestEarlyCommitPanicBeforeCommitReachesHandlerGoroutine(t *testing.T) {
	setup := lib.StartStreamSetup(schemas.NewBifrostContext(context.Background(), schemas.NoDeadline), true, func() (chan *schemas.BifrostStreamChunk, *schemas.BifrostError) {
		panic("setup failed")
	})
	defer func() {
		if recovered := recover(); recovered != "setup failed" {
			t.Fatalf("recovered %v, want the setup panic re-raised for the recovery middleware", recovered)
		}
	}()
	setup.Await()
	t.Fatal("Await must re-raise a setup panic")
}

func TestEarlyCommitIsNotOfferedWhenTheRouteCannotCarryComments(t *testing.T) {
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	offered := make(chan bool, 1)
	setup := lib.StartStreamSetup(ctx, false, func() (chan *schemas.BifrostStreamChunk, *schemas.BifrostError) {
		offered <- schemas.StreamAdmissionWaitFrom(ctx) != nil
		return readyStream(), nil
	})
	if _, _, committed := setup.Await(); committed {
		t.Fatal("a route that cannot carry comments must never commit early")
	}
	if <-offered {
		t.Fatal("no wait signal may be offered on such a route; the wait stays silent")
	}
}
