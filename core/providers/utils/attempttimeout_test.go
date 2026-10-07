package utils

import (
	"bufio"
	"context"
	"errors"
	"io"
	"net"
	"testing"
	"time"

	"github.com/maximhq/bifrost/core/schemas"
	"github.com/valyala/fasthttp"
	"github.com/valyala/fasthttp/fasthttputil"
)

// newAttemptTimeoutServer serves handler in memory and returns a client that dials it through
// Bifrost's context transport, as every provider client does.
func newAttemptTimeoutServer(t *testing.T, handler fasthttp.RequestHandler, stream bool) *fasthttp.Client {
	t.Helper()
	ln := fasthttputil.NewInmemoryListener()
	server := &fasthttp.Server{Handler: handler}
	go server.Serve(ln) //nolint:errcheck
	t.Cleanup(func() { _ = ln.Close() })
	return &fasthttp.Client{
		Dial:               func(string) (net.Conn, error) { return ln.Dial() },
		Transport:          NewContextTransport(),
		StreamResponseBody: stream,
	}
}

func attemptTimeoutContext(timeout time.Duration) *schemas.BifrostContext {
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	if timeout > 0 {
		ctx.SetValue(schemas.BifrostContextKeyAttemptRequestTimeout, timeout)
	}
	return ctx
}

func TestMakeRequestWithContext_AttemptRequestTimeout_TimesOutWithoutCancellingRequest(t *testing.T) {
	release := make(chan struct{})
	t.Cleanup(func() { close(release) })
	client := newAttemptTimeoutServer(t, func(ctx *fasthttp.RequestCtx) {
		select {
		case <-release:
		case <-time.After(5 * time.Second):
		}
		ctx.SetStatusCode(fasthttp.StatusOK)
	}, false)
	ctx := attemptTimeoutContext(150 * time.Millisecond)

	req := fasthttp.AcquireRequest()
	resp := fasthttp.AcquireResponse()
	defer fasthttp.ReleaseRequest(req)
	defer fasthttp.ReleaseResponse(resp)
	req.SetRequestURI("http://upstream/v1/chat/completions")
	req.Header.SetMethod(fasthttp.MethodPost)

	started := time.Now()
	_, bifrostErr, wait := MakeRequestWithContext(ctx, client, req, resp)
	wait()
	elapsed := time.Since(started)

	if bifrostErr == nil {
		t.Fatal("MakeRequestWithContext returned no error for an upstream slower than the attempt timeout")
	}
	if bifrostErr.Error == nil || bifrostErr.Error.Type == nil || *bifrostErr.Error.Type != schemas.RequestTimedOut {
		t.Errorf("error type = %v, want %q (a timeout lets fallbacks run)", bifrostErr.Error, schemas.RequestTimedOut)
	}
	if bifrostErr.StatusCode == nil || *bifrostErr.StatusCode != 504 {
		t.Errorf("status = %v, want 504", bifrostErr.StatusCode)
	}
	if elapsed > 2*time.Second {
		t.Errorf("call returned after %s, want about the 150ms attempt timeout", elapsed)
	}
	if ctx.Err() != nil {
		t.Errorf("the attempt timeout cancelled the request context: %v", ctx.Err())
	}
}

func TestMakeRequestWithContext_NoAttemptRequestTimeout_WaitsForUpstream(t *testing.T) {
	client := newAttemptTimeoutServer(t, func(ctx *fasthttp.RequestCtx) {
		time.Sleep(300 * time.Millisecond)
		ctx.SetStatusCode(fasthttp.StatusOK)
		ctx.SetBodyString(`{"ok":true}`)
	}, false)

	for _, timeout := range []time.Duration{0, -time.Second, 5 * time.Second} {
		req := fasthttp.AcquireRequest()
		resp := fasthttp.AcquireResponse()
		req.SetRequestURI("http://upstream/v1/chat/completions")
		_, bifrostErr, wait := MakeRequestWithContext(attemptTimeoutContext(timeout), client, req, resp)
		wait()
		if bifrostErr != nil {
			t.Errorf("timeout %s: unexpected error %v", timeout, bifrostErr.Error)
		} else if string(resp.Body()) != `{"ok":true}` {
			t.Errorf("timeout %s: body = %q", timeout, resp.Body())
		}
		fasthttp.ReleaseRequest(req)
		fasthttp.ReleaseResponse(resp)
	}
}

func TestMakeRequestWithContext_AttemptRequestTimeout_RequestCancellationStaysCancellation(t *testing.T) {
	release := make(chan struct{})
	t.Cleanup(func() { close(release) })
	client := newAttemptTimeoutServer(t, func(ctx *fasthttp.RequestCtx) {
		select {
		case <-release:
		case <-time.After(5 * time.Second):
		}
	}, false)
	ctx := attemptTimeoutContext(5 * time.Second)
	time.AfterFunc(100*time.Millisecond, ctx.Cancel)

	req := fasthttp.AcquireRequest()
	resp := fasthttp.AcquireResponse()
	defer fasthttp.ReleaseRequest(req)
	defer fasthttp.ReleaseResponse(resp)
	req.SetRequestURI("http://upstream/v1/chat/completions")
	_, bifrostErr, wait := MakeRequestWithContext(ctx, client, req, resp)
	wait()
	if bifrostErr == nil || bifrostErr.Error == nil || bifrostErr.Error.Type == nil || *bifrostErr.Error.Type != schemas.RequestCancelled {
		t.Fatalf("error = %+v, want %q: cancelling the request must not read as an attempt timeout", bifrostErr, schemas.RequestCancelled)
	}
}

func TestDoStreamingRequest_AttemptRequestTimeout_BoundsHeaderWait(t *testing.T) {
	release := make(chan struct{})
	t.Cleanup(func() { close(release) })
	client := newAttemptTimeoutServer(t, func(ctx *fasthttp.RequestCtx) {
		select {
		case <-release:
		case <-time.After(5 * time.Second):
		}
	}, true)
	ctx := attemptTimeoutContext(150 * time.Millisecond)

	req := fasthttp.AcquireRequest()
	resp := fasthttp.AcquireResponse()
	defer fasthttp.ReleaseRequest(req)
	defer ReleaseStreamingResponse(ctx, resp)
	req.SetRequestURI("http://upstream/v1/chat/completions")

	started := time.Now()
	err := DoStreamingRequest(ctx, client, req, resp)
	if err == nil {
		t.Fatal("DoStreamingRequest returned no error for an upstream that never sends headers")
	}
	if !errors.Is(err, fasthttp.ErrTimeout) && !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("error = %v, want a timeout (fasthttp.ErrTimeout or context.DeadlineExceeded)", err)
	}
	if elapsed := time.Since(started); elapsed > 2*time.Second {
		t.Errorf("header wait lasted %s, want about the 150ms attempt timeout", elapsed)
	}
	if ctx.Err() != nil {
		t.Errorf("the attempt timeout cancelled the request context: %v", ctx.Err())
	}
}

func TestDoStreamingRequest_AttemptRequestTimeout_DoesNotBoundBody(t *testing.T) {
	client := newAttemptTimeoutServer(t, func(ctx *fasthttp.RequestCtx) {
		ctx.SetContentType("text/event-stream")
		ctx.SetBodyStreamWriter(func(w *bufio.Writer) {
			_, _ = w.WriteString("data: first\n\n")
			_ = w.Flush()
			time.Sleep(400 * time.Millisecond)
			_, _ = w.WriteString("data: second\n\n")
			_ = w.Flush()
		})
	}, true)
	ctx := attemptTimeoutContext(150 * time.Millisecond)

	req := fasthttp.AcquireRequest()
	resp := fasthttp.AcquireResponse()
	defer fasthttp.ReleaseRequest(req)
	defer ReleaseStreamingResponse(ctx, resp)
	req.SetRequestURI("http://upstream/v1/chat/completions")

	if err := DoStreamingRequest(ctx, client, req, resp); err != nil {
		t.Fatalf("DoStreamingRequest: %v", err)
	}
	body, err := io.ReadAll(resp.BodyStream())
	if err != nil {
		t.Fatalf("reading a body slower than the attempt timeout: %v", err)
	}
	if want := "data: first\n\ndata: second\n\n"; string(body) != want {
		t.Errorf("body = %q, want %q", body, want)
	}
}

func TestWithAttemptRequestTimeout_KeepsValuesAndEarlierDeadline(t *testing.T) {
	parent := schemas.NewBifrostContext(context.Background(), time.Now().Add(50*time.Millisecond))
	parent.SetValue(schemas.BifrostContextKeyAttemptRequestTimeout, time.Hour)
	parent.SetValue(schemas.BifrostContextKeyRequestID, "req-1")
	child, cancel := withAttemptRequestTimeout(parent)
	defer cancel()
	if got, _ := child.Value(schemas.BifrostContextKeyRequestID).(string); got != "req-1" {
		t.Errorf("child lost the request's values: request id %q", got)
	}
	deadline, ok := child.Deadline()
	if !ok || time.Until(deadline) > time.Second {
		t.Errorf("child deadline = %v (set %v), want the request's earlier deadline", deadline, ok)
	}
}
