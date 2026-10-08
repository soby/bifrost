package utils

import (
	"context"
	"errors"
	"net"
	"testing"
	"time"

	"github.com/maximhq/bifrost/core/schemas"
	"github.com/valyala/fasthttp"
	"github.com/valyala/fasthttp/fasthttputil"
)

// The client ReadTimeout in these tests stands in for default_request_timeout_in_seconds
// (300 s): an upstream slower than it is the long unary call it must not cut on a
// request-scoped client.
const (
	scaledReadTimeout = 150 * time.Millisecond
	slowUpstream      = 600 * time.Millisecond
)

// newReadTimeoutClient serves handler in memory behind a client built the way provider
// constructors build theirs: ReadTimeout from the network config, then ConfigureDialerFor.
func newReadTimeoutClient(t *testing.T, handler fasthttp.RequestHandler, networkConfig schemas.NetworkConfig, stream bool) *fasthttp.Client {
	t.Helper()
	ln := fasthttputil.NewInmemoryListener()
	server := &fasthttp.Server{Handler: handler}
	go server.Serve(ln) //nolint:errcheck
	t.Cleanup(func() { _ = ln.Close() })
	client := &fasthttp.Client{
		Dial:               func(string) (net.Conn, error) { return ln.Dial() },
		ReadTimeout:        scaledReadTimeout,
		WriteTimeout:       scaledReadTimeout,
		StreamResponseBody: stream,
	}
	return ConfigureDialerFor(client, networkConfig)
}

// slowHandler answers after delay, or when the test ends.
func slowHandler(t *testing.T, delay time.Duration) fasthttp.RequestHandler {
	release := make(chan struct{})
	t.Cleanup(func() { close(release) })
	return func(ctx *fasthttp.RequestCtx) {
		select {
		case <-release:
		case <-time.After(delay):
		}
		ctx.SetStatusCode(fasthttp.StatusOK)
		ctx.SetBodyString(`{"ok":true}`)
	}
}

func requestScopedNetworkConfig() schemas.NetworkConfig {
	return schemas.NetworkConfig{LoopbackIsPrivate: true, ContextBoundReads: true}
}

func makeSlowUnaryRequest(ctx context.Context, client *fasthttp.Client) (*schemas.BifrostError, string, time.Duration) {
	req := fasthttp.AcquireRequest()
	resp := fasthttp.AcquireResponse()
	defer fasthttp.ReleaseRequest(req)
	defer fasthttp.ReleaseResponse(resp)
	req.SetRequestURI("http://upstream/v1/chat/completions")
	req.Header.SetMethod(fasthttp.MethodPost)
	started := time.Now()
	_, bifrostErr, wait := MakeRequestWithContext(ctx, client, req, resp)
	wait()
	return bifrostErr, string(resp.Body()), time.Since(started)
}

func TestContextBoundReads_UnaryOutlivesReadTimeout(t *testing.T) {
	client := newReadTimeoutClient(t, slowHandler(t, slowUpstream), requestScopedNetworkConfig(), false)
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	defer ctx.Cancel()

	bifrostErr, body, elapsed := makeSlowUnaryRequest(ctx, client)
	if bifrostErr != nil {
		t.Fatalf("a request-scoped unary call slower than ReadTimeout failed after %s: %+v", elapsed, bifrostErr.Error)
	}
	if body != `{"ok":true}` {
		t.Errorf("body = %q", body)
	}
	if elapsed < slowUpstream {
		t.Errorf("call returned after %s, before the upstream answered", elapsed)
	}
}

func TestContextBoundReads_ConfiguredProviderKeepsReadTimeout(t *testing.T) {
	client := newReadTimeoutClient(t, slowHandler(t, slowUpstream), schemas.NetworkConfig{}, false)
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	defer ctx.Cancel()

	bifrostErr, _, elapsed := makeSlowUnaryRequest(ctx, client)
	if bifrostErr == nil {
		t.Fatal("a configured provider's ReadTimeout no longer bounds a unary call")
	}
	if elapsed >= slowUpstream {
		t.Errorf("call returned after %s, want about the %s ReadTimeout", elapsed, scaledReadTimeout)
	}
}

func TestContextBoundReads_AttemptRequestTimeoutCutsUnary(t *testing.T) {
	client := newReadTimeoutClient(t, slowHandler(t, 5*time.Second), requestScopedNetworkConfig(), false)
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	defer ctx.Cancel()
	ctx.SetValue(schemas.BifrostContextKeyAttemptRequestTimeout, 2*scaledReadTimeout)

	bifrostErr, _, elapsed := makeSlowUnaryRequest(ctx, client)
	if bifrostErr == nil || bifrostErr.Error == nil || bifrostErr.Error.Type == nil || *bifrostErr.Error.Type != schemas.RequestTimedOut {
		t.Fatalf("error = %+v, want %q from the attempt timeout", bifrostErr, schemas.RequestTimedOut)
	}
	if elapsed < 2*scaledReadTimeout || elapsed > 2*time.Second {
		t.Errorf("call returned after %s, want about the %s attempt timeout", elapsed, 2*scaledReadTimeout)
	}
	if ctx.Err() != nil {
		t.Errorf("the attempt timeout cancelled the request context: %v", ctx.Err())
	}
}

func TestContextBoundReads_ContextDeadlineCutsUnary(t *testing.T) {
	client := newReadTimeoutClient(t, slowHandler(t, 5*time.Second), requestScopedNetworkConfig(), false)
	ctx := schemas.NewBifrostContext(context.Background(), time.Now().Add(2*scaledReadTimeout))
	defer ctx.Cancel()

	bifrostErr, _, elapsed := makeSlowUnaryRequest(ctx, client)
	if bifrostErr == nil {
		t.Fatal("a request-scoped unary call outlived its context deadline")
	}
	if elapsed < 2*scaledReadTimeout || elapsed > 2*time.Second {
		t.Errorf("call returned after %s, want about the %s context deadline", elapsed, 2*scaledReadTimeout)
	}
}

func TestContextBoundReads_CancellationCutsUnary(t *testing.T) {
	client := newReadTimeoutClient(t, slowHandler(t, 5*time.Second), requestScopedNetworkConfig(), false)
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	time.AfterFunc(2*scaledReadTimeout, ctx.Cancel)

	bifrostErr, _, elapsed := makeSlowUnaryRequest(ctx, client)
	if bifrostErr == nil || bifrostErr.Error == nil || bifrostErr.Error.Type == nil || *bifrostErr.Error.Type != schemas.RequestCancelled {
		t.Fatalf("error = %+v, want %q", bifrostErr, schemas.RequestCancelled)
	}
	if elapsed > 2*time.Second {
		t.Errorf("call returned after %s, want about the %s cancellation", elapsed, 2*scaledReadTimeout)
	}
}

// A request that reaches the transport with no context (a bare client.Do) can never be
// cancelled, so ReadTimeout stays its bound.
func TestContextBoundReads_UnboundRequestKeepsReadTimeout(t *testing.T) {
	client := newReadTimeoutClient(t, slowHandler(t, 5*time.Second), requestScopedNetworkConfig(), false)
	req := fasthttp.AcquireRequest()
	resp := fasthttp.AcquireResponse()
	defer fasthttp.ReleaseRequest(req)
	defer fasthttp.ReleaseResponse(resp)
	req.SetRequestURI("http://upstream/v1/chat/completions")

	started := time.Now()
	err := client.Do(req, resp)
	if !errors.Is(err, fasthttp.ErrTimeout) {
		t.Fatalf("error = %v, want fasthttp.ErrTimeout from ReadTimeout", err)
	}
	if elapsed := time.Since(started); elapsed > 2*time.Second {
		t.Errorf("call returned after %s, want about the %s ReadTimeout", elapsed, scaledReadTimeout)
	}
}

func TestContextBoundReads_StreamHeaderWaitOutlivesReadTimeout(t *testing.T) {
	client := newReadTimeoutClient(t, slowHandler(t, slowUpstream), requestScopedNetworkConfig(), true)
	client = BuildStreamingClient(client)
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	defer ctx.Cancel()

	req := fasthttp.AcquireRequest()
	resp := fasthttp.AcquireResponse()
	defer fasthttp.ReleaseRequest(req)
	defer ReleaseStreamingResponse(ctx, resp)
	req.SetRequestURI("http://upstream/v1/chat/completions")

	if err := DoStreamingRequest(ctx, client, req, resp); err != nil {
		t.Fatalf("a request-scoped stream whose headers come after ReadTimeout failed: %v", err)
	}
	if resp.StatusCode() != fasthttp.StatusOK {
		t.Errorf("status = %d", resp.StatusCode())
	}
}

func TestContextBoundReads_AttemptRequestTimeoutCutsStreamHeaderWait(t *testing.T) {
	client := BuildStreamingClient(newReadTimeoutClient(t, slowHandler(t, 5*time.Second), requestScopedNetworkConfig(), true))
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	defer ctx.Cancel()
	ctx.SetValue(schemas.BifrostContextKeyAttemptRequestTimeout, 2*scaledReadTimeout)

	req := fasthttp.AcquireRequest()
	resp := fasthttp.AcquireResponse()
	defer fasthttp.ReleaseRequest(req)
	defer ReleaseStreamingResponse(ctx, resp)
	req.SetRequestURI("http://upstream/v1/chat/completions")

	started := time.Now()
	err := DoStreamingRequest(ctx, client, req, resp)
	if !errors.Is(err, fasthttp.ErrTimeout) && !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("error = %v, want a timeout from the attempt timeout", err)
	}
	if elapsed := time.Since(started); elapsed > 2*time.Second {
		t.Errorf("header wait lasted %s, want about the %s attempt timeout", elapsed, 2*scaledReadTimeout)
	}
}
