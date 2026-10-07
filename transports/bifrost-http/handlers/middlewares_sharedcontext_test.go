package handlers

import (
	"testing"

	"github.com/maximhq/bifrost/core/schemas"
	"github.com/maximhq/bifrost/transports/bifrost-http/lib"
	"github.com/valyala/fasthttp"
)

// Fork-only: the transport middleware, the request handler, the LLM hooks and
// the transport post-hook share one BifrostContext per request.

// traceIDCtxKey is the context key the echo plugin round-trips: a value written
// on the request context during the request and read back in the post-hook.
const traceIDCtxKey = "request_trace_id"

// traceIDHeader is the response header the echo plugin sets from that value.
const traceIDHeader = "X-Trace-Id"

// traceEchoPlugin copies a value read from the request context in its post-hook
// into a response header, the way a plugin surfaces something computed during
// the request.
type traceEchoPlugin struct {
	postHookErr error
}

func (p *traceEchoPlugin) GetName() string { return "trace-echo" }
func (p *traceEchoPlugin) Cleanup() error  { return nil }
func (p *traceEchoPlugin) HTTPTransportPreAuthHook(*schemas.BifrostContext, *schemas.HTTPRequest) (*schemas.HTTPResponse, error) {
	return nil, nil
}
func (p *traceEchoPlugin) HTTPTransportPreHook(*schemas.BifrostContext, *schemas.HTTPRequest) (*schemas.HTTPResponse, error) {
	return nil, nil
}
func (p *traceEchoPlugin) HTTPTransportPostHook(ctx *schemas.BifrostContext, _ *schemas.HTTPRequest, resp *schemas.HTTPResponse) error {
	if v, ok := ctx.Value(traceIDCtxKey).(string); ok && v != "" {
		if resp.Headers == nil {
			resp.Headers = make(map[string]string, 1)
		}
		resp.Headers[traceIDHeader] = v
	}
	p.postHookErr = ctx.Err()
	return nil
}
func (p *traceEchoPlugin) HTTPTransportResponseHeadersHook(*schemas.BifrostContext, *schemas.HTTPRequest, *schemas.HTTPResponseMetadata) error {
	return nil
}
func (p *traceEchoPlugin) HTTPTransportStreamChunkHook(_ *schemas.BifrostContext, _ *schemas.HTTPRequest, chunk *schemas.BifrostStreamChunk) (*schemas.BifrostStreamChunk, error) {
	return chunk, nil
}

// TestEnsureSharedBifrostContext_ReusedByInnerPipeline: the context the
// transport middleware establishes is the exact context the request handler
// (lib.ConvertToBifrostContext) adopts.
func TestEnsureSharedBifrostContext_ReusedByInnerPipeline(t *testing.T) {
	ctx := &fasthttp.RequestCtx{}

	shared, cancel := lib.EnsureSharedBifrostContext(ctx)
	defer cancel()
	again, againCancel := lib.EnsureSharedBifrostContext(ctx)
	defer againCancel()
	if again != shared {
		t.Fatal("a second call created another context")
	}

	inner, innerCancel := lib.ConvertToBifrostContext(ctx, nil)
	defer innerCancel()
	if inner != shared {
		t.Fatal("the request handler built a separate context instead of adopting the shared one")
	}
	inner.SetValue(traceIDCtxKey, "trace-abc123")
	if got, _ := shared.Value(traceIDCtxKey).(string); got != "trace-abc123" {
		t.Fatalf("value written by the handler is not on the shared context: got %q", got)
	}
}

// TestTransportInterceptorMiddleware_PostHookSeesInnerPipelineValue drives the
// middleware: the post-hook reads a value the handler wrote during the request
// and sees the handler's cancellation of the request context.
func TestTransportInterceptorMiddleware_PostHookSeesInnerPipelineValue(t *testing.T) {
	cfg := &lib.Config{}
	plugin := &traceEchoPlugin{}
	plugins := []schemas.HTTPTransportPlugin{plugin}
	cfg.HTTPTransportPlugins.Store(&plugins)

	next := func(ctx *fasthttp.RequestCtx) {
		inner, cancel := lib.ConvertToBifrostContext(ctx, nil)
		defer cancel()
		inner.SetValue(traceIDCtxKey, "trace-abc123")
	}
	handler := TransportInterceptorMiddleware(cfg)(next)

	ctx := &fasthttp.RequestCtx{}
	ctx.Request.Header.SetMethod("POST")
	ctx.Request.SetRequestURI("/v1/chat/completions")
	handler(ctx)

	if got := string(ctx.Response.Header.Peek(traceIDHeader)); got != "trace-abc123" {
		t.Fatalf("post-hook did not observe the handler's value; %s = %q", traceIDHeader, got)
	}
	if plugin.postHookErr == nil {
		t.Fatal("post-hook saw a live context; the handler's cancellation of the shared context did not reach it")
	}
}
