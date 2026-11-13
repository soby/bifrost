package handlers

import (
	"bytes"
	"context"
	"errors"
	"testing"

	"github.com/maximhq/bifrost/core/schemas"
	"github.com/maximhq/bifrost/transports/v2/bifrost-http/lib"
	"github.com/stretchr/testify/require"
	"github.com/valyala/fasthttp"
)

// TransportInterceptorMiddleware resolves the transport post-hook slot on every path out of
// it, so a stream's producer, which waits for the slot without a timer, is always released.

func postHookSlotPlugin(postHook func()) *fakeResponseHeadersPlugin {
	return &fakeResponseHeadersPlugin{
		name: "post-hook-slot",
		hook: func(*schemas.BifrostContext, *schemas.HTTPRequest, *schemas.HTTPResponseMetadata) error { return nil },
		postHook: func(*schemas.BifrostContext, *schemas.HTTPRequest, *schemas.HTTPResponse) error {
			postHook()
			return nil
		},
	}
}

// A streamed response: the completer is published and runs the post-hooks when the producer
// asks, not before.
func TestTransportInterceptorMiddleware_StreamResolvesSlotWithCompleter(t *testing.T) {
	SetLogger(&mockLogger{})
	calls := 0
	var slot *lib.TransportPostHookSlot
	handler := TransportInterceptorMiddleware(responseHeadersTestConfig(t, postHookSlotPlugin(func() { calls++ })))(func(ctx *fasthttp.RequestCtx) {
		ctx.SetUserValue(schemas.BifrostContextKeyDeferTraceCompletion, true)
		slot = lib.NewTransportPostHookCompleterSlot(ctx)
		ctx.Response.SetBodyStream(bytes.NewReader([]byte("data: ok\n\n")), -1)
	})
	ctx := preAuthTestCtx()
	handler(ctx)
	defer ctx.Response.CloseBodyStream() //nolint:errcheck

	require.NotNil(t, slot, "the middleware marks the request, so the handler gets a slot")
	require.Zero(t, calls, "a stream's post-hooks do not run before the stream ends")
	lib.RunTransportPostHookCompleter(slot, nil)
	require.Equal(t, 1, calls, "the published completer runs the post-hooks once")
}

// A handler that allocated a slot but answered without streaming: the post-hooks ran
// synchronously, and the slot is resolved with none.
func TestTransportInterceptorMiddleware_UnaryResolvesSlotWithNone(t *testing.T) {
	SetLogger(&mockLogger{})
	calls := 0
	var slot *lib.TransportPostHookSlot
	handler := TransportInterceptorMiddleware(responseHeadersTestConfig(t, postHookSlotPlugin(func() { calls++ })))(func(ctx *fasthttp.RequestCtx) {
		slot = lib.NewTransportPostHookCompleterSlot(ctx)
		ctx.SetBodyString("ok")
	})
	handler(preAuthTestCtx())

	require.NotNil(t, slot)
	require.Nil(t, slot.Wait(), "no deferred completer for a response that was not streamed")
	require.Equal(t, 1, calls, "the post-hooks ran synchronously, once")
}

// A handler that panics after allocating a slot: the panic passes through, and the slot is
// still resolved, so a producer already started is not left waiting.
func TestTransportInterceptorMiddleware_PanicResolvesSlotWithNone(t *testing.T) {
	SetLogger(&mockLogger{})
	var slot *lib.TransportPostHookSlot
	handler := TransportInterceptorMiddleware(responseHeadersTestConfig(t, postHookSlotPlugin(func() {})))(func(ctx *fasthttp.RequestCtx) {
		ctx.SetUserValue(schemas.BifrostContextKeyDeferTraceCompletion, true)
		slot = lib.NewTransportPostHookCompleterSlot(ctx)
		panic("handler failed")
	})
	require.PanicsWithValue(t, "handler failed", func() { handler(preAuthTestCtx()) })
	require.NotNil(t, slot)
	require.Nil(t, slot.Wait(), "a panicking handler leaves the slot resolved with none")
}

// A pre-hook that answers the request itself, or fails: the handler never runs, so no slot
// exists and nothing is left for a producer to wait on.
func TestTransportInterceptorMiddleware_PreHookShortCircuitLeavesNoPendingSlot(t *testing.T) {
	SetLogger(&mockLogger{})
	for name, preHook := range map[string]func(*schemas.BifrostContext, *schemas.HTTPRequest) (*schemas.HTTPResponse, error){
		"response": func(*schemas.BifrostContext, *schemas.HTTPRequest) (*schemas.HTTPResponse, error) {
			return &schemas.HTTPResponse{StatusCode: fasthttp.StatusTeapot}, nil
		},
		"error": func(*schemas.BifrostContext, *schemas.HTTPRequest) (*schemas.HTTPResponse, error) {
			return nil, errors.New("pre-hook failed")
		},
	} {
		t.Run(name, func(t *testing.T) {
			plugin := postHookSlotPlugin(func() { t.Error("no post-hook pairs with a short-circuited pre-hook") })
			plugin.preHook = preHook
			handlerRan := false
			handler := TransportInterceptorMiddleware(responseHeadersTestConfig(t, plugin))(func(*fasthttp.RequestCtx) { handlerRan = true })
			ctx := preAuthTestCtx()
			handler(ctx)

			require.False(t, handlerRan)
			_, pending := ctx.UserValue(schemas.BifrostContextKeyTransportPostHookCompleter).(*lib.TransportPostHookSlot)
			require.False(t, pending, "a short-circuited request has no slot")
		})
	}
}

// With no transport plugin loaded the middleware marks nothing: a streaming handler gets no
// slot and its producer waits for nothing.
func TestTransportInterceptorMiddleware_NoPluginsNoSlot(t *testing.T) {
	var slot *lib.TransportPostHookSlot
	handler := TransportInterceptorMiddleware(&lib.Config{})(func(ctx *fasthttp.RequestCtx) {
		slot = lib.NewTransportPostHookCompleterSlot(ctx)
	})
	handler(preAuthTestCtx())
	require.Nil(t, slot)
	require.Nil(t, lib.RunTransportPostHookCompleter(slot, nil))
}

// What the stream's producer recorded decides what the post-hooks are told: a request that
// ended on its own, or a stream the transport could not deliver, reaches them as a cancelled
// context; a stream the producer ended itself after writing to the client does not, although
// the producer cancelled the handler's context to stop the provider.
func TestTransportInterceptorMiddleware_StreamPostHookContextFollowsTheProducersRecord(t *testing.T) {
	SetLogger(&mockLogger{})
	for name, tc := range map[string]struct {
		record    func(slot *lib.TransportPostHookSlot, requestCtx *schemas.BifrostContext)
		cancelled bool
	}{
		"delivered, then the producer's own cancel": {
			record: func(slot *lib.TransportPostHookSlot, requestCtx *schemas.BifrostContext) {
				slot.CaptureRequestState(requestCtx, nil)
			},
		},
		"the transport could not deliver a released chunk": {
			record:    func(slot *lib.TransportPostHookSlot, _ *schemas.BifrostContext) { slot.MarkUndelivered() },
			cancelled: true,
		},
	} {
		t.Run(name, func(t *testing.T) {
			var postHookErr error
			plugin := &fakeResponseHeadersPlugin{
				name: "post-hook-context",
				hook: func(*schemas.BifrostContext, *schemas.HTTPRequest, *schemas.HTTPResponseMetadata) error { return nil },
				postHook: func(ctx *schemas.BifrostContext, _ *schemas.HTTPRequest, _ *schemas.HTTPResponse) error {
					postHookErr = ctx.Err()
					return nil
				},
			}
			var slot *lib.TransportPostHookSlot
			var requestCtx *schemas.BifrostContext
			var cancel func()
			handler := TransportInterceptorMiddleware(responseHeadersTestConfig(t, plugin))(func(ctx *fasthttp.RequestCtx) {
				requestCtx, cancel = lib.ConvertToBifrostContext(ctx, nil)
				ctx.SetUserValue(schemas.BifrostContextKeyDeferTraceCompletion, true)
				slot = lib.NewTransportPostHookCompleterSlot(ctx)
				ctx.Response.SetBodyStream(bytes.NewReader([]byte("data: ok\n\n")), -1)
			})
			ctx := preAuthTestCtx()
			handler(ctx)
			defer ctx.Response.CloseBodyStream() //nolint:errcheck

			// The producer's end of stream: record, cancel to stop the provider, run the post-hooks.
			tc.record(slot, requestCtx)
			cancel()
			lib.RunTransportPostHookCompleter(slot, nil)
			if tc.cancelled {
				require.ErrorIs(t, postHookErr, context.Canceled)
			} else {
				require.NoError(t, postHookErr)
			}
		})
	}
}
