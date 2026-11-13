package lib

import (
	"testing"

	"github.com/maximhq/bifrost/core/schemas"
	"github.com/valyala/fasthttp"
)

func markedForTransportPostHooks() *fasthttp.RequestCtx {
	ctx := &fasthttp.RequestCtx{}
	ctx.SetUserValue(schemas.BifrostContextKeyTransportPostHookCompleter, true)
	return ctx
}

// A producer that reaches the end of its stream before the middleware has published the
// completer waits for it: the post-hooks still run, however late they arrive.
func TestTransportPostHookSlotLateCompleterStillRuns(t *testing.T) {
	ctx := markedForTransportPostHooks()
	slot := NewTransportPostHookCompleterSlot(ctx)
	if slot == nil {
		t.Fatal("a marked request gets a slot")
	}
	if stored, _ := ctx.UserValue(schemas.BifrostContextKeyTransportPostHookCompleter).(*TransportPostHookSlot); stored != slot {
		t.Fatal("the slot replaces the marker on the request, for the middleware to resolve")
	}

	waiting := make(chan struct{})
	got := make(chan []schemas.PluginLogEntry, 1)
	go func() {
		close(waiting)
		got <- RunTransportPostHookCompleter(slot, nil)
	}()
	<-waiting
	select {
	case <-got:
		t.Fatal("the producer must wait for the middleware to resolve the slot")
	default:
	}

	calls := 0
	want := []schemas.PluginLogEntry{{PluginName: "transport", Message: "ran"}}
	slot.Resolve(func() ([]schemas.PluginLogEntry, error) {
		calls++
		return want, nil
	})
	// A second resolution, like the middleware's deferred "none", changes nothing.
	slot.Resolve(nil)
	logs := <-got
	if calls != 1 || len(logs) != 1 || logs[0].Message != "ran" {
		t.Fatalf("calls=%d logs=%+v, want the late completer run exactly once", calls, logs)
	}
}

// With no transport plugin loaded the request is not marked: there is no slot, and a stream
// has nothing to wait for.
func TestTransportPostHookSlotAbsentWithoutTransportPlugins(t *testing.T) {
	slot := NewTransportPostHookCompleterSlot(&fasthttp.RequestCtx{})
	if slot != nil {
		t.Fatal("an unmarked request must get no slot")
	}
	if logs := RunTransportPostHookCompleter(slot, nil); logs != nil {
		t.Fatalf("no slot runs nothing, got %+v", logs)
	}
	slot.Resolve(nil) // resolving no slot is harmless
}

// A slot resolved with nil means there are no post-hooks: the producer is released and runs none.
func TestTransportPostHookSlotResolvedWithNone(t *testing.T) {
	slot := NewTransportPostHookCompleterSlot(markedForTransportPostHooks())
	slot.Resolve(nil)
	slot.Resolve(func() ([]schemas.PluginLogEntry, error) {
		t.Error("a completer published after the slot was resolved must not run")
		return nil, nil
	})
	if logs := RunTransportPostHookCompleter(slot, nil); logs != nil {
		t.Fatalf("got %+v, want no logs", logs)
	}
}

// The slot remembers how the request stood the first time the producer says so: a cancel the
// producer issues afterwards, to stop the provider, does not change the answer.
func TestTransportPostHookSlotCapturesRequestStateOnce(t *testing.T) {
	live := NewTransportPostHookCompleterSlot(markedForTransportPostHooks())
	if _, captured := live.RequestEnded(); captured {
		t.Fatal("nothing is recorded until the producer captures the state")
	}
	ctx, cancel := schemas.NewBifrostContextWithCancel(t.Context())
	reader := NewSSEStreamReader()
	live.CaptureRequestState(ctx, reader)
	cancel()
	_ = reader.Close()
	live.CaptureRequestState(ctx, reader)
	if ended, captured := live.RequestEnded(); !captured || ended {
		t.Fatalf("ended=%v captured=%v, want the live state recorded before the producer's own cancel", ended, captured)
	}

	cancelled := NewTransportPostHookCompleterSlot(markedForTransportPostHooks())
	cancelled.CaptureRequestState(ctx, NewSSEStreamReader())
	if ended, _ := cancelled.RequestEnded(); !ended {
		t.Fatal("a request whose context was already done is recorded as ended")
	}

	writeFailed := NewTransportPostHookCompleterSlot(markedForTransportPostHooks())
	writeFailed.CaptureRequestState(schemas.NewBifrostContext(t.Context(), schemas.NoDeadline), reader)
	if ended, _ := writeFailed.RequestEnded(); !ended {
		t.Fatal("a client stream closed under the producer is recorded as ended")
	}

	var none *TransportPostHookSlot
	none.CaptureRequestState(ctx, reader)
	if ended, captured := none.RequestEnded(); ended || captured {
		t.Fatal("no slot records nothing")
	}
}
