package schemas

import (
	"context"
	"testing"
)

func TestStreamAdmissionWaitBeginSignalsOnce(t *testing.T) {
	wait := NewStreamAdmissionWait()
	select {
	case <-wait.Requested():
		t.Fatal("an unsignalled wait must not report a request")
	default:
	}
	wait.Begin()
	wait.Begin() // a second call must neither panic nor block
	select {
	case <-wait.Requested():
	default:
		t.Fatal("Begin must signal the transport")
	}
}

func TestStreamAdmissionWaitNilIsSilent(t *testing.T) {
	var wait *StreamAdmissionWait
	wait.Begin()
	if wait.Requested() != nil {
		t.Fatal("a nil wait never signals")
	}
}

func TestStreamAdmissionWaitFromReadsTheRequestContext(t *testing.T) {
	if StreamAdmissionWaitFrom(nil) != nil { //nolint:staticcheck // nil context is the case under test
		t.Fatal("a nil context offers no wait")
	}
	ctx := NewBifrostContext(context.Background(), NoDeadline)
	if StreamAdmissionWaitFrom(ctx) != nil {
		t.Fatal("a context the transport did not prepare offers no wait")
	}
	wait := NewStreamAdmissionWait()
	ctx.SetValue(BifrostContextKeyStreamAdmissionWait, wait)
	if StreamAdmissionWaitFrom(ctx) != wait {
		t.Fatal("the wait stored by the transport must be returned")
	}
}
