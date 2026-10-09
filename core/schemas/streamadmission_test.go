package schemas

import (
	"context"
	"testing"
	"time"
)

func TestStreamAdmissionWaitDeliversOnlyTheFirstValidBegin(t *testing.T) {
	wait := NewStreamAdmissionWait()
	wait.Begin(StreamAdmissionWaitOptions{CommitAfter: -time.Second, CommentInterval: time.Second})
	wait.Begin(StreamAdmissionWaitOptions{CommitAfter: time.Second, CommentInterval: 0})
	select {
	case options := <-wait.Requested():
		t.Fatalf("invalid options must keep the wait silent, got %+v", options)
	default:
	}
	first := StreamAdmissionWaitOptions{CommitAfter: 0, CommentInterval: 2 * time.Second}
	wait.Begin(first)
	wait.Begin(StreamAdmissionWaitOptions{CommitAfter: time.Minute, CommentInterval: time.Minute})
	if got := <-wait.Requested(); got != first {
		t.Fatalf("got %+v, want the first valid Begin %+v", got, first)
	}
	select {
	case options := <-wait.Requested():
		t.Fatalf("only one signal may be delivered, got a second %+v", options)
	default:
	}
}

func TestStreamAdmissionWaitIsAbsentUnlessTheTransportOffersIt(t *testing.T) {
	if StreamAdmissionWaitFrom(context.Background()) != nil {
		t.Fatal("no wait without the transport's value")
	}
	var missing *StreamAdmissionWait
	missing.Begin(StreamAdmissionWaitOptions{CommitAfter: time.Second, CommentInterval: time.Second})
	if missing.Requested() != nil {
		t.Fatal("a nil wait has no signal channel")
	}
	ctx := NewBifrostContext(context.Background(), NoDeadline)
	offered := NewStreamAdmissionWait()
	ctx.SetValue(BifrostContextKeyStreamAdmissionWait, offered)
	if StreamAdmissionWaitFrom(ctx) != offered {
		t.Fatal("the offered wait must be readable from the request context")
	}
}
