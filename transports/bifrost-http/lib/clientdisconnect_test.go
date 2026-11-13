package lib

import (
	"context"
	"errors"
	"testing"

	"github.com/valyala/fasthttp"
)

// ClientDisconnectContext ends when the client closes its socket while the handler is still
// working on the request.
func TestClientDisconnectContextEndsWhenTheClientLeaves(t *testing.T) {
	if !clientDisconnectPeekSupported {
		t.Skip("no socket peeking on this platform, so a client disconnect is not observed")
	}
	started := make(chan struct{})
	outcome := make(chan error, 1)
	client := serveOneConnection(t, func(ctx *fasthttp.RequestCtx) {
		watched, cancel := ClientDisconnectContext(ctx)
		defer cancel()
		close(started)
		<-watched.Done()
		outcome <- watched.Err()
	})
	if _, err := client.Write([]byte(chatCompletionRawRequest)); err != nil {
		t.Fatalf("write request: %v", err)
	}
	<-started
	client.Close()
	if err := <-outcome; !errors.Is(err, context.Canceled) {
		t.Fatalf("context error = %v, want context.Canceled once the client is gone", err)
	}
}

// Its cancel func ends it, and a RequestCtx with no server or connection behind it (as in
// handler unit tests) still yields a usable context.
func TestClientDisconnectContextCancel(t *testing.T) {
	watched, cancel := ClientDisconnectContext(&fasthttp.RequestCtx{})
	if err := watched.Err(); err != nil {
		t.Fatalf("a fresh context must be live, got %v", err)
	}
	cancel()
	<-watched.Done()
	if err := watched.Err(); !errors.Is(err, context.Canceled) {
		t.Fatalf("context error = %v, want context.Canceled", err)
	}
}
