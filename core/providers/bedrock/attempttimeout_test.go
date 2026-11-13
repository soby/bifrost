package bedrock

import (
	"bufio"
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/maximhq/bifrost/core/schemas"
	"github.com/stretchr/testify/require"
)

// The provider's request timeout in these tests is 1 s and stands in for
// default_request_timeout_in_seconds (300 s); slowAnswer is the long call it must not cut on
// an instance with context-bound reads.
const slowAnswer = 1500 * time.Millisecond

// newSlowBedrockTarget serves Bedrock runtime calls after delay, or when the test ends, and
// returns a provider built by NewBedrockProvider with a 1 s request timeout and a key whose
// runtime endpoint is the server.
func newSlowBedrockTarget(t *testing.T, delay time.Duration, contextBoundReads bool) (*BedrockProvider, schemas.Key) {
	t.Helper()
	release := make(chan struct{})
	server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-release:
		case <-r.Context().Done():
		case <-time.After(delay):
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	t.Cleanup(func() {
		close(release)
		server.Close()
	})
	config := &schemas.ProviderConfig{NetworkConfig: schemas.NetworkConfig{
		DefaultRequestTimeoutInSeconds: 1,
		InsecureSkipVerify:             true,
		ContextBoundReads:              contextBoundReads,
	}}
	provider, err := NewBedrockProvider(config, noopLogger{})
	require.NoError(t, err)
	host := strings.TrimPrefix(server.URL, "https://")
	key := schemas.Key{
		Value: *schemas.NewSecretVar("bedrock-api-key"),
		BedrockKeyConfig: &schemas.BedrockKeyConfig{
			Region:    schemas.NewSecretVar("us-east-1"),
			Endpoints: &schemas.BedrockEndpoints{Runtime: schemas.NewSecretVar(host)},
		},
	}
	return provider, key
}

func runSlowConverse(t *testing.T, provider *BedrockProvider, key schemas.Key, ctx *schemas.BifrostContext) (string, *schemas.BifrostError, time.Duration) {
	t.Helper()
	started := time.Now()
	body, _, _, bifrostErr := provider.completeRequest(ctx, []byte(`{}`), "test-model/converse", key, "test-model")
	return string(body), bifrostErr, time.Since(started)
}

func bedrockTestContext(t *testing.T, deadline time.Time) *schemas.BifrostContext {
	t.Helper()
	ctx := schemas.NewBifrostContext(context.Background(), deadline)
	t.Cleanup(ctx.Cancel)
	return ctx
}

func requireTimedOut(t *testing.T, bifrostErr *schemas.BifrostError) {
	t.Helper()
	require.NotNil(t, bifrostErr, "the call must fail")
	require.NotNil(t, bifrostErr.Error)
	require.NotNil(t, bifrostErr.Error.Type, "error %v has no type", bifrostErr.Error.Error)
	require.Equal(t, schemas.RequestTimedOut, *bifrostErr.Error.Type, "a timeout lets fallbacks run")
	require.NotNil(t, bifrostErr.StatusCode)
	require.Equal(t, 504, *bifrostErr.StatusCode)
}

func TestContextBoundReads_BedrockUnaryOutlivesRequestTimeout(t *testing.T) {
	provider, key := newSlowBedrockTarget(t, slowAnswer, true)
	body, bifrostErr, elapsed := runSlowConverse(t, provider, key, bedrockTestContext(t, schemas.NoDeadline))
	require.Nil(t, bifrostErr, "a call slower than the request timeout failed after %s", elapsed)
	require.JSONEq(t, `{"ok":true}`, body)
	require.GreaterOrEqual(t, elapsed, slowAnswer)
}

func TestContextBoundReads_ConfiguredBedrockKeepsRequestTimeout(t *testing.T) {
	provider, key := newSlowBedrockTarget(t, slowAnswer, false)
	_, bifrostErr, elapsed := runSlowConverse(t, provider, key, bedrockTestContext(t, schemas.NoDeadline))
	requireTimedOut(t, bifrostErr)
	require.Less(t, elapsed, slowAnswer, "a configured provider's request timeout no longer bounds the call")
}

func TestAttemptRequestTimeout_CutsBedrockUnary(t *testing.T) {
	for _, contextBoundReads := range []bool{true, false} {
		provider, key := newSlowBedrockTarget(t, 5*time.Second, contextBoundReads)
		ctx := bedrockTestContext(t, schemas.NoDeadline)
		ctx.SetValue(schemas.BifrostContextKeyAttemptRequestTimeout, 300*time.Millisecond)

		_, bifrostErr, elapsed := runSlowConverse(t, provider, key, ctx)
		requireTimedOut(t, bifrostErr)
		require.ErrorIs(t, bifrostErr.Error.Error, context.DeadlineExceeded)
		require.ErrorContains(t, bifrostErr.Error.Error, "attempt request timeout", "the attempt timeout, not the request context, ended the call")
		require.Less(t, elapsed, 900*time.Millisecond, "context-bound=%v: call returned after %s, want about the 300ms attempt timeout", contextBoundReads, elapsed)
		require.NoError(t, ctx.Err(), "the attempt timeout must not cancel the request context")
	}
}

func TestContextDeadline_CutsBedrockUnary(t *testing.T) {
	provider, key := newSlowBedrockTarget(t, 5*time.Second, true)
	_, bifrostErr, elapsed := runSlowConverse(t, provider, key, bedrockTestContext(t, time.Now().Add(300*time.Millisecond)))
	requireTimedOut(t, bifrostErr)
	require.Less(t, elapsed, 900*time.Millisecond, "call returned after %s, want about the 300ms context deadline", elapsed)
}

func TestContextCancellation_CutsBedrockUnary(t *testing.T) {
	provider, key := newSlowBedrockTarget(t, 5*time.Second, true)
	ctx := bedrockTestContext(t, schemas.NoDeadline)
	ctx.SetValue(schemas.BifrostContextKeyAttemptRequestTimeout, 5*time.Second)
	time.AfterFunc(300*time.Millisecond, ctx.Cancel)

	_, bifrostErr, elapsed := runSlowConverse(t, provider, key, ctx)
	require.NotNil(t, bifrostErr)
	require.NotNil(t, bifrostErr.Error.Type)
	require.Equal(t, schemas.RequestCancelled, *bifrostErr.Error.Type, "cancelling the request must not read as an attempt timeout")
	require.Less(t, elapsed, 900*time.Millisecond)
}

// newBedrockStreamTarget sends response headers after headerDelay, then a first chunk, and a
// second one bodyGap later.
func newBedrockStreamTarget(t *testing.T, headerDelay, bodyGap time.Duration) (*BedrockProvider, schemas.Key) {
	t.Helper()
	release := make(chan struct{})
	server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-time.After(headerDelay):
		case <-r.Context().Done():
			return
		case <-release:
			return
		}
		w.Header().Set("Content-Type", "application/vnd.amazon.eventstream")
		w.WriteHeader(http.StatusOK)
		bw := bufio.NewWriter(w)
		_, _ = bw.WriteString("first")
		_ = bw.Flush()
		w.(http.Flusher).Flush()
		select {
		case <-time.After(bodyGap):
		case <-r.Context().Done():
			return
		}
		_, _ = w.Write([]byte("second"))
	}))
	t.Cleanup(func() {
		close(release)
		server.Close()
	})
	config := &schemas.ProviderConfig{NetworkConfig: schemas.NetworkConfig{
		DefaultRequestTimeoutInSeconds: 1,
		InsecureSkipVerify:             true,
		ContextBoundReads:              true,
	}}
	provider, err := NewBedrockProvider(config, noopLogger{})
	require.NoError(t, err)
	host := strings.TrimPrefix(server.URL, "https://")
	return provider, schemas.Key{
		Value: *schemas.NewSecretVar("bedrock-api-key"),
		BedrockKeyConfig: &schemas.BedrockKeyConfig{
			Region:    schemas.NewSecretVar("us-east-1"),
			Endpoints: &schemas.BedrockEndpoints{Runtime: schemas.NewSecretVar(host)},
		},
	}
}

func TestContextBoundReads_BedrockStreamHeaderWaitOutlivesRequestTimeout(t *testing.T) {
	provider, key := newBedrockStreamTarget(t, slowAnswer, 0)
	resp, bifrostErr := provider.makeStreamingRequest(bedrockTestContext(t, schemas.NoDeadline), []byte(`{}`), key, "test-model", "converse-stream")
	require.Nil(t, bifrostErr, "a stream whose headers come after the request timeout failed")
	defer func() { _ = resp.Body.Close() }()
	require.Equal(t, http.StatusOK, resp.StatusCode)
}

func TestAttemptRequestTimeout_BoundsBedrockStreamHeaderWaitOnly(t *testing.T) {
	provider, key := newBedrockStreamTarget(t, 5*time.Second, 0)
	ctx := bedrockTestContext(t, schemas.NoDeadline)
	ctx.SetValue(schemas.BifrostContextKeyAttemptRequestTimeout, 300*time.Millisecond)
	started := time.Now()
	_, bifrostErr := provider.makeStreamingRequest(ctx, []byte(`{}`), key, "test-model", "converse-stream")
	requireTimedOut(t, bifrostErr)
	require.Less(t, time.Since(started), 900*time.Millisecond)
	require.NoError(t, ctx.Err(), "the attempt timeout must not cancel the request context")

	// Headers in time: the body may take longer than the attempt timeout.
	provider, key = newBedrockStreamTarget(t, 0, 600*time.Millisecond)
	ctx = bedrockTestContext(t, schemas.NoDeadline)
	ctx.SetValue(schemas.BifrostContextKeyAttemptRequestTimeout, 300*time.Millisecond)
	resp, bifrostErr := provider.makeStreamingRequest(ctx, []byte(`{}`), key, "test-model", "converse-stream")
	require.Nil(t, bifrostErr)
	defer func() { _ = resp.Body.Close() }()
	body := new(strings.Builder)
	buf := make([]byte, 64)
	for {
		n, err := resp.Body.Read(buf)
		body.Write(buf[:n])
		if err != nil {
			require.ErrorIs(t, err, io.EOF, "reading a body slower than the attempt timeout")
			break
		}
	}
	require.Equal(t, "firstsecond", body.String())
}
