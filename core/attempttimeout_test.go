package bifrost

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/maximhq/bifrost/core/schemas"
)

// attemptTimeoutPlugin configures two request-scoped OpenAI attempts the way a gateway with
// two OpenAI-compatible providers of different credentials does: PreRequestHook gives the
// request its configuration, PreLLMHook rewrites it and sets the attempt's own timeout.
type attemptTimeoutPlugin struct {
	t        *testing.T
	urls     [2]string
	timeouts [2]time.Duration
	// seenTimeout records the attempt timeout each attempt found on the context before its
	// PreLLMHook set one.
	seenTimeout [2]atomic.Int64
}

func (p *attemptTimeoutPlugin) GetName() string { return "attempt-timeout-test" }
func (p *attemptTimeoutPlugin) Cleanup() error  { return nil }
func (p *attemptTimeoutPlugin) PreRequestHook(_ *schemas.BifrostContext, req *schemas.BifrostRequest) error {
	mustConfigureOpenAI(p.t, req, "sk-a", p.urls[0])
	return nil
}
func (p *attemptTimeoutPlugin) PreLLMHook(ctx *schemas.BifrostContext, req *schemas.BifrostRequest) (*schemas.BifrostRequest, *schemas.LLMPluginShortCircuit, error) {
	index, _ := ctx.Value(schemas.BifrostContextKeyFallbackIndex).(int)
	if index > 1 {
		return req, nil, nil
	}
	existing, _ := ctx.Value(schemas.BifrostContextKeyAttemptRequestTimeout).(time.Duration)
	p.seenTimeout[index].Store(int64(existing))
	mustConfigureOpenAI(p.t, req, fmt.Sprintf("sk-%c", 'a'+index), p.urls[index])
	if p.timeouts[index] > 0 {
		ctx.SetValue(schemas.BifrostContextKeyAttemptRequestTimeout, p.timeouts[index])
	}
	return req, nil, nil
}
func (p *attemptTimeoutPlugin) PostLLMHook(_ *schemas.BifrostContext, resp *schemas.BifrostResponse, err *schemas.BifrostError) (*schemas.BifrostResponse, *schemas.BifrostError, error) {
	return resp, err, nil
}

// hangingUpstream accepts requests and never answers until the test ends.
func hangingUpstream(t *testing.T, calls *atomic.Int64) *httptest.Server {
	t.Helper()
	release := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		select {
		case <-release:
		case <-r.Context().Done():
		case <-time.After(10 * time.Second):
		}
	}))
	t.Cleanup(func() {
		close(release)
		server.Close()
	})
	return server
}

// answeringUpstream answers chat completions (JSON or SSE) after delay. record, when set,
// receives every request body.
func answeringUpstream(t *testing.T, calls *atomic.Int64, delay time.Duration, record func(body string)) *httptest.Server {
	t.Helper()
	stream := sseHandler(`{"id":"c1","object":"chat.completion.chunk","created":1,"model":"model-b","choices":[{"index":0,"delta":{"role":"assistant","content":"hello"},"finish_reason":null}]}`,
		`{"id":"c1","object":"chat.completion.chunk","created":1,"model":"model-b","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		if record != nil {
			body, _ := io.ReadAll(r.Body)
			record(string(body))
		}
		time.Sleep(delay)
		if r.Header.Get("Accept") == "text/event-stream" {
			stream(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"c1","object":"chat.completion","created":1,"model":"model-b","choices":[{"index":0,"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
	}))
	t.Cleanup(server.Close)
	return server
}

func TestAttemptRequestTimeout_PrimaryTimeoutRunsFallback(t *testing.T) {
	for _, streaming := range []bool{false, true} {
		t.Run(fmt.Sprintf("stream=%t", streaming), func(t *testing.T) {
			var primaryCalls, fallbackCalls atomic.Int64
			primary := hangingUpstream(t, &primaryCalls)
			fallback := answeringUpstream(t, &fallbackCalls, 0, nil)
			plugin := &attemptTimeoutPlugin{t: t, urls: [2]string{primary.URL, fallback.URL}, timeouts: [2]time.Duration{200 * time.Millisecond, 0}}
			client := newRequestScopedTestClient(t, NewMockAccount(), plugin, nil)

			ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
			request := requestScopedChat(schemas.OpenAI, "model-a", schemas.Fallback{Provider: schemas.OpenAI, Model: "model-b"})
			started := time.Now()
			var served schemas.ModelProvider
			var servedModel string
			if streaming {
				stream, err := client.ChatCompletionStreamRequest(ctx, request)
				if err != nil {
					t.Fatalf("stream request: %s", err.GetErrorString())
				}
				for chunk := range stream {
					if chunk.BifrostError != nil {
						t.Fatalf("stream error chunk: %s", chunk.BifrostError.GetErrorString())
					}
					if chunk.BifrostChatResponse != nil {
						served = chunk.BifrostChatResponse.ExtraFields.RoutingInfo.Provider
						servedModel = chunk.BifrostChatResponse.ExtraFields.RoutingInfo.Model
					}
				}
			} else {
				resp, err := client.ChatCompletionRequest(ctx, request)
				if err != nil {
					t.Fatalf("request: %s", err.GetErrorString())
				}
				served = resp.ExtraFields.RoutingInfo.Provider
				servedModel = resp.ExtraFields.RoutingInfo.Model
			}
			elapsed := time.Since(started)

			if served != schemas.OpenAI || servedModel != "model-b" {
				t.Errorf("served by %s/%s, want the openai/model-b fallback", served, servedModel)
			}
			if primaryCalls.Load() != 1 || fallbackCalls.Load() != 1 {
				t.Errorf("upstream calls primary=%d fallback=%d, want 1 and 1", primaryCalls.Load(), fallbackCalls.Load())
			}
			if elapsed > 5*time.Second {
				t.Errorf("request took %s, want the primary cut off near its 200ms attempt timeout", elapsed)
			}
			if ctx.Err() != nil {
				t.Errorf("the attempt timeout cancelled the request context: %v", ctx.Err())
			}
		})
	}
}

func TestAttemptRequestTimeout_ClearedBeforeFallback(t *testing.T) {
	var primaryCalls, fallbackCalls atomic.Int64
	primary := hangingUpstream(t, &primaryCalls)
	// Slower than the primary's timeout: the fallback only succeeds if that timeout did not
	// carry over to it.
	fallback := answeringUpstream(t, &fallbackCalls, 600*time.Millisecond, nil)
	plugin := &attemptTimeoutPlugin{t: t, urls: [2]string{primary.URL, fallback.URL}, timeouts: [2]time.Duration{200 * time.Millisecond, 0}}
	client := newRequestScopedTestClient(t, NewMockAccount(), plugin, nil)

	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	resp, err := client.ChatCompletionRequest(ctx, requestScopedChat(schemas.OpenAI, "model-a", schemas.Fallback{Provider: schemas.OpenAI, Model: "model-b"}))
	if err != nil {
		t.Fatalf("request: %s", err.GetErrorString())
	}
	if resp.ExtraFields.RoutingInfo.Model != "model-b" {
		t.Errorf("served model %q, want the fallback's model-b", resp.ExtraFields.RoutingInfo.Model)
	}
	if got := time.Duration(plugin.seenTimeout[1].Load()); got != 0 {
		t.Errorf("the fallback attempt found attempt timeout %s on the context, want it cleared", got)
	}
}

// TestAttemptRequestTimeout_ReleasesSlotAndShutdown pins that a request-scoped attempt cut off
// by its attempt timeout gives back its slot and its place among the attempts Shutdown waits
// for: Shutdown, started while the attempt hangs, returns once the timeout fires rather than
// when the upstream gives up.
func TestAttemptRequestTimeout_ReleasesSlotAndShutdown(t *testing.T) {
	for _, streaming := range []bool{false, true} {
		t.Run(fmt.Sprintf("stream=%t", streaming), func(t *testing.T) {
			var calls atomic.Int64
			upstream := hangingUpstream(t, &calls)
			plugin := &attemptTimeoutPlugin{t: t, urls: [2]string{upstream.URL, upstream.URL}, timeouts: [2]time.Duration{200 * time.Millisecond, 0}}
			client := newRequestScopedTestClient(t, NewMockAccount(), plugin, nil)
			instance, err := client.getRequestScopedProvider(requestScopedClass{provider: schemas.OpenAI, allowPrivateNetwork: true})
			if err != nil {
				t.Fatal(err)
			}

			result := make(chan *schemas.BifrostError, 1)
			go func() {
				ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
				if streaming {
					stream, err := client.ChatCompletionStreamRequest(ctx, requestScopedChat(schemas.OpenAI, "model-a"))
					if err == nil {
						for chunk := range stream {
							if chunk.BifrostError != nil {
								err = chunk.BifrostError
							}
						}
					}
					result <- err
					return
				}
				_, err := client.ChatCompletionRequest(ctx, requestScopedChat(schemas.OpenAI, "model-a"))
				result <- err
			}()
			for calls.Load() == 0 {
				time.Sleep(time.Millisecond)
			}
			started := time.Now()
			shutdown := make(chan struct{})
			go func() {
				client.Shutdown()
				close(shutdown)
			}()
			select {
			case <-shutdown:
			case <-time.After(5 * time.Second):
				t.Fatal("Shutdown still waiting 5s after a timed-out attempt")
			}
			if elapsed := time.Since(started); elapsed > 2*time.Second {
				t.Errorf("Shutdown took %s, want it to return near the 200ms attempt timeout", elapsed)
			}
			if err := <-result; err == nil || err.StatusCode == nil || *err.StatusCode != http.StatusGatewayTimeout {
				t.Errorf("error = %v, want the 504 attempt timeout", err)
			}
			if n := len(instance.slots); n != 0 {
				t.Errorf("%d request-scoped slots still taken after the attempt timed out", n)
			}
		})
	}
}
