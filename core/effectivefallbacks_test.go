package bifrost

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/maximhq/bifrost/core/schemas"
)

// replacingRoutePlugin selects the route in PreLLMHook the way the gateway's supervision
// runtime does: on the primary attempt it returns a new request with rewritten content and
// the fallbacks it chose. Later attempts receive copies of that request.
type replacingRoutePlugin struct {
	t        *testing.T
	urls     []string
	contents []string // content each attempt's PreLLMHook received
	mu       sync.Mutex
}

func (p *replacingRoutePlugin) GetName() string { return "replacing-route-test" }
func (p *replacingRoutePlugin) Cleanup() error  { return nil }
func (p *replacingRoutePlugin) PreRequestHook(_ *schemas.BifrostContext, req *schemas.BifrostRequest) error {
	// Request-scoped configuration defers provider resolution until after PreLLMHook.
	mustConfigureOpenAI(p.t, req, "sk-0", p.urls[0])
	return nil
}
func (p *replacingRoutePlugin) PreLLMHook(ctx *schemas.BifrostContext, req *schemas.BifrostRequest) (*schemas.BifrostRequest, *schemas.LLMPluginShortCircuit, error) {
	index, _ := ctx.Value(schemas.BifrostContextKeyFallbackIndex).(int)
	p.mu.Lock()
	p.contents = append(p.contents, *req.ChatRequest.Input[0].Content.ContentStr)
	p.mu.Unlock()
	if index == 0 {
		replaced := *req
		chat := *req.ChatRequest
		rewritten := "rewritten by the primary's pre-hook"
		chat.Input = []schemas.ChatMessage{{Role: schemas.ChatMessageRoleUser, Content: &schemas.ChatMessageContent{ContentStr: &rewritten}}}
		chat.Fallbacks = []schemas.Fallback{{Provider: schemas.OpenAI, Model: "model-b"}}
		replaced.ChatRequest = &chat
		mustConfigureOpenAI(p.t, &replaced, "sk-0", p.urls[0])
		return &replaced, nil, nil
	}
	mustConfigureOpenAI(p.t, req, fmt.Sprintf("sk-%d", index), p.urls[index])
	return req, nil, nil
}
func (p *replacingRoutePlugin) PostLLMHook(_ *schemas.BifrostContext, resp *schemas.BifrostResponse, err *schemas.BifrostError) (*schemas.BifrostResponse, *schemas.BifrostError, error) {
	return resp, err, nil
}

func TestEffectiveFallbacks_FollowRequestReturnedByPrimaryPreLLMHook(t *testing.T) {
	for _, streaming := range []bool{false, true} {
		t.Run(fmt.Sprintf("stream=%t", streaming), func(t *testing.T) {
			failing := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusInternalServerError)
				_, _ = fmt.Fprint(w, `{"error":{"message":"upstream failed","type":"server_error"}}`)
			}))
			t.Cleanup(failing.Close)
			var fallbackBodies []string
			var mu sync.Mutex
			var fallbackCalls atomic.Int64
			recording := answeringUpstream(t, &fallbackCalls, 0, func(body string) {
				mu.Lock()
				fallbackBodies = append(fallbackBodies, body)
				mu.Unlock()
			})

			plugin := &replacingRoutePlugin{t: t, urls: []string{failing.URL, recording.URL}}
			client := newRequestScopedTestClient(t, NewMockAccount(), plugin, nil)
			ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
			// The caller's request carries no fallbacks: only the pre-hook's replacement does.
			request := requestScopedChat(schemas.OpenAI, "model-a")

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
						servedModel = chunk.BifrostChatResponse.ExtraFields.RoutingInfo.Model
					}
				}
			} else {
				resp, err := client.ChatCompletionRequest(ctx, request)
				if err != nil {
					t.Fatalf("request: %s", err.GetErrorString())
				}
				servedModel = resp.ExtraFields.RoutingInfo.Model
			}
			if servedModel != "model-b" {
				t.Errorf("served model %q, want the fallback the primary's pre-hook attached", servedModel)
			}
			mu.Lock()
			defer mu.Unlock()
			if len(fallbackBodies) != 1 || !strings.Contains(fallbackBodies[0], "rewritten by the primary's pre-hook") {
				t.Errorf("fallback bodies %q, want one carrying the primary pre-hook's rewritten content", fallbackBodies)
			}
			plugin.mu.Lock()
			defer plugin.mu.Unlock()
			if len(plugin.contents) != 2 || plugin.contents[1] != "rewritten by the primary's pre-hook" {
				t.Errorf("pre-hook inputs %q, want the fallback attempt to start from the replaced request", plugin.contents)
			}
		})
	}
}
