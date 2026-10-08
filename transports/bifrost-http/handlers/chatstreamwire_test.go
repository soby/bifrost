package handlers

import (
	"encoding/json"
	"testing"

	providerUtils "github.com/maximhq/bifrost/core/providers/utils"
	"github.com/maximhq/bifrost/core/schemas"
)

func terminalChatChunk(usage *schemas.BifrostLLMUsage) *schemas.BifrostChatResponse {
	content := "hello"
	stop := "stop"
	return &schemas.BifrostChatResponse{
		ID:      "chatcmpl-test",
		Object:  "chat.completion.chunk",
		Created: 1,
		Model:   "test-model",
		Choices: []schemas.BifrostResponseChoice{{
			Index:        0,
			FinishReason: &stop,
			ChatStreamResponseChoice: &schemas.ChatStreamResponseChoice{
				Delta: &schemas.ChatStreamResponseChoiceDelta{Content: &content},
			},
		}},
		Usage: usage,
	}
}

func decodeJSONMap(t *testing.T, encoded []byte) map[string]any {
	t.Helper()
	var value map[string]any
	if err := json.Unmarshal(encoded, &value); err != nil {
		t.Fatalf("unmarshal wire event: %v", err)
	}
	return value
}

func TestMarshalChatCompletionStreamEvents_DefaultOmitsUsage(t *testing.T) {
	response := terminalChatChunk(&schemas.BifrostLLMUsage{PromptTokens: 2, CompletionTokens: 1, TotalTokens: 3})
	primary, usage, err := marshalChatCompletionStreamEvents(response, false)
	if err != nil {
		t.Fatal(err)
	}
	if usage != nil {
		t.Fatalf("unexpected usage event: %s", usage)
	}
	wire := decodeJSONMap(t, primary)
	if _, ok := wire["usage"]; ok {
		t.Fatalf("default stream exposed usage: %s", primary)
	}
	choices := wire["choices"].([]any)
	choice := choices[0].(map[string]any)
	if _, ok := choice["delta"]; !ok {
		t.Fatalf("terminal choice is not a delta: %s", primary)
	}
	if _, ok := choice["message"]; ok {
		t.Fatalf("terminal choice contains message: %s", primary)
	}
	if response.Usage == nil {
		t.Fatal("wire normalization mutated internal accounting response")
	}
}

func TestMarshalChatCompletionStreamEvents_IncludeUsageSplitsTerminalChunk(t *testing.T) {
	response := terminalChatChunk(&schemas.BifrostLLMUsage{PromptTokens: 2, CompletionTokens: 1, TotalTokens: 3})
	primary, usage, err := marshalChatCompletionStreamEvents(response, true)
	if err != nil {
		t.Fatal(err)
	}
	primaryWire := decodeJSONMap(t, primary)
	if value, ok := primaryWire["usage"]; !ok || value != nil {
		t.Fatalf("ordinary include_usage chunk must carry usage:null: %s", primary)
	}
	if len(primaryWire["choices"].([]any)) != 1 {
		t.Fatalf("ordinary terminal chunk lost its choice: %s", primary)
	}

	usageWire := decodeJSONMap(t, usage)
	if len(usageWire["choices"].([]any)) != 0 {
		t.Fatalf("usage event choices must be empty: %s", usage)
	}
	usageValue, ok := usageWire["usage"].(map[string]any)
	if !ok || usageValue["total_tokens"] != float64(3) {
		t.Fatalf("usage event missing totals: %s", usage)
	}
}

func TestMarshalChatCompletionStreamEvents_IncludeUsageAddsNullToOrdinaryChunk(t *testing.T) {
	response := terminalChatChunk(nil)
	response.Choices[0].FinishReason = nil
	primary, usage, err := marshalChatCompletionStreamEvents(response, true)
	if err != nil {
		t.Fatal(err)
	}
	if usage != nil {
		t.Fatalf("unexpected usage event: %s", usage)
	}
	wire := decodeJSONMap(t, primary)
	if value, ok := wire["usage"]; !ok || value != nil {
		t.Fatalf("ordinary include_usage chunk must carry usage:null: %s", primary)
	}
}

func TestMarshalChatCompletionStreamEvents_UsageOnlyChunkHonorsRequest(t *testing.T) {
	response := terminalChatChunk(&schemas.BifrostLLMUsage{TotalTokens: 3})
	response.Choices = []schemas.BifrostResponseChoice{}

	primary, usage, err := marshalChatCompletionStreamEvents(response, false)
	if err != nil {
		t.Fatal(err)
	}
	if primary != nil || usage != nil {
		t.Fatalf("default stream exposed usage-only event: primary=%s usage=%s", primary, usage)
	}

	primary, usage, err = marshalChatCompletionStreamEvents(response, true)
	if err != nil {
		t.Fatal(err)
	}
	if primary != nil {
		t.Fatalf("usage-only response produced ordinary event: %s", primary)
	}
	wire := decodeJSONMap(t, usage)
	if len(wire["choices"].([]any)) != 0 {
		t.Fatalf("usage-only event choices must be empty: %s", usage)
	}
}

// PLATFORM-4090: the synthetic post-finish accounting chunk keeps the wire
// shape released Pretxt NeMo Relay workers accept: no finish_reason or logprobs
// member on its content-free choice. The bytes are pinned so a fork refresh
// cannot silently change them again.
func TestMarshalChatCompletionStreamEvents_AccountingChunkOmitsNullTerminalFields(t *testing.T) {
	response := providerUtils.CreateBifrostChatCompletionChunkResponse("chatcmpl-test", nil, nil, 24, "test-model", 1)
	for _, includeUsage := range []bool{false, true} {
		primary, usage, err := marshalChatCompletionStreamEvents(response, includeUsage)
		if err != nil {
			t.Fatal(err)
		}
		want := `{"id":"chatcmpl-test","choices":[{"index":0,"delta":{}}],"created":1,"model":"test-model","object":"chat.completion.chunk","system_fingerprint":"","extra_fields":{"request_type":"","routing_info":{},"latency":0,"chunk_index":25}}`
		if includeUsage {
			want = want[:len(want)-1] + `,"usage":null}`
		}
		if usage != nil || string(primary) != want {
			t.Fatalf("accounting chunk wire shape changed (include_usage=%v):\n got %s\nwant %s", includeUsage, primary, want)
		}
	}
	if response.Choices[0].FinishReason != nil || response.Choices[0].LogProbs != nil {
		t.Fatal("wire normalization mutated the internal accounting response")
	}
}

// Only that content-free choice changes: content, role, tool-call and finish
// chunks keep upstream's required nullable members.
func TestMarshalChatCompletionStreamEvents_OtherChunksKeepNullTerminalFields(t *testing.T) {
	content, role, stop := "hi", "assistant", "stop"
	for name, choice := range map[string]schemas.BifrostResponseChoice{
		"content": {ChatStreamResponseChoice: &schemas.ChatStreamResponseChoice{Delta: &schemas.ChatStreamResponseChoiceDelta{Content: &content}}},
		"role":    {ChatStreamResponseChoice: &schemas.ChatStreamResponseChoice{Delta: &schemas.ChatStreamResponseChoiceDelta{Role: &role}}},
		"tool calls": {ChatStreamResponseChoice: &schemas.ChatStreamResponseChoice{Delta: &schemas.ChatStreamResponseChoiceDelta{
			ToolCalls: []schemas.ChatAssistantMessageToolCall{{Index: 0}},
		}}},
		"second index": {Index: 1, ChatStreamResponseChoice: &schemas.ChatStreamResponseChoice{Delta: &schemas.ChatStreamResponseChoiceDelta{}}},
	} {
		t.Run(name, func(t *testing.T) {
			response := &schemas.BifrostChatResponse{ID: "chatcmpl-test", Object: "chat.completion.chunk", Created: 1, Model: "test-model", Choices: []schemas.BifrostResponseChoice{choice}}
			primary, _, err := marshalChatCompletionStreamEvents(response, false)
			if err != nil {
				t.Fatal(err)
			}
			wire := decodeJSONMap(t, primary)["choices"].([]any)[0].(map[string]any)
			for _, key := range []string{"finish_reason", "logprobs"} {
				if value, ok := wire[key]; !ok || value != nil {
					t.Fatalf("%s lost upstream's null %s: %s", name, key, primary)
				}
			}
		})
	}
	finish := terminalChatChunk(nil)
	finish.Choices[0].FinishReason = &stop
	primary, _, err := marshalChatCompletionStreamEvents(finish, false)
	if err != nil {
		t.Fatal(err)
	}
	if wire := decodeJSONMap(t, primary)["choices"].([]any)[0].(map[string]any); wire["finish_reason"] != "stop" {
		t.Fatalf("finish chunk lost its finish reason: %s", primary)
	}
}
