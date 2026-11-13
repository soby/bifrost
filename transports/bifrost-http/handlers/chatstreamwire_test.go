package handlers

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/maximhq/bifrost/core/schemas"
	"github.com/maximhq/bifrost/transports/v2/bifrost-http/lib"
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

// A provider that reports explicit zero token counts keeps them on the requested usage
// event, as the non-streaming response does.
func TestMarshalChatCompletionStreamEvents_UsageEventKeepsExplicitZeroCounts(t *testing.T) {
	var usage schemas.BifrostLLMUsage
	if err := json.Unmarshal([]byte(`{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}`), &usage); err != nil {
		t.Fatal(err)
	}
	response := terminalChatChunk(&usage)
	whole, err := schemas.MarshalSorted(response.WithPreservedUsage())
	if err != nil {
		t.Fatal(err)
	}
	_, usageEvent, err := marshalChatCompletionStreamEvents(response, true)
	if err != nil {
		t.Fatal(err)
	}
	want, _ := json.Marshal(decodeJSONMap(t, whole)["usage"])
	got, _ := json.Marshal(decodeJSONMap(t, usageEvent)["usage"])
	if string(got) != string(want) {
		t.Fatalf("usage event usage = %s, want what the unsplit chunk would carry: %s", got, want)
	}
}

func TestStreamingResponseOmitsUsageUnlessRequested(t *testing.T) {
	terminal := func() *schemas.BifrostStreamChunk {
		chunk := contentChunk("hello")
		chunk.BifrostChatResponse.Choices[0].FinishReason = schemas.Ptr("stop")
		chunk.BifrostChatResponse.Usage = &schemas.BifrostLLMUsage{PromptTokens: 2, CompletionTokens: 1, TotalTokens: 3}
		return chunk
	}

	body := serveChatStreamingResponse(t, &lib.Config{}, nil, false, []*schemas.BifrostStreamChunk{contentChunk("hi"), terminal()})
	if strings.Contains(body, `"usage"`) {
		t.Fatalf("a stream that did not request usage must not carry a usage member:\n%s", body)
	}
	if !strings.HasSuffix(body, "data: [DONE]\n\n") {
		t.Fatalf("stream must end with [DONE]:\n%s", body)
	}

	body = serveChatStreamingResponse(t, &lib.Config{}, nil, true, []*schemas.BifrostStreamChunk{contentChunk("hi"), terminal()})
	events := strings.Split(strings.TrimSuffix(body, "data: [DONE]\n\n"), "\n\n")
	var data []string
	for _, event := range events {
		if strings.HasPrefix(event, "data: ") {
			data = append(data, strings.TrimPrefix(event, "data: "))
		}
	}
	if len(data) != 3 {
		t.Fatalf("want two ordinary chunks and one usage chunk, got %d:\n%s", len(data), body)
	}
	for _, ordinary := range data[:2] {
		wire := decodeJSONMap(t, []byte(ordinary))
		if value, ok := wire["usage"]; !ok || value != nil {
			t.Fatalf("ordinary chunk must carry usage:null: %s", ordinary)
		}
		if len(wire["choices"].([]any)) != 1 {
			t.Fatalf("ordinary chunk lost its choice: %s", ordinary)
		}
	}
	last := decodeJSONMap(t, []byte(data[2]))
	if len(last["choices"].([]any)) != 0 || last["usage"].(map[string]any)["total_tokens"] != float64(3) {
		t.Fatalf("the last chunk before [DONE] must be the usage chunk with empty choices: %s", data[2])
	}
}
