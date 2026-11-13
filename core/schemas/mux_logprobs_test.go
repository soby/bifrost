package schemas

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func helloWorldChatLogProbs() []ContentLogProb {
	return []ContentLogProb{
		{Token: "Hello", LogProb: -0.5, Bytes: []int{72, 101, 108, 108, 111}, TopLogProbs: []LogProb{{Token: "Hi", LogProb: -1}}},
		{Token: " world", LogProb: -0.25, Bytes: []int{32, 119, 111, 114, 108, 100}},
	}
}

func assertHelloWorldResponsesLogProbs(t *testing.T, logProbs []ResponsesOutputMessageContentTextLogProb, msg string) {
	t.Helper()
	require.Len(t, logProbs, 2, msg)
	assert.Equal(t, "Hello", logProbs[0].Token, msg)
	assert.InDelta(t, -0.5, logProbs[0].LogProb, 1e-9, msg)
	assert.Equal(t, []int{72, 101, 108, 108, 111}, logProbs[0].Bytes, msg)
	require.Len(t, logProbs[0].TopLogProbs, 1, msg)
	assert.Equal(t, "Hi", logProbs[0].TopLogProbs[0].Token, msg)
	assert.Equal(t, " world", logProbs[1].Token, msg)
	assert.NotNil(t, logProbs[1].TopLogProbs, "%s: top_logprobs is a list, never null", msg)
}

func assertHelloWorldChatLogProbs(t *testing.T, logProbs *BifrostLogProbs, msg string) {
	t.Helper()
	require.NotNil(t, logProbs, msg)
	require.Len(t, logProbs.Content, 2, msg)
	assert.Equal(t, "Hello", logProbs.Content[0].Token, msg)
	assert.InDelta(t, -0.5, logProbs.Content[0].LogProb, 1e-9, msg)
	require.Len(t, logProbs.Content[0].TopLogProbs, 1, msg)
	assert.Equal(t, " world", logProbs.Content[1].Token, msg)
}

// Chat logprobs: true becomes include "message.output_text.logprobs" and back.
func TestChatResponsesRequestConversion_Logprobs(t *testing.T) {
	chat := &BifrostChatRequest{Params: &ChatParameters{LogProbs: Ptr(true), TopLogProbs: Ptr(4)}}
	responses := chat.ToResponsesRequest()
	require.NotNil(t, responses.Params)
	assert.Equal(t, []string{ResponsesIncludeOutputTextLogprobs}, responses.Params.Include)
	require.NotNil(t, responses.Params.TopLogProbs)
	assert.Equal(t, 4, *responses.Params.TopLogProbs)

	responses = (&BifrostChatRequest{Params: &ChatParameters{LogProbs: Ptr(false)}}).ToResponsesRequest()
	assert.Empty(t, responses.Params.Include)

	back := (&BifrostResponsesRequest{Params: &ResponsesParameters{
		Include:     []string{"reasoning.encrypted_content", ResponsesIncludeOutputTextLogprobs},
		TopLogProbs: Ptr(4),
	}}).ToChatRequest()
	require.NotNil(t, back.Params.LogProbs)
	assert.True(t, *back.Params.LogProbs)
	require.NotNil(t, back.Params.TopLogProbs)
	assert.Equal(t, 4, *back.Params.TopLogProbs)

	back = (&BifrostResponsesRequest{Params: &ResponsesParameters{}}).ToChatRequest()
	assert.Nil(t, back.Params.LogProbs)
}

// A Responses request's extra_params.seed (the API has no seed field) becomes Chat's
// typed seed on the chat fallback; the Responses request keeps its own extra params.
func TestResponsesToChatRequest_SeedFromExtraParams(t *testing.T) {
	extraParams := map[string]interface{}{"seed": 42, "top_k": 5}
	chat := (&BifrostResponsesRequest{Params: &ResponsesParameters{ExtraParams: extraParams}}).ToChatRequest()
	require.NotNil(t, chat.Params.Seed)
	assert.Equal(t, 42, *chat.Params.Seed)
	assert.NotContains(t, chat.Params.ExtraParams, "seed")
	assert.Contains(t, chat.Params.ExtraParams, "top_k")
	assert.Contains(t, extraParams, "seed", "the Responses request's extra params must not be mutated")
}

// A chat response's choices[].logprobs reach the output_text part when it is served
// to a Responses caller, and a Responses output_text's logprobs reach
// choices[].logprobs when served to a Chat caller.
func TestChatResponsesResponseConversion_Logprobs(t *testing.T) {
	chat := &BifrostChatResponse{
		ID: "chatcmpl-1",
		Choices: []BifrostResponseChoice{{
			FinishReason: Ptr("stop"),
			LogProbs:     &BifrostLogProbs{Content: helloWorldChatLogProbs()},
			ChatNonStreamResponseChoice: &ChatNonStreamResponseChoice{Message: &ChatMessage{
				Role:    ChatMessageRoleAssistant,
				Content: &ChatMessageContent{ContentStr: Ptr("Hello world")},
			}},
		}},
	}
	responses := chat.ToBifrostResponsesResponse()
	block := firstOutputTextBlock(responses.Output)
	require.NotNil(t, block)
	assertHelloWorldResponsesLogProbs(t, block.LogProbs, "chat -> responses")

	back := responses.ToBifrostChatResponse()
	require.Len(t, back.Choices, 1)
	assertHelloWorldChatLogProbs(t, back.Choices[0].LogProbs, "responses -> chat")

	// No logprobs: the output_text part keeps an empty list and the choice none.
	chat.Choices[0].LogProbs = nil
	responses = chat.ToBifrostResponsesResponse()
	block = firstOutputTextBlock(responses.Output)
	require.NotNil(t, block)
	assert.NotNil(t, block.LogProbs)
	assert.Empty(t, block.LogProbs)
	assert.Nil(t, responses.ToBifrostChatResponse().Choices[0].LogProbs)
}

// Streaming: each chat chunk's logprobs ride its output_text.delta, the text item's
// done events and the terminal response carry them all, and an output_text.delta's
// logprobs reach the chat chunk's choices[].logprobs.
func TestChatResponsesStreamConversion_Logprobs(t *testing.T) {
	all := helloWorldChatLogProbs()
	chunks := []*BifrostChatResponse{
		{ID: "c1", Choices: []BifrostResponseChoice{{ChatStreamResponseChoice: &ChatStreamResponseChoice{Delta: &ChatStreamResponseChoiceDelta{Role: Ptr("assistant")}}}}},
		{ID: "c1", Choices: []BifrostResponseChoice{{
			LogProbs:                 &BifrostLogProbs{Content: all[:1]},
			ChatStreamResponseChoice: &ChatStreamResponseChoice{Delta: &ChatStreamResponseChoiceDelta{Content: Ptr("Hello")}},
		}}},
		{ID: "c1", Choices: []BifrostResponseChoice{{
			LogProbs:                 &BifrostLogProbs{Content: all[1:]},
			ChatStreamResponseChoice: &ChatStreamResponseChoice{Delta: &ChatStreamResponseChoiceDelta{Content: Ptr(" world")}},
		}}},
		{ID: "c1", Choices: []BifrostResponseChoice{{FinishReason: Ptr("stop"), ChatStreamResponseChoice: &ChatStreamResponseChoice{Delta: &ChatStreamResponseChoiceDelta{}}}}},
	}
	state := AcquireChatToResponsesStreamState()
	var events []*BifrostResponsesStreamResponse
	for _, chunk := range chunks {
		events = append(events, chunk.ToBifrostResponsesStreamResponse(state)...)
	}

	var deltas []*BifrostResponsesStreamResponse
	var textDone, partDone, itemDone, terminal *BifrostResponsesStreamResponse
	for _, event := range events {
		switch event.Type {
		case ResponsesStreamResponseTypeOutputTextDelta:
			deltas = append(deltas, event)
		case ResponsesStreamResponseTypeOutputTextDone:
			textDone = event
		case ResponsesStreamResponseTypeContentPartDone:
			partDone = event
		case ResponsesStreamResponseTypeOutputItemDone:
			itemDone = event
		case ResponsesStreamResponseTypeCompleted:
			terminal = event
		}
	}
	require.Len(t, deltas, 2)
	require.Len(t, deltas[0].LogProbs, 1)
	assert.Equal(t, "Hello", deltas[0].LogProbs[0].Token)
	require.Len(t, deltas[1].LogProbs, 1)
	assert.Equal(t, " world", deltas[1].LogProbs[0].Token)
	require.NotNil(t, textDone)
	assertHelloWorldResponsesLogProbs(t, textDone.LogProbs, "output_text.done")
	require.NotNil(t, partDone)
	assertHelloWorldResponsesLogProbs(t, partDone.Part.LogProbs, "content_part.done")
	require.NotNil(t, itemDone)
	assertHelloWorldResponsesLogProbs(t, firstOutputTextBlock([]ResponsesMessage{*itemDone.Item}).LogProbs, "output_item.done")
	require.NotNil(t, terminal)
	require.NotNil(t, terminal.Response)
	assertHelloWorldResponsesLogProbs(t, firstOutputTextBlock(terminal.Response.Output).LogProbs, "response.completed")

	// Responses stream -> chat chunk.
	chatChunk := deltas[0].ToBifrostChatResponse()
	require.Len(t, chatChunk.Choices, 1)
	require.NotNil(t, chatChunk.Choices[0].LogProbs)
	require.Len(t, chatChunk.Choices[0].LogProbs.Content, 1)
	assert.Equal(t, "Hello", chatChunk.Choices[0].LogProbs.Content[0].Token)
	assert.InDelta(t, -0.5, chatChunk.Choices[0].LogProbs.Content[0].LogProb, 1e-9)

	// The pooled state does not carry one stream's logprobs into the next.
	ReleaseChatToResponsesStreamState(state)
	state = AcquireChatToResponsesStreamState()
	assert.Nil(t, state.TextLogProbs)
	ReleaseChatToResponsesStreamState(state)
}
