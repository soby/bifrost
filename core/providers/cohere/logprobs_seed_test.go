package cohere

import (
	"encoding/json"
	"testing"

	providerUtils "github.com/maximhq/bifrost/core/providers/utils"
	"github.com/maximhq/bifrost/core/schemas"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func cohereWireBody(t *testing.T, request *CohereChatRequest) map[string]json.RawMessage {
	t.Helper()
	body, err := providerUtils.MarshalProviderRequest(request)
	require.NoError(t, err)
	var wire map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(body, &wire), "%s", body)
	return wire
}

func cohereChatRequest(params *schemas.ChatParameters) *schemas.BifrostChatRequest {
	return &schemas.BifrostChatRequest{
		Provider: schemas.Cohere,
		Model:    "command-a-03-2025",
		Input: []schemas.ChatMessage{{
			Role:    schemas.ChatMessageRoleUser,
			Content: &schemas.ChatMessageContent{ContentStr: schemas.Ptr("Hello")},
		}},
		Params: params,
	}
}

func requireCallerError(t *testing.T, err error, field string) {
	t.Helper()
	badRequest, ok := providerUtils.AsBifrostBadRequestError(err)
	require.True(t, ok, "want a caller error naming %s, got %v", field, err)
	require.NotNil(t, badRequest.StatusCode)
	assert.Equal(t, 400, *badRequest.StatusCode)
	assert.Contains(t, badRequest.Error.Message, field)
}

// Chat logprobs and seed reach Cohere v2's "logprobs" and "seed" fields; the old
// "log_probs" spelling is never sent.
func TestToCohereChatCompletionRequest_LogprobsAndSeedOnWire(t *testing.T) {
	request, err := ToCohereChatCompletionRequest(cohereChatRequest(&schemas.ChatParameters{
		LogProbs: schemas.Ptr(true),
		Seed:     schemas.Ptr(42),
	}))
	require.NoError(t, err)
	wire := cohereWireBody(t, request)
	assert.JSONEq(t, `true`, string(wire["logprobs"]))
	assert.JSONEq(t, `42`, string(wire["seed"]))
	assert.NotContains(t, wire, "log_probs")

	// extra_params.log_probs, the older spelling, still asks for logprobs.
	request, err = ToCohereChatCompletionRequest(cohereChatRequest(&schemas.ChatParameters{
		ExtraParams: map[string]interface{}{"log_probs": true},
	}))
	require.NoError(t, err)
	wire = cohereWireBody(t, request)
	assert.JSONEq(t, `true`, string(wire["logprobs"]))
	assert.NotContains(t, wire, "log_probs")

	request, err = ToCohereChatCompletionRequest(cohereChatRequest(&schemas.ChatParameters{}))
	require.NoError(t, err)
	wire = cohereWireBody(t, request)
	assert.NotContains(t, wire, "logprobs")
	assert.NotContains(t, wire, "seed")
}

// Cohere returns no alternative tokens, so a non-zero top_logprobs is a caller error
// rather than a silent drop; zero asks for nothing and passes.
func TestToCohereChatCompletionRequest_TopLogprobsRefused(t *testing.T) {
	_, err := ToCohereChatCompletionRequest(cohereChatRequest(&schemas.ChatParameters{
		LogProbs:    schemas.Ptr(true),
		TopLogProbs: schemas.Ptr(3),
	}))
	requireCallerError(t, err, "top_logprobs")

	_, err = ToCohereChatCompletionRequest(cohereChatRequest(&schemas.ChatParameters{TopLogProbs: schemas.Ptr(0)}))
	require.NoError(t, err)
}

// A Cohere-dialect request's logprobs and seed become the typed chat parameters, so
// any provider serving it carries them.
func TestCohereChatRequestIngress_LogprobsAndSeedTyped(t *testing.T) {
	var req CohereChatRequest
	require.NoError(t, json.Unmarshal([]byte(`{"model":"command-a-03-2025","messages":[{"role":"user","content":"hi"}],"logprobs":true,"seed":7}`), &req))
	bifrostReq := req.ToBifrostChatRequest(nil)
	require.NotNil(t, bifrostReq.Params)
	require.NotNil(t, bifrostReq.Params.LogProbs)
	assert.True(t, *bifrostReq.Params.LogProbs)
	require.NotNil(t, bifrostReq.Params.Seed)
	assert.Equal(t, 7, *bifrostReq.Params.Seed)
	assert.NotContains(t, bifrostReq.Params.ExtraParams, "log_probs")
}

const cohereLogprobsResponse = `{
	"id": "resp-1",
	"finish_reason": "COMPLETE",
	"message": {"role": "assistant", "content": [{"type": "text", "text": "Hello world"}]},
	"logprobs": [
		{"token_ids": [1], "text": "Hello", "logprobs": [-0.5]},
		{"token_ids": [2, 3], "text": " world", "logprobs": [-0.25, -0.125]}
	]
}`

func assertCohereHelloWorld(t *testing.T, content []schemas.ContentLogProb) {
	t.Helper()
	require.Len(t, content, 2)
	assert.Equal(t, "Hello", content[0].Token)
	assert.InDelta(t, -0.5, content[0].LogProb, 1e-9)
	assert.Equal(t, []int{'H', 'e', 'l', 'l', 'o'}, content[0].Bytes)
	assert.NotNil(t, content[0].TopLogProbs)
	assert.Empty(t, content[0].TopLogProbs)
	assert.Equal(t, " world", content[1].Token)
	assert.InDelta(t, -0.375, content[1].LogProb, 1e-9, "a multi-token chunk carries its tokens' joint logprob")
}

// Cohere's response logprobs reach choices[].logprobs, unary and streaming.
func TestCohereChatResponse_LogprobsReachChoices(t *testing.T) {
	var response CohereChatResponse
	require.NoError(t, json.Unmarshal([]byte(cohereLogprobsResponse), &response))
	bifrostResp := response.ToBifrostChatResponse("command-a-03-2025")
	require.Len(t, bifrostResp.Choices, 1)
	require.NotNil(t, bifrostResp.Choices[0].LogProbs)
	assertCohereHelloWorld(t, bifrostResp.Choices[0].LogProbs.Content)

	var event CohereStreamEvent
	require.NoError(t, json.Unmarshal([]byte(`{"type":"content-delta","index":0,"delta":{"message":{"content":{"text":"Hello"}}},"logprobs":{"token_ids":[1],"text":"Hello","logprobs":[-0.5]}}`), &event))
	chunk, bifrostErr, _ := event.ToBifrostChatCompletionStream()
	require.Nil(t, bifrostErr)
	require.NotNil(t, chunk)
	require.NotNil(t, chunk.Choices[0].LogProbs)
	require.Len(t, chunk.Choices[0].LogProbs.Content, 1)
	assert.Equal(t, "Hello", chunk.Choices[0].LogProbs.Content[0].Token)
	assert.InDelta(t, -0.5, chunk.Choices[0].LogProbs.Content[0].LogProb, 1e-9)
}

func cohereResponsesRequest(params *schemas.ResponsesParameters) *schemas.BifrostResponsesRequest {
	return &schemas.BifrostResponsesRequest{
		Provider: schemas.Cohere,
		Model:    "command-a-03-2025",
		Input: []schemas.ResponsesMessage{{
			Role:    schemas.Ptr(schemas.ResponsesInputMessageRoleUser),
			Content: &schemas.ResponsesMessageContent{ContentStr: schemas.Ptr("Hello")},
		}},
		Params: params,
	}
}

// On the Responses path include "message.output_text.logprobs" asks Cohere for
// logprobs, extra_params.seed reaches seed (Responses has no seed field), and a
// non-zero top_logprobs is a caller error.
func TestToCohereResponsesRequest_LogprobsAndSeed(t *testing.T) {
	extraParams := map[string]interface{}{"seed": 11}
	request, err := ToCohereResponsesRequest(cohereResponsesRequest(&schemas.ResponsesParameters{
		Include:     []string{schemas.ResponsesIncludeOutputTextLogprobs},
		ExtraParams: extraParams,
	}))
	require.NoError(t, err)
	wire := cohereWireBody(t, request)
	assert.JSONEq(t, `true`, string(wire["logprobs"]))
	assert.JSONEq(t, `11`, string(wire["seed"]))
	assert.Contains(t, extraParams, "seed", "the request's own extra params are left intact for a retry or fallback")

	_, err = ToCohereResponsesRequest(cohereResponsesRequest(&schemas.ResponsesParameters{TopLogProbs: schemas.Ptr(2)}))
	requireCallerError(t, err, "top_logprobs")
}

// Cohere's response logprobs reach the Responses output_text logprobs, unary and
// streaming (delta, done events and response.completed).
func TestCohereResponses_LogprobsReachOutputText(t *testing.T) {
	var response CohereChatResponse
	require.NoError(t, json.Unmarshal([]byte(cohereLogprobsResponse), &response))
	bifrostResp := response.ToBifrostResponsesResponse()
	require.NotEmpty(t, bifrostResp.Output)
	require.NotNil(t, bifrostResp.Output[0].Content)
	block := bifrostResp.Output[0].Content.ContentBlocks[0]
	require.NotNil(t, block.ResponsesOutputMessageContentText)
	assertCohereHelloWorld(t, schemas.ChatLogProbsFromResponses(block.LogProbs))

	withLogProbs := func(event CohereStreamEvent, text string, logprob float64) CohereStreamEvent {
		event.LogProbs = &CohereLogProb{TokenIDs: []int{1}, Text: schemas.Ptr(text), LogProbs: []float64{logprob}}
		return event
	}
	state := acquireCohereResponsesStreamState()
	defer releaseCohereResponsesStreamState(state)
	events := runCohereResponsesStream(t, state, []CohereStreamEvent{
		cohereMessageStartEvent("msg-logprobs"),
		cohereContentStartEvent(0, CohereContentBlockTypeText),
		withLogProbs(cohereTextDeltaEvent(0, "Hello"), "Hello", -0.5),
		withLogProbs(cohereTextDeltaEvent(0, " world"), " world", -0.375),
		cohereContentEndEvent(0),
		cohereMessageEndEvent(FinishReasonComplete, 10, 20),
	})

	deltas := filterStreamEvents(events, schemas.ResponsesStreamResponseTypeOutputTextDelta)
	require.Len(t, deltas, 2)
	require.Len(t, deltas[0].LogProbs, 1)
	assert.Equal(t, "Hello", deltas[0].LogProbs[0].Token)
	require.Len(t, deltas[1].LogProbs, 1)
	assert.Equal(t, " world", deltas[1].LogProbs[0].Token)

	textDone := filterStreamEvents(events, schemas.ResponsesStreamResponseTypeOutputTextDone)
	require.Len(t, textDone, 1)
	assertCohereHelloWorld(t, schemas.ChatLogProbsFromResponses(textDone[0].LogProbs))
	partDone := filterStreamEvents(events, schemas.ResponsesStreamResponseTypeContentPartDone)
	require.Len(t, partDone, 1)
	assertCohereHelloWorld(t, schemas.ChatLogProbsFromResponses(partDone[0].Part.LogProbs))

	completed := completedResponse(t, events)
	require.NotEmpty(t, completed.Output)
	require.NotNil(t, completed.Output[0].Content)
	assertCohereHelloWorld(t, schemas.ChatLogProbsFromResponses(completed.Output[0].Content.ContentBlocks[0].LogProbs))
}
