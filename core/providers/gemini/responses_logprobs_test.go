package gemini

import (
	"encoding/json"
	"math"
	"testing"

	providerUtils "github.com/maximhq/bifrost/core/providers/utils"
	"github.com/maximhq/bifrost/core/schemas"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func logprobsResponsesRequest(provider schemas.ModelProvider, params *schemas.ResponsesParameters) *schemas.BifrostResponsesRequest {
	return &schemas.BifrostResponsesRequest{
		Provider: provider,
		Model:    "gemini-2.5-flash",
		Input: []schemas.ResponsesMessage{{
			Role:    schemas.Ptr(schemas.ResponsesInputMessageRoleUser),
			Content: &schemas.ResponsesMessageContent{ContentStr: schemas.Ptr("Hello")},
		}},
		Params: params,
	}
}

func responsesGenerationConfigOnWire(t *testing.T, request *GeminiGenerationRequest) map[string]json.RawMessage {
	t.Helper()
	body, err := providerUtils.MarshalProviderRequest(request)
	require.NoError(t, err)
	var wire struct {
		GenerationConfig map[string]json.RawMessage `json:"generationConfig"`
	}
	require.NoError(t, json.Unmarshal(body, &wire), "%s", body)
	return wire.GenerationConfig
}

// include "message.output_text.logprobs" and top_logprobs reach generationConfig on
// Gemini and Vertex (which shares the Gemini Responses conversion).
func TestToGeminiResponsesRequest_LogprobsReachGenerationConfig(t *testing.T) {
	for _, provider := range []schemas.ModelProvider{schemas.Gemini, schemas.Vertex} {
		request, err := ToGeminiResponsesRequest(nil, logprobsResponsesRequest(provider, &schemas.ResponsesParameters{
			Include: []string{schemas.ResponsesIncludeOutputTextLogprobs},
		}))
		require.NoError(t, err)
		config := responsesGenerationConfigOnWire(t, request)
		assert.JSONEq(t, `true`, string(config["responseLogprobs"]), "%s include", provider)
		assert.NotContains(t, config, "logprobs", "%s: include alone asks for no alternatives", provider)

		for _, count := range []int{1, 5, 20, 21, -1} {
			request, err := ToGeminiResponsesRequest(nil, logprobsResponsesRequest(provider, &schemas.ResponsesParameters{
				TopLogProbs: schemas.Ptr(count),
			}))
			require.NoError(t, err, "%s top_logprobs %d", provider, count)
			config := responsesGenerationConfigOnWire(t, request)
			assert.JSONEq(t, `true`, string(config["responseLogprobs"]), "%s top_logprobs %d", provider, count)
			assert.JSONEq(t, string(mustMarshal(t, count)), string(config["logprobs"]), "%s top_logprobs %d", provider, count)
		}

		request, err = ToGeminiResponsesRequest(nil, logprobsResponsesRequest(provider, &schemas.ResponsesParameters{TopLogProbs: schemas.Ptr(0)}))
		require.NoError(t, err)
		config = responsesGenerationConfigOnWire(t, request)
		assert.NotContains(t, config, "responseLogprobs", "%s: top_logprobs 0 asks for nothing", provider)
		assert.NotContains(t, config, "logprobs", "%s: top_logprobs 0 asks for nothing", provider)

		_, err = ToGeminiResponsesRequest(nil, logprobsResponsesRequest(provider, &schemas.ResponsesParameters{TopLogProbs: schemas.Ptr(math.MaxInt32 + 1)}))
		badRequest, ok := providerUtils.AsBifrostBadRequestError(err)
		require.True(t, ok, "%s: a top_logprobs outside int32 must be a caller error, got %v", provider, err)
		require.NotNil(t, badRequest.StatusCode)
		assert.Equal(t, 400, *badRequest.StatusCode)
		assert.Contains(t, badRequest.Error.Message, "top_logprobs")
	}
}

func logprobsResult() *LogprobsResult {
	return &LogprobsResult{
		ChosenCandidates: []*LogprobsResultCandidate{
			{Token: "Hello", LogProbability: -0.5},
			{Token: " world", LogProbability: -0.25},
		},
		TopCandidates: []*LogprobsResultTopCandidates{
			{Candidates: []*LogprobsResultCandidate{{Token: "Hello", LogProbability: -0.5}, {Token: "Hi", LogProbability: -1}}},
			{Candidates: []*LogprobsResultCandidate{{Token: " world", LogProbability: -0.25}}},
		},
	}
}

func assertHelloWorldLogProbs(t *testing.T, logProbs []schemas.ResponsesOutputMessageContentTextLogProb, msg string) {
	t.Helper()
	require.Len(t, logProbs, 2, msg)
	assert.Equal(t, "Hello", logProbs[0].Token, msg)
	assert.InDelta(t, -0.5, logProbs[0].LogProb, 1e-9, msg)
	assert.Equal(t, []int{'H', 'e', 'l', 'l', 'o'}, logProbs[0].Bytes, msg)
	require.Len(t, logProbs[0].TopLogProbs, 2, msg)
	assert.Equal(t, "Hi", logProbs[0].TopLogProbs[1].Token, msg)
	assert.InDelta(t, -1, logProbs[0].TopLogProbs[1].LogProb, 1e-9, msg)
	assert.Equal(t, " world", logProbs[1].Token, msg)
	assert.InDelta(t, -0.25, logProbs[1].LogProb, 1e-9, msg)
}

// A unary candidate's logprobsResult reaches the logprobs of its output_text part.
func TestGeminiResponsesResponse_LogprobsReachOutputText(t *testing.T) {
	response := &GenerateContentResponse{
		Candidates: []*Candidate{{
			Content:        &Content{Role: "model", Parts: []*Part{{Text: "Hello world"}}},
			FinishReason:   FinishReasonStop,
			LogprobsResult: logprobsResult(),
		}},
	}
	bifrostResp := response.ToResponsesBifrostResponsesResponse()
	require.NotNil(t, bifrostResp)
	require.NotEmpty(t, bifrostResp.Output)
	block := textBlockOf(bifrostResp.Output, 0)
	require.NotNil(t, block)
	assertHelloWorldLogProbs(t, block.LogProbs, "unary output_text")

	// Without logprobsResult the part keeps an empty list, not null.
	response.Candidates[0].LogprobsResult = nil
	bifrostResp = response.ToResponsesBifrostResponsesResponse()
	block = textBlockOf(bifrostResp.Output, 0)
	require.NotNil(t, block)
	assert.NotNil(t, block.LogProbs)
	assert.Empty(t, block.LogProbs)
}

// Each streamed chunk's logprobsResult rides that chunk's output_text.delta, and the
// text item's done events and response.completed carry the whole item's logprobs.
func TestGeminiResponsesStream_LogprobsReachDeltasAndDone(t *testing.T) {
	full := logprobsResult()
	chunks := []*GenerateContentResponse{
		{
			ResponseID: "resp_1",
			Candidates: []*Candidate{{
				Content: &Content{Role: "model", Parts: []*Part{{Text: "Hello"}}},
				LogprobsResult: &LogprobsResult{
					ChosenCandidates: full.ChosenCandidates[:1],
					TopCandidates:    full.TopCandidates[:1],
				},
			}},
		},
		{
			Candidates: []*Candidate{{
				Content: &Content{Role: "model", Parts: []*Part{{Text: " world"}}},
				LogprobsResult: &LogprobsResult{
					ChosenCandidates: full.ChosenCandidates[1:],
					TopCandidates:    full.TopCandidates[1:],
				},
				FinishReason: FinishReasonStop,
			}},
		},
	}

	state := &GeminiResponsesStreamState{}
	state.flush()
	var all []*schemas.BifrostResponsesStreamResponse
	seq := 0
	for _, chunk := range chunks {
		events, bifrostErr := chunk.ToBifrostResponsesStream(seq, state)
		require.Nil(t, bifrostErr)
		all = append(all, events...)
		seq += len(events)
	}

	var deltas []*schemas.BifrostResponsesStreamResponse
	var textDone, partDone, itemDone, completed *schemas.BifrostResponsesStreamResponse
	for _, event := range all {
		switch event.Type {
		case schemas.ResponsesStreamResponseTypeOutputTextDelta:
			deltas = append(deltas, event)
		case schemas.ResponsesStreamResponseTypeOutputTextDone:
			textDone = event
		case schemas.ResponsesStreamResponseTypeContentPartDone:
			partDone = event
		case schemas.ResponsesStreamResponseTypeOutputItemDone:
			itemDone = event
		case schemas.ResponsesStreamResponseTypeCompleted:
			completed = event
		}
	}
	require.Len(t, deltas, 2)
	require.Len(t, deltas[0].LogProbs, 1)
	assert.Equal(t, "Hello", deltas[0].LogProbs[0].Token)
	require.Len(t, deltas[1].LogProbs, 1)
	assert.Equal(t, " world", deltas[1].LogProbs[0].Token)

	require.NotNil(t, textDone)
	assertHelloWorldLogProbs(t, textDone.LogProbs, "output_text.done")
	require.NotNil(t, partDone)
	require.NotNil(t, partDone.Part.ResponsesOutputMessageContentText)
	assertHelloWorldLogProbs(t, partDone.Part.LogProbs, "content_part.done")
	require.NotNil(t, itemDone)
	require.NotNil(t, itemDone.Item.Content)
	assertHelloWorldLogProbs(t, itemDone.Item.Content.ContentBlocks[0].LogProbs, "output_item.done")
	require.NotNil(t, completed)
	require.NotEmpty(t, completed.Response.Output)
	block := textBlockOf(completed.Response.Output, 0)
	require.NotNil(t, block)
	assertHelloWorldLogProbs(t, block.LogProbs, "response.completed")

	// A pooled state does not leak one stream's logprobs into the next.
	state.flush()
	assert.Nil(t, state.TextLogProbs)
	assert.Nil(t, state.PendingLogProbs)
}

func mustMarshal(t *testing.T, value any) []byte {
	t.Helper()
	encoded, err := json.Marshal(value)
	require.NoError(t, err)
	return encoded
}
