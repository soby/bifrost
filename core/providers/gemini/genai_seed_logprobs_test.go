package gemini

import (
	"context"
	"encoding/json"
	"math"
	"testing"

	providerUtils "github.com/maximhq/bifrost/core/providers/utils"
	"github.com/maximhq/bifrost/core/schemas"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func genaiRequest(t *testing.T, body string) *GeminiGenerationRequest {
	t.Helper()
	var request GeminiGenerationRequest
	require.NoError(t, json.Unmarshal([]byte(body), &request))
	request.Model = "gemini/gemini-2.5-flash"
	return &request
}

// A GenAI request's responseLogprobs becomes the Responses include, which every
// provider conversion reads, instead of an extra param only passthrough would send;
// logprobs stays top_logprobs and seed rides extra_params.seed (Responses has no seed).
func TestGenAIIngress_LogprobsAndSeedMapped(t *testing.T) {
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	request := genaiRequest(t, `{"contents":[{"role":"user","parts":[{"text":"hi"}]}],"generationConfig":{"responseLogprobs":true,"logprobs":3,"seed":1234}}`)
	bifrostReq := request.ToBifrostResponsesRequest(ctx)
	require.NotNil(t, bifrostReq.Params)
	assert.Equal(t, []string{schemas.ResponsesIncludeOutputTextLogprobs}, bifrostReq.Params.Include)
	require.NotNil(t, bifrostReq.Params.TopLogProbs)
	assert.Equal(t, 3, *bifrostReq.Params.TopLogProbs)
	assert.NotContains(t, bifrostReq.Params.ExtraParams, "response_logprobs")
	assert.Equal(t, 1234, bifrostReq.Params.ExtraParams["seed"])

	// Served by Gemini: all three reach generationConfig, and seed is not also sent
	// as an unknown top-level field.
	geminiReq, err := ToGeminiResponsesRequest(ctx, bifrostReq)
	require.NoError(t, err)
	config := responsesGenerationConfigOnWire(t, geminiReq)
	assert.JSONEq(t, `true`, string(config["responseLogprobs"]))
	assert.JSONEq(t, `3`, string(config["logprobs"]))
	assert.JSONEq(t, `1234`, string(config["seed"]))
	assert.NotContains(t, geminiReq.ExtraParams, "seed")
	assert.Contains(t, bifrostReq.Params.ExtraParams, "seed", "the request keeps its extra params for a retry or fallback")

	// Served by a chat provider through the chat fallback: typed logprobs and seed.
	chatReq := bifrostReq.ToChatRequest()
	require.NotNil(t, chatReq.Params.LogProbs)
	assert.True(t, *chatReq.Params.LogProbs)
	require.NotNil(t, chatReq.Params.Seed)
	assert.Equal(t, 1234, *chatReq.Params.Seed)
	assert.NotContains(t, chatReq.Params.ExtraParams, "seed")

	// Absent fields stay absent.
	bifrostReq = genaiRequest(t, `{"contents":[{"role":"user","parts":[{"text":"hi"}]}],"generationConfig":{"temperature":0.1}}`).ToBifrostResponsesRequest(ctx)
	assert.Empty(t, bifrostReq.Params.Include)
	assert.NotContains(t, bifrostReq.Params.ExtraParams, "seed")
}

// extra_params.seed outside generationConfig.seed's int32 range is a caller error on
// the Gemini Responses path, never truncated.
func TestToGeminiResponsesRequest_ExtraParamsSeedOutsideInt32Refused(t *testing.T) {
	_, err := ToGeminiResponsesRequest(nil, logprobsResponsesRequest(schemas.Gemini, &schemas.ResponsesParameters{
		ExtraParams: map[string]interface{}{"seed": math.MaxInt32 + 1},
	}))
	badRequest, ok := providerUtils.AsBifrostBadRequestError(err)
	require.True(t, ok, "want a caller error, got %v", err)
	assert.Contains(t, badRequest.Error.Message, "seed")
}
