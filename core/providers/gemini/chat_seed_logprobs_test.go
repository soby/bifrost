package gemini_test

import (
	"encoding/json"
	"math"
	"testing"

	"github.com/maximhq/bifrost/core/providers/gemini"
	providerUtils "github.com/maximhq/bifrost/core/providers/utils"
	"github.com/maximhq/bifrost/core/schemas"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func seedLogprobsChatRequest(provider schemas.ModelProvider, params *schemas.ChatParameters) *schemas.BifrostChatRequest {
	return &schemas.BifrostChatRequest{
		Provider: provider,
		Model:    "gemini-2.5-flash",
		Input: []schemas.ChatMessage{{
			Role:    schemas.ChatMessageRoleUser,
			Content: &schemas.ChatMessageContent{ContentStr: schemas.Ptr("Hello")},
		}},
		Params: params,
	}
}

// generationConfigOnWire marshals the converted request the way the provider sends
// it and returns its generationConfig object.
func generationConfigOnWire(t *testing.T, request *gemini.GeminiGenerationRequest) map[string]json.RawMessage {
	t.Helper()
	body, err := providerUtils.MarshalProviderRequest(request)
	require.NoError(t, err)
	var wire struct {
		GenerationConfig map[string]json.RawMessage `json:"generationConfig"`
	}
	require.NoError(t, json.Unmarshal(body, &wire), "%s", body)
	return wire.GenerationConfig
}

// The chat seed reaches generationConfig.seed on Gemini and on Vertex, which shares
// the Gemini chat conversion.
func TestToGeminiChatCompletionRequest_SeedReachesGenerationConfig(t *testing.T) {
	for _, provider := range []schemas.ModelProvider{schemas.Gemini, schemas.Vertex} {
		for _, seed := range []int{0, 1234567, math.MaxInt32, math.MinInt32} {
			request, err := gemini.ToGeminiChatCompletionRequest(nil, seedLogprobsChatRequest(provider, &schemas.ChatParameters{Seed: schemas.Ptr(seed)}))
			require.NoError(t, err, "%s seed %d", provider, seed)
			config := generationConfigOnWire(t, request)
			assert.JSONEq(t, string(mustJSON(t, seed)), string(config["seed"]), "%s seed %d", provider, seed)
		}
		request, err := gemini.ToGeminiChatCompletionRequest(nil, seedLogprobsChatRequest(provider, &schemas.ChatParameters{}))
		require.NoError(t, err)
		assert.NotContains(t, generationConfigOnWire(t, request), "seed", "%s: an absent seed must stay absent", provider)
	}
}

// A seed outside generationConfig.seed's int32 range is refused as a caller error
// (400) naming the field and range; it is never truncated.
func TestToGeminiChatCompletionRequest_SeedOutsideInt32IsRefused(t *testing.T) {
	for _, provider := range []schemas.ModelProvider{schemas.Gemini, schemas.Vertex} {
		for _, seed := range []int{math.MaxInt32 + 1, math.MinInt32 - 1, 9007199254740993} {
			_, err := gemini.ToGeminiChatCompletionRequest(nil, seedLogprobsChatRequest(provider, &schemas.ChatParameters{Seed: schemas.Ptr(seed)}))
			require.Error(t, err, "%s seed %d", provider, seed)
			badRequest, ok := providerUtils.AsBifrostBadRequestError(err)
			require.True(t, ok, "%s seed %d: want a caller error, got %v", provider, seed, err)
			require.NotNil(t, badRequest.StatusCode)
			assert.Equal(t, 400, *badRequest.StatusCode)
			assert.Contains(t, badRequest.Error.Message, "seed")
			assert.Contains(t, badRequest.Error.Message, "-2147483648")
			assert.Contains(t, badRequest.Error.Message, "2147483647")
		}
	}
}

// logprobs and top_logprobs map to responseLogprobs and logprobs with the caller's
// count, including counts Gemini itself rejects: Gemini answers those, not Bifrost.
func TestToGeminiChatCompletionRequest_TopLogprobsSentAsGiven(t *testing.T) {
	for _, provider := range []schemas.ModelProvider{schemas.Gemini, schemas.Vertex} {
		for _, count := range []int{1, 5, 20, 21, 100, -1} {
			request, err := gemini.ToGeminiChatCompletionRequest(nil, seedLogprobsChatRequest(provider, &schemas.ChatParameters{
				LogProbs:    schemas.Ptr(true),
				TopLogProbs: schemas.Ptr(count),
			}))
			require.NoError(t, err, "%s top_logprobs %d", provider, count)
			config := generationConfigOnWire(t, request)
			assert.JSONEq(t, `true`, string(config["responseLogprobs"]), "%s top_logprobs %d", provider, count)
			assert.JSONEq(t, string(mustJSON(t, count)), string(config["logprobs"]), "%s top_logprobs %d", provider, count)
		}

		request, err := gemini.ToGeminiChatCompletionRequest(nil, seedLogprobsChatRequest(provider, &schemas.ChatParameters{LogProbs: schemas.Ptr(true)}))
		require.NoError(t, err)
		config := generationConfigOnWire(t, request)
		assert.JSONEq(t, `true`, string(config["responseLogprobs"]), "%s logprobs only", provider)
		assert.NotContains(t, config, "logprobs", "%s: no top_logprobs, no logprobs count", provider)

		_, err = gemini.ToGeminiChatCompletionRequest(nil, seedLogprobsChatRequest(provider, &schemas.ChatParameters{TopLogProbs: schemas.Ptr(math.MaxInt32 + 1)}))
		badRequest, ok := providerUtils.AsBifrostBadRequestError(err)
		require.True(t, ok, "%s: a top_logprobs outside int32 must be a caller error, got %v", provider, err)
		assert.Contains(t, badRequest.Error.Message, "top_logprobs")
	}
}

func mustJSON(t *testing.T, value any) []byte {
	t.Helper()
	encoded, err := json.Marshal(value)
	require.NoError(t, err)
	return encoded
}
