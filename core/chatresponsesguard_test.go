package bifrost

import (
	"context"
	"fmt"
	"testing"

	"github.com/maximhq/bifrost/core/schemas"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// dispatchConvertedChat runs a chat request that a plugin marked for conversion to the
// Responses API (as compat's chat-to-responses conversion does) through core dispatch.
func dispatchConvertedChat(t *testing.T, params *schemas.ChatParameters, stream, rawBody bool) (*billingHeaderCaptureProvider, *schemas.BifrostError) {
	t.Helper()
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	ctx.SetValue(schemas.BifrostContextKeyChangeRequestType, schemas.ResponsesRequest)
	req := &schemas.BifrostChatRequest{
		Provider: schemas.OpenAI,
		Model:    "o1-pro",
		Input:    []schemas.ChatMessage{{Role: schemas.ChatMessageRoleUser, Content: &schemas.ChatMessageContent{ContentStr: new("Reply with OK.")}}},
		Params:   params,
	}
	if rawBody {
		ctx.SetValue(schemas.BifrostContextKeyUseRawRequestBody, true)
		req.RawRequestBody = []byte(`{"model":"o1-pro","seed":7,"messages":[{"role":"user","content":"Reply with OK."}]}`)
	}
	message := &ChannelMessage{Context: ctx, BifrostRequest: schemas.BifrostRequest{RequestType: schemas.ChatCompletionRequest, ChatRequest: req}}
	provider := &billingHeaderCaptureProvider{stubProvider: stubProvider{key: schemas.OpenAI}}
	client := &Bifrost{}
	var bifrostErr *schemas.BifrostError
	if stream {
		message.RequestType = schemas.ChatCompletionStreamRequest
		_, bifrostErr = client.handleProviderStreamRequest(provider, nil, message, schemas.Key{}, nil, nil)
	} else {
		_, bifrostErr = client.handleProviderRequest(provider, nil, message, schemas.Key{}, nil)
	}
	return provider, bifrostErr
}

// A chat request converted to the Responses API by a plugin is refused, naming the
// parameters, when it sets chat parameters the Responses API cannot carry: the
// conversion would drop them silently.
func TestConvertedChatToResponses_RefusesParamsWithoutEquivalent(t *testing.T) {
	cases := []struct {
		name   string
		want   []string
		params schemas.ChatParameters
	}{
		{name: "seed", want: []string{"seed"}, params: schemas.ChatParameters{Seed: new(7)}},
		{name: "stop", want: []string{"stop"}, params: schemas.ChatParameters{Stop: []string{"END"}}},
		{name: "n", want: []string{"n > 1"}, params: schemas.ChatParameters{N: new(2)}},
		{name: "presence_penalty", want: []string{"presence_penalty"}, params: schemas.ChatParameters{PresencePenalty: new(0.5)}},
		{name: "several", want: []string{"seed", "logit_bias"}, params: schemas.ChatParameters{Seed: new(1), LogitBias: &map[string]float64{"50256": -100}}},
	}
	for _, tc := range cases {
		for _, stream := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/stream=%t", tc.name, stream), func(t *testing.T) {
				params := tc.params
				provider, bifrostErr := dispatchConvertedChat(t, &params, stream, false)
				require.NotNil(t, bifrostErr, "%s must be refused, not dropped by the conversion", tc.name)
				require.NotNil(t, bifrostErr.StatusCode)
				assert.Equal(t, 400, *bifrostErr.StatusCode)
				for _, field := range tc.want {
					assert.Contains(t, bifrostErr.Error.Message, field)
				}
				assert.Nil(t, provider.responses)
				assert.Nil(t, provider.chat)
			})
		}
	}
}

// logprobs and top_logprobs have Responses equivalents, so a converted request that
// sets them is served, with include "message.output_text.logprobs" and top_logprobs.
func TestConvertedChatToResponses_CarriesLogprobs(t *testing.T) {
	for _, stream := range []bool{false, true} {
		t.Run(fmt.Sprintf("stream=%t", stream), func(t *testing.T) {
			provider, bifrostErr := dispatchConvertedChat(t, &schemas.ChatParameters{LogProbs: new(true), TopLogProbs: new(3)}, stream, false)
			require.Nil(t, bifrostErr)
			require.NotNil(t, provider.responses)
			require.NotNil(t, provider.responses.Params)
			assert.Equal(t, []string{schemas.ResponsesIncludeOutputTextLogprobs}, provider.responses.Params.Include)
			require.NotNil(t, provider.responses.Params.TopLogProbs)
			assert.Equal(t, 3, *provider.responses.Params.TopLogProbs)
		})
	}
}

// A raw-body request is sent as is, so the guard leaves it alone.
func TestConvertedChatToResponses_RawBodyNotRefused(t *testing.T) {
	for _, stream := range []bool{false, true} {
		t.Run(fmt.Sprintf("stream=%t", stream), func(t *testing.T) {
			_, bifrostErr := dispatchConvertedChat(t, &schemas.ChatParameters{Seed: new(7)}, stream, true)
			require.Nil(t, bifrostErr)
		})
	}
}
