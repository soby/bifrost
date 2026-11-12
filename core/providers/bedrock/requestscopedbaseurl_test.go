package bedrock

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/maximhq/bifrost/core/schemas"
	"github.com/stretchr/testify/require"
)

// requestScopedTestConfig mirrors the configuration Bifrost builds request-scoped instances
// with (loopback is private, reads are context-bound), with allowPrivateNetwork as given.
func requestScopedTestConfig(allowPrivateNetwork bool) *schemas.ProviderConfig {
	networkConfig := schemas.DefaultNetworkConfig
	networkConfig.AllowPrivateNetwork = allowPrivateNetwork
	networkConfig.LoopbackIsPrivate = true
	networkConfig.ContextBoundReads = true
	return &schemas.ProviderConfig{
		NetworkConfig:            networkConfig,
		ConcurrencyAndBufferSize: schemas.ConcurrencyAndBufferSize{Concurrency: 1, BufferSize: 1},
	}
}

func bearerTestKey() schemas.Key {
	return schemas.Key{
		Value:            *schemas.NewSecretVar("bedrock-api-key"),
		BedrockKeyConfig: &schemas.BedrockKeyConfig{Region: schemas.NewSecretVar("us-east-1")},
	}
}

func staticTestKey() schemas.Key {
	return schemas.Key{BedrockKeyConfig: &schemas.BedrockKeyConfig{
		Region:    schemas.NewSecretVar("us-east-1"),
		AccessKey: *schemas.NewSecretVar("AKIAEXAMPLE"),
		SecretKey: *schemas.NewSecretVar("secret"),
	}}
}

// recordedRequest is what the test origin saw of one request.
type recordedRequest struct {
	method        string
	requestURI    string
	authorization string
	accept        string
	host          string
}

// newRuntimeOrigin starts a plain-HTTP loopback server standing in for a runtime origin
// (a VPC endpoint or signing proxy) that answers every request with status and body, and
// records what it receives.
func newRuntimeOrigin(t *testing.T, status int, contentType, body string) (*httptest.Server, func() []recordedRequest) {
	t.Helper()
	var mu sync.Mutex
	var seen []recordedRequest
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		mu.Lock()
		seen = append(seen, recordedRequest{
			method:        r.Method,
			requestURI:    r.RequestURI,
			authorization: r.Header.Get("Authorization"),
			accept:        r.Header.Get("Accept"),
			host:          r.Host,
		})
		mu.Unlock()
		w.Header().Set("Content-Type", contentType)
		w.WriteHeader(status)
		_, _ = io.WriteString(w, body)
	}))
	t.Cleanup(server.Close)
	return server, func() []recordedRequest {
		mu.Lock()
		defer mu.Unlock()
		return append([]recordedRequest(nil), seen...)
	}
}

// scopedProvider builds a request-scoped instance and runs ForRequest with baseURL.
func scopedProvider(t *testing.T, allowPrivateNetwork bool, requestType schemas.RequestType, key schemas.Key, baseURL string) *BedrockProvider {
	t.Helper()
	base, err := NewBedrockProvider(requestScopedTestConfig(allowPrivateNetwork), noopLogger{})
	require.NoError(t, err)
	scoped, err := base.ForRequest(requestType, key, baseURL)
	require.NoError(t, err)
	return scoped.(*BedrockProvider)
}

func TestForRequestBaseURLOnlyForModelInvocation(t *testing.T) {
	base, err := NewBedrockProvider(requestScopedTestConfig(false), noopLogger{})
	require.NoError(t, err)
	const baseURL = "http://127.0.0.1:9000/bedrock"

	for _, requestType := range []schemas.RequestType{
		schemas.TextCompletionRequest, schemas.TextCompletionStreamRequest,
		schemas.ChatCompletionRequest, schemas.ChatCompletionStreamRequest,
		schemas.ResponsesRequest, schemas.ResponsesStreamRequest,
		schemas.EmbeddingRequest, schemas.ImageGenerationRequest, schemas.ImageEditRequest, schemas.ImageVariationRequest,
	} {
		t.Run(string(requestType), func(t *testing.T) {
			scoped, err := base.ForRequest(requestType, bearerTestKey(), baseURL)
			require.NoError(t, err)
			require.NotSame(t, base, scoped, "the base URL goes on a copy, never on the shared instance")
			require.Equal(t, baseURL, scoped.(*BedrockProvider).runtimeBaseURL)
			require.Empty(t, base.runtimeBaseURL)
			require.Same(t, base.client, scoped.(*BedrockProvider).client, "the copy shares the instance's clients")
		})
	}

	for _, requestType := range []schemas.RequestType{
		schemas.ListModelsRequest, schemas.CountTokensRequest, schemas.RerankRequest,
		schemas.FileUploadRequest, schemas.FileListRequest, schemas.FileContentRequest,
		schemas.BatchCreateRequest, schemas.BatchRetrieveRequest, schemas.BatchResultsRequest,
		schemas.PassthroughRequest, schemas.PassthroughStreamRequest,
	} {
		t.Run("refused "+string(requestType), func(t *testing.T) {
			_, err := base.ForRequest(requestType, staticTestKey(), baseURL)
			require.ErrorContains(t, err, "only for model invocation")
		})
	}

	t.Run("no base URL keeps the instance", func(t *testing.T) {
		scoped, err := base.ForRequest(schemas.ChatCompletionRequest, bearerTestKey(), "")
		require.NoError(t, err)
		require.Same(t, base, scoped)
	})

	t.Run("credential rules unchanged", func(t *testing.T) {
		noCredentials := schemas.Key{BedrockKeyConfig: &schemas.BedrockKeyConfig{Region: schemas.NewSecretVar("us-east-1")}}
		_, err := base.ForRequest(schemas.ChatCompletionRequest, noCredentials, baseURL)
		require.ErrorContains(t, err, "API key")

		role := bearerTestKey()
		role.BedrockKeyConfig.RoleARN = schemas.NewSecretVar("arn:aws:iam::1:role/r")
		_, err = base.ForRequest(schemas.ChatCompletionRequest, role, baseURL)
		require.ErrorContains(t, err, "role assumption")

		endpoints := bearerTestKey()
		endpoints.BedrockKeyConfig.Endpoints = &schemas.BedrockEndpoints{Runtime: schemas.NewSecretVar("vpce.example.com")}
		_, err = base.ForRequest(schemas.ChatCompletionRequest, endpoints, baseURL)
		require.ErrorContains(t, err, "endpoint overrides")

		noRegion := bearerTestKey()
		noRegion.BedrockKeyConfig.Region = nil
		_, err = base.ForRequest(schemas.ChatCompletionRequest, noRegion, baseURL)
		require.ErrorContains(t, err, "region")
	})
}

func TestBaseURLConverseGoesToTheOrigin(t *testing.T) {
	const converseBody = `{"output":{"message":{"role":"assistant","content":[{"text":"hi"}]}},"stopReason":"end_turn","usage":{"inputTokens":1,"outputTokens":1,"totalTokens":2}}`
	server, seen := newRuntimeOrigin(t, http.StatusOK, "application/json", converseBody)
	provider := scopedProvider(t, true, schemas.ChatCompletionRequest, bearerTestKey(), server.URL+"/bedrock")

	ctx := bedrockTestContext(t, time.Now().Add(10*time.Second))
	response, bifrostErr := provider.ChatCompletion(ctx, bearerTestKey(), &schemas.BifrostChatRequest{
		Provider: schemas.Bedrock,
		Model:    "anthropic.claude-sonnet-4-20250514-v1:0",
		Input:    []schemas.ChatMessage{{Role: schemas.ChatMessageRoleUser, Content: &schemas.ChatMessageContent{ContentStr: schemas.Ptr("hello")}}},
	})
	require.Nil(t, bifrostErr, "%v", bifrostErr)
	require.NotNil(t, response)

	requests := seen()
	require.Len(t, requests, 1)
	require.Equal(t, http.MethodPost, requests[0].method)
	require.Equal(t, "/bedrock/model/anthropic.claude-sonnet-4-20250514-v1:0/converse", requests[0].requestURI)
	require.Equal(t, "Bearer bedrock-api-key", requests[0].authorization)
}

func TestBaseURLSignsTheURLSent(t *testing.T) {
	server, seen := newRuntimeOrigin(t, http.StatusOK, "application/json", `{}`)
	provider := scopedProvider(t, true, schemas.ChatCompletionRequest, staticTestKey(), server.URL)

	ctx := bedrockTestContext(t, time.Now().Add(10*time.Second))
	// An application inference profile's model identifier is percent-encoded into one path
	// segment; the origin must receive it exactly as encoded.
	_, _, _, bifrostErr := provider.completeRequest(ctx, []byte(`{}`), "arn%3Aaws%3Abedrock%3Aus-east-1%3A1%3Aapplication-inference-profile%2Fabc/converse", staticTestKey(), "m")
	require.Nil(t, bifrostErr, "%v", bifrostErr)

	requests := seen()
	require.Len(t, requests, 1)
	require.Equal(t, "/model/arn%3Aaws%3Abedrock%3Aus-east-1%3A1%3Aapplication-inference-profile%2Fabc/converse", requests[0].requestURI)
	require.True(t, strings.HasPrefix(requests[0].authorization, "AWS4-HMAC-SHA256 Credential=AKIAEXAMPLE/"), requests[0].authorization)
	require.Contains(t, requests[0].authorization, "/us-east-1/bedrock/aws4_request")
	require.Equal(t, strings.TrimPrefix(server.URL, "http://"), requests[0].host)
}

func TestBaseURLConverseStreamGoesToTheOrigin(t *testing.T) {
	server, seen := newRuntimeOrigin(t, http.StatusOK, "application/vnd.amazon.eventstream", "")
	provider := scopedProvider(t, true, schemas.ChatCompletionStreamRequest, bearerTestKey(), server.URL)

	ctx := bedrockTestContext(t, time.Now().Add(10*time.Second))
	resp, bifrostErr := provider.makeStreamingRequest(ctx, []byte(`{}`), bearerTestKey(), "anthropic.claude-sonnet-4-20250514-v1:0", "converse-stream")
	require.Nil(t, bifrostErr, "%v", bifrostErr)
	_ = resp.Body.Close()

	requests := seen()
	require.Len(t, requests, 1)
	require.Equal(t, "/model/anthropic.claude-sonnet-4-20250514-v1:0/converse-stream", requests[0].requestURI)
	require.Equal(t, "application/vnd.amazon.eventstream", requests[0].accept)
}

func TestBaseURLInvokeModelGoesToTheOrigin(t *testing.T) {
	const messageBody = `{"id":"msg_1","type":"message","role":"assistant","model":"claude","content":[{"type":"text","text":"hi"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`
	server, seen := newRuntimeOrigin(t, http.StatusOK, "application/json", messageBody)
	provider := scopedProvider(t, true, schemas.ChatCompletionRequest, bearerTestKey(), server.URL+"/bedrock")

	requestURL, _ := provider.invokeURL(bedrockTestContext(t, time.Now().Add(time.Second)), bearerTestKey(), "anthropic.claude-sonnet-4-20250514-v1:0", bedrockInvokeStreamAction)
	require.Equal(t, server.URL+"/bedrock/model/anthropic.claude-sonnet-4-20250514-v1:0/"+bedrockInvokeStreamAction, requestURL)

	ctx := bedrockTestContext(t, time.Now().Add(10*time.Second))
	response, bifrostErr := provider.invokeAnthropicChatCompletion(ctx, bearerTestKey(), &schemas.BifrostChatRequest{
		Provider: schemas.Bedrock,
		Model:    "anthropic.claude-sonnet-4-20250514-v1:0",
		Input:    []schemas.ChatMessage{{Role: schemas.ChatMessageRoleUser, Content: &schemas.ChatMessageContent{ContentStr: schemas.Ptr("hello")}}},
	})
	require.Nil(t, bifrostErr, "%v", bifrostErr)
	require.NotNil(t, response)

	requests := seen()
	require.Len(t, requests, 1)
	require.Equal(t, "/bedrock/model/anthropic.claude-sonnet-4-20250514-v1:0/"+bedrockInvokeAction, requests[0].requestURI)
	require.Equal(t, "Bearer bedrock-api-key", requests[0].authorization)
}

func TestBaseURLPrivateNetworkPolicy(t *testing.T) {
	server, seen := newRuntimeOrigin(t, http.StatusOK, "application/json", `{}`)
	provider := scopedProvider(t, false, schemas.ChatCompletionRequest, bearerTestKey(), server.URL)
	ctx := bedrockTestContext(t, time.Now().Add(10*time.Second))

	_, _, _, bifrostErr := provider.completeRequest(ctx, []byte(`{}`), "m/converse", bearerTestKey(), "m")
	require.NotNil(t, bifrostErr, "a loopback origin needs the private-network opt-in")
	require.Contains(t, bifrostErr.Error.Error.Error(), "private IP")

	_, bifrostErr = provider.makeStreamingRequest(ctx, []byte(`{}`), bearerTestKey(), "m", "converse-stream")
	require.NotNil(t, bifrostErr, "a loopback origin needs the private-network opt-in")

	_, bifrostErr = provider.invokeAnthropicChatCompletion(ctx, bearerTestKey(), &schemas.BifrostChatRequest{
		Provider: schemas.Bedrock,
		Model:    "anthropic.claude-sonnet-4-20250514-v1:0",
		Input:    []schemas.ChatMessage{{Role: schemas.ChatMessageRoleUser, Content: &schemas.ChatMessageContent{ContentStr: schemas.Ptr("hello")}}},
	})
	require.NotNil(t, bifrostErr, "a loopback origin needs the private-network opt-in")
	require.Empty(t, seen())
}

func TestBaseURLRefusesOtherSurfaces(t *testing.T) {
	server, seen := newRuntimeOrigin(t, http.StatusOK, "application/json", `{}`)
	ctx := bedrockTestContext(t, time.Now().Add(10*time.Second))
	input := []schemas.ChatMessage{{Role: schemas.ChatMessageRoleUser, Content: &schemas.ChatMessageContent{ContentStr: schemas.Ptr("hello")}}}
	responsesInput := []schemas.ResponsesMessage{{Role: schemas.Ptr(schemas.ResponsesInputMessageRoleUser), Content: &schemas.ResponsesMessageContent{ContentStr: schemas.Ptr("hello")}}}

	// gpt-oss is served by Mantle only.
	const mantleModel = "openai.gpt-oss-120b-1:0"
	provider := scopedProvider(t, true, schemas.ChatCompletionRequest, bearerTestKey(), server.URL)
	_, bifrostErr := provider.ChatCompletion(ctx, bearerTestKey(), &schemas.BifrostChatRequest{Provider: schemas.Bedrock, Model: mantleModel, Input: input})
	requireBaseURLSurfaceRefused(t, bifrostErr)
	_, bifrostErr = provider.ChatCompletionStream(ctx, nil, nil, bearerTestKey(), &schemas.BifrostChatRequest{Provider: schemas.Bedrock, Model: mantleModel, Input: input})
	requireBaseURLSurfaceRefused(t, bifrostErr)
	_, bifrostErr = provider.Responses(ctx, bearerTestKey(), &schemas.BifrostResponsesRequest{Provider: schemas.Bedrock, Model: mantleModel, Input: responsesInput})
	requireBaseURLSurfaceRefused(t, bifrostErr)
	_, bifrostErr = provider.ResponsesStream(ctx, nil, nil, bearerTestKey(), &schemas.BifrostResponsesRequest{Provider: schemas.Bedrock, Model: mantleModel, Input: responsesInput})
	requireBaseURLSurfaceRefused(t, bifrostErr)

	// A key that opts into the runtime's OpenAI-compatible surface for an OpenAI-family model.
	openAIKey := bearerTestKey()
	openAIKey.UseOpenAIEndpoints = schemas.Ptr(true)
	provider = scopedProvider(t, true, schemas.ResponsesRequest, openAIKey, server.URL)
	_, bifrostErr = provider.Responses(ctx, openAIKey, &schemas.BifrostResponsesRequest{Provider: schemas.Bedrock, Model: "us.openai.gpt-5.6-luna", Input: responsesInput})
	requireBaseURLSurfaceRefused(t, bifrostErr)

	require.Empty(t, seen(), "a refused attempt sends nothing")
}

func requireBaseURLSurfaceRefused(t *testing.T, bifrostErr *schemas.BifrostError) {
	t.Helper()
	require.NotNil(t, bifrostErr)
	require.NotNil(t, bifrostErr.StatusCode)
	require.Equal(t, http.StatusBadRequest, *bifrostErr.StatusCode)
	require.Contains(t, bifrostErr.Error.Message, "a base URL is limited to")
}
