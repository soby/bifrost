package bifrost

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/maximhq/bifrost/core/internal/memtest"
	schemas "github.com/maximhq/bifrost/core/schemas"
)

// TestRequestScopedProviderInstances builds every schemas.RequestScopedProvider implementer the
// way request-scoped attempts do and pins that ForRequest returns the receiver without
// allocating when there is no base URL, copies it for a base URL, and that the instance reaches
// a loopback endpoint only when private networks are allowed.
func TestRequestScopedProviderInstances(t *testing.T) {
	var hits atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte(`{"error":{"message":"test"}}`))
	}))
	defer server.Close()

	baseURLKey := func(string) schemas.Key { return schemas.Key{ID: "k", Value: *schemas.NewSecretVar("sk-test")} }
	tests := []struct {
		provider schemas.ModelProvider
		model    string
		key      func(url string) schemas.Key
		baseURL  func(url string) string // empty when the endpoint is on the key
	}{
		{provider: schemas.OpenAI, key: baseURLKey},
		{provider: schemas.Anthropic, key: baseURLKey},
		{provider: schemas.Gemini, key: baseURLKey, baseURL: func(url string) string { return url + "/v1beta" }},
		{provider: schemas.Cohere, key: baseURLKey},
		{provider: schemas.Cerebras, key: baseURLKey},
		{provider: schemas.Groq, key: baseURLKey},
		{provider: schemas.Mistral, key: baseURLKey},
		{provider: schemas.Nebius, key: baseURLKey},
		{provider: schemas.OpenRouter, key: baseURLKey},
		{provider: schemas.Parasail, key: baseURLKey},
		{provider: schemas.Perplexity, key: baseURLKey},
		{provider: schemas.XAI, key: baseURLKey},
		{provider: schemas.Replicate, key: baseURLKey},
		{provider: schemas.HuggingFace, model: "org/model", key: baseURLKey},
		{provider: schemas.Ollama, key: baseURLKey},
		{provider: schemas.SGL, key: baseURLKey},
		{provider: schemas.VLLM, key: func(url string) schemas.Key {
			return schemas.Key{ID: "k", VLLMKeyConfig: &schemas.VLLMKeyConfig{URL: *schemas.NewSecretVar(url)}}
		}},
		{provider: schemas.Azure, key: func(url string) schemas.Key {
			return schemas.Key{ID: "k", Value: *schemas.NewSecretVar("sk-test"), AzureKeyConfig: &schemas.AzureKeyConfig{Endpoint: *schemas.NewSecretVar(url)}}
		}},
	}
	b := &Bifrost{logger: NewDefaultLogger(schemas.LogLevelError)}
	for _, tt := range tests {
		t.Run(string(tt.provider), func(t *testing.T) {
			endpointOnKey := tt.provider == schemas.VLLM || tt.provider == schemas.Azure
			baseURL := ""
			if !endpointOnKey {
				baseURL = server.URL
				if tt.baseURL != nil {
					baseURL = tt.baseURL(server.URL)
				}
			}
			model := tt.model
			if model == "" {
				model = "test-model"
			}
			key := tt.key(server.URL)
			for _, allowPrivateNetwork := range []bool{true, false} {
				instance, err := b.createBaseProvider(tt.provider, requestScopedProviderConfig(allowPrivateNetwork))
				if err != nil {
					t.Fatalf("createBaseProvider: %v", err)
				}
				scopedProvider, ok := instance.(schemas.RequestScopedProvider)
				if !ok {
					t.Fatalf("%T does not implement schemas.RequestScopedProvider", instance)
				}

				same, err := scopedProvider.ForRequest(schemas.ChatCompletionRequest, key, "")
				if err != nil || same != instance {
					t.Fatalf("ForRequest without a base URL = (%p, %v), want the receiver %p", same, err, instance)
				}
				if allocs := testing.AllocsPerRun(100, func() { _, _ = scopedProvider.ForRequest(schemas.ChatCompletionRequest, key, "") }); allocs != 0 {
					t.Errorf("ForRequest without a base URL: %v allocs, want 0", allocs)
				}
				attempt := same
				if !endpointOnKey {
					attempt, err = scopedProvider.ForRequest(schemas.ChatCompletionRequest, key, baseURL)
					if err != nil || attempt == instance {
						t.Fatalf("ForRequest with a base URL = (%p, %v), want a copy of %p", attempt, err, instance)
					}
					if attempt.GetProviderKey() != tt.provider {
						t.Fatalf("copy reports provider %q", attempt.GetProviderKey())
					}
					if allocs := testing.AllocsPerRun(100, func() { _, _ = scopedProvider.ForRequest(schemas.ChatCompletionRequest, key, baseURL) }); allocs != 1 {
						t.Errorf("ForRequest with a base URL: %v allocs, want 1", allocs)
					}
				}

				before := hits.Load()
				ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
				content := "hello"
				_, bifrostErr := attempt.ChatCompletion(ctx, key, &schemas.BifrostChatRequest{
					Provider: tt.provider,
					Model:    model,
					Input:    []schemas.ChatMessage{{Role: schemas.ChatMessageRoleUser, Content: &schemas.ChatMessageContent{ContentStr: &content}}},
					Params:   &schemas.ChatParameters{},
				})
				reached := hits.Load() - before
				if allowPrivateNetwork {
					if reached != 1 {
						t.Fatalf("with private networks allowed the loopback endpoint was reached %d times, want 1 (error: %v)", reached, bifrostErr)
					}
					continue
				}
				if reached != 0 || bifrostErr == nil || bifrostErr.Error == nil || bifrostErr.Error.Error == nil ||
					!strings.Contains(bifrostErr.Error.Error.Error(), "private IP") {
					t.Fatalf("without private networks the loopback endpoint must be refused at dial time: reached %d times, error %v", reached, bifrostErr)
				}
			}
		})
	}
}

// TestRequestScopedProviderRules pins the configurations each implementer refuses because it would
// serve them with ambient credentials, per-credential caches or an endpoint it cannot honor, and
// that providers without the interface cannot be configured per request.
func TestRequestScopedProviderRules(t *testing.T) {
	region := func(r string) *schemas.SecretVar { return schemas.NewSecretVar(r) }
	azureKey := func(mutate func(*schemas.Key)) schemas.Key {
		k := schemas.Key{Value: *schemas.NewSecretVar("az-key"), AzureKeyConfig: &schemas.AzureKeyConfig{Endpoint: *schemas.NewSecretVar("https://x.openai.azure.com")}}
		if mutate != nil {
			mutate(&k)
		}
		return k
	}
	bedrockBearer := func(mutate func(*schemas.BedrockKeyConfig)) schemas.Key {
		cfg := &schemas.BedrockKeyConfig{Region: region("us-east-1")}
		if mutate != nil {
			mutate(cfg)
		}
		return schemas.Key{Value: *schemas.NewSecretVar("bedrock-api-key"), BedrockKeyConfig: cfg}
	}
	vertexToken := func(mutate func(*schemas.Key)) schemas.Key {
		k := schemas.Key{VertexKeyConfig: &schemas.VertexKeyConfig{
			ProjectID: *schemas.NewSecretVar("my-project"), Region: *schemas.NewSecretVar("us-central1"), AccessToken: "ya29.test-token",
		}}
		if mutate != nil {
			mutate(&k)
		}
		return k
	}
	bedrockStatic := schemas.Key{BedrockKeyConfig: &schemas.BedrockKeyConfig{
		AccessKey: *schemas.NewSecretVar("AKIAEXAMPLE"), SecretKey: *schemas.NewSecretVar("secret"), Region: region("eu-west-2"),
	}}

	tests := []struct {
		name        string
		provider    schemas.ModelProvider
		requestType schemas.RequestType
		key         schemas.Key
		baseURL     string
		wantErr     string
	}{
		{name: "openai keyless", provider: schemas.OpenAI, requestType: schemas.EmbeddingRequest},
		{name: "openai batch", provider: schemas.OpenAI, requestType: schemas.BatchCreateRequest, key: schemas.Key{Value: *schemas.NewSecretVar("sk")}},
		{name: "vllm base URL", provider: schemas.VLLM, requestType: schemas.ChatCompletionRequest, baseURL: "https://vllm.example.com", wantErr: "vllm_key_config.url"},
		{name: "azure api key", provider: schemas.Azure, requestType: schemas.ResponsesRequest, key: azureKey(nil)},
		{name: "azure base URL", provider: schemas.Azure, requestType: schemas.ChatCompletionRequest, key: azureKey(nil), baseURL: "https://x.example.com", wantErr: "not a base URL"},
		{name: "azure no endpoint", provider: schemas.Azure, requestType: schemas.ChatCompletionRequest, key: azureKey(func(k *schemas.Key) { k.AzureKeyConfig = nil }), wantErr: "endpoint"},
		{name: "azure no key", provider: schemas.Azure, requestType: schemas.ChatCompletionRequest, key: azureKey(func(k *schemas.Key) { k.Value = schemas.SecretVar{} }), wantErr: "API key"},
		{name: "azure service principal", provider: schemas.Azure, requestType: schemas.ChatCompletionRequest, key: azureKey(func(k *schemas.Key) {
			k.AzureKeyConfig.ClientID = schemas.NewSecretVar("client")
		}), wantErr: "service principal"},
		{name: "azure batch results", provider: schemas.Azure, requestType: schemas.BatchResultsRequest, key: azureKey(nil), wantErr: "ambient storage"},
		{name: "bedrock bearer chat", provider: schemas.Bedrock, requestType: schemas.ChatCompletionRequest, key: bedrockBearer(nil)},
		{name: "bedrock bearer rerank", provider: schemas.Bedrock, requestType: schemas.RerankRequest, key: bedrockBearer(nil)},
		{name: "bedrock static batch", provider: schemas.Bedrock, requestType: schemas.BatchCreateRequest, key: bedrockStatic},
		{name: "bedrock static file", provider: schemas.Bedrock, requestType: schemas.FileUploadRequest, key: bedrockStatic},
		{name: "bedrock bearer batch", provider: schemas.Bedrock, requestType: schemas.BatchCreateRequest, key: bedrockBearer(nil), wantErr: "access key and secret key"},
		{name: "bedrock bearer file", provider: schemas.Bedrock, requestType: schemas.FileListRequest, key: bedrockBearer(nil), wantErr: "access key and secret key"},
		{name: "bedrock no credentials", provider: schemas.Bedrock, requestType: schemas.ChatCompletionRequest, key: schemas.Key{BedrockKeyConfig: &schemas.BedrockKeyConfig{Region: region("us-east-1")}}, wantErr: "API key"},
		{name: "bedrock no key config", provider: schemas.Bedrock, requestType: schemas.ChatCompletionRequest, key: schemas.Key{Value: *schemas.NewSecretVar("k")}, wantErr: "region"},
		{name: "bedrock no region", provider: schemas.Bedrock, requestType: schemas.ChatCompletionRequest, key: bedrockBearer(func(c *schemas.BedrockKeyConfig) { c.Region = nil }), wantErr: "region"},
		{name: "bedrock crafted region", provider: schemas.Bedrock, requestType: schemas.ChatCompletionRequest, key: bedrockBearer(func(c *schemas.BedrockKeyConfig) { c.Region = region("evil.com/#") }), wantErr: "region"},
		{name: "bedrock crafted alias region", provider: schemas.Bedrock, requestType: schemas.ChatCompletionRequest, key: func() schemas.Key {
			k := bedrockBearer(nil)
			k.Aliases = schemas.KeyAliases{"m": {ModelID: "m", Region: region("evil.com/#")}}
			return k
		}(), wantErr: "alias regions"},
		{name: "bedrock role", provider: schemas.Bedrock, requestType: schemas.ChatCompletionRequest, key: bedrockBearer(func(c *schemas.BedrockKeyConfig) { c.RoleARN = schemas.NewSecretVar("arn:aws:iam::1:role/r") }), wantErr: "role assumption"},
		{name: "bedrock endpoints", provider: schemas.Bedrock, requestType: schemas.ChatCompletionRequest, key: bedrockBearer(func(c *schemas.BedrockKeyConfig) {
			c.Endpoints = &schemas.BedrockEndpoints{Runtime: schemas.NewSecretVar("vpce.example.com")}
		}), wantErr: "endpoint overrides"},
		{name: "bedrock base URL chat", provider: schemas.Bedrock, requestType: schemas.ChatCompletionRequest, key: bedrockBearer(nil), baseURL: "https://x.example.com"},
		{name: "bedrock base URL list models", provider: schemas.Bedrock, requestType: schemas.ListModelsRequest, key: bedrockBearer(nil), baseURL: "https://x.example.com", wantErr: "only for model invocation"},
		{name: "bedrock base URL batch", provider: schemas.Bedrock, requestType: schemas.BatchCreateRequest, key: bedrockStatic, baseURL: "https://x.example.com", wantErr: "only for model invocation"},
		{name: "bedrock base URL endpoints", provider: schemas.Bedrock, requestType: schemas.ChatCompletionRequest, key: bedrockBearer(func(c *schemas.BedrockKeyConfig) {
			c.Endpoints = &schemas.BedrockEndpoints{Runtime: schemas.NewSecretVar("vpce.example.com")}
		}), baseURL: "https://x.example.com", wantErr: "endpoint overrides"},
		// Vertex with a caller-supplied access token.
		{name: "vertex access token chat", provider: schemas.Vertex, requestType: schemas.ChatCompletionRequest, key: vertexToken(nil)},
		{name: "vertex access token batch", provider: schemas.Vertex, requestType: schemas.BatchCreateRequest, key: vertexToken(nil)},
		{name: "vertex access token files", provider: schemas.Vertex, requestType: schemas.FileUploadRequest, key: vertexToken(nil)},
		{name: "vertex access token list models", provider: schemas.Vertex, requestType: schemas.ListModelsRequest, key: vertexToken(nil)},
		{name: "vertex domain-scoped project", provider: schemas.Vertex, requestType: schemas.ChatCompletionRequest, key: vertexToken(func(k *schemas.Key) {
			k.VertexKeyConfig.ProjectID = *schemas.NewSecretVar("example.com:my-project")
		})},
		{name: "vertex no key config", provider: schemas.Vertex, requestType: schemas.ChatCompletionRequest, key: schemas.Key{}, wantErr: "vertex_key_config"},
		{name: "vertex empty token", provider: schemas.Vertex, requestType: schemas.ChatCompletionRequest, key: vertexToken(func(k *schemas.Key) { k.VertexKeyConfig.AccessToken = "" }), wantErr: "non-empty"},
		{name: "vertex token with newline", provider: schemas.Vertex, requestType: schemas.ChatCompletionRequest, key: vertexToken(func(k *schemas.Key) {
			k.VertexKeyConfig.AccessToken = "ya29.x\r\nX-Injected: 1"
		}), wantErr: "control characters"},
		{name: "vertex API key", provider: schemas.Vertex, requestType: schemas.ChatCompletionRequest, key: vertexToken(func(k *schemas.Key) { k.Value = *schemas.NewSecretVar("AIza") }), wantErr: "API keys"},
		{name: "vertex service account", provider: schemas.Vertex, requestType: schemas.ChatCompletionRequest, key: vertexToken(func(k *schemas.Key) {
			k.VertexKeyConfig.AuthCredentials = *schemas.NewSecretVar(`{"type":"service_account"}`)
		}), wantErr: "service-account"},
		{name: "vertex AWS workload identity", provider: schemas.Vertex, requestType: schemas.ChatCompletionRequest, key: vertexToken(func(k *schemas.Key) {
			k.VertexKeyConfig.AWSWorkloadIdentity = &schemas.VertexAWSWorkloadIdentityConfig{Audience: *schemas.NewSecretVar("//iam.googleapis.com/x")}
		}), wantErr: "AWS workload identity"},
		{name: "vertex no region", provider: schemas.Vertex, requestType: schemas.ChatCompletionRequest, key: vertexToken(func(k *schemas.Key) { k.VertexKeyConfig.Region = schemas.SecretVar{} }), wantErr: "region"},
		{name: "vertex crafted region", provider: schemas.Vertex, requestType: schemas.ChatCompletionRequest, key: vertexToken(func(k *schemas.Key) {
			k.VertexKeyConfig.Region = *schemas.NewSecretVar("evil.com/#")
		}), wantErr: "region"},
		{name: "vertex no project", provider: schemas.Vertex, requestType: schemas.ChatCompletionRequest, key: vertexToken(func(k *schemas.Key) { k.VertexKeyConfig.ProjectID = schemas.SecretVar{} }), wantErr: "project_id"},
		{name: "vertex crafted project", provider: schemas.Vertex, requestType: schemas.ChatCompletionRequest, key: vertexToken(func(k *schemas.Key) {
			k.VertexKeyConfig.ProjectID = *schemas.NewSecretVar("p/../../x")
		}), wantErr: "project_id"},
		{name: "vertex crafted project number", provider: schemas.Vertex, requestType: schemas.ChatCompletionRequest, key: vertexToken(func(k *schemas.Key) {
			k.VertexKeyConfig.ProjectNumber = *schemas.NewSecretVar("12/34")
		}), wantErr: "project_number"},
		{name: "vertex crafted alias region", provider: schemas.Vertex, requestType: schemas.ChatCompletionRequest, key: vertexToken(func(k *schemas.Key) {
			k.Aliases = schemas.KeyAliases{"m": {ModelID: "m", Region: region("evil.com/#")}}
		}), wantErr: "alias regions"},
		{name: "vertex crafted alias project", provider: schemas.Vertex, requestType: schemas.ChatCompletionRequest, key: vertexToken(func(k *schemas.Key) {
			k.Aliases = schemas.KeyAliases{"m": {ModelID: "m", VertexAliasCfg: &schemas.VertexAliasCfg{ProjectID: region("a?b")}}}
		}), wantErr: "alias project IDs"},
		{name: "vertex base URL", provider: schemas.Vertex, requestType: schemas.ChatCompletionRequest, key: vertexToken(nil), baseURL: "https://x.example.com", wantErr: "not a base URL"},
	}
	b := &Bifrost{logger: NewDefaultLogger(schemas.LogLevelError)}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			instance, err := b.createBaseProvider(tt.provider, requestScopedProviderConfig(false))
			if err != nil {
				t.Fatalf("createBaseProvider: %v", err)
			}
			_, err = instance.(schemas.RequestScopedProvider).ForRequest(tt.requestType, tt.key, tt.baseURL)
			if tt.wantErr == "" {
				if err != nil {
					t.Fatalf("ForRequest: %v", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("ForRequest error = %v, want one containing %q", err, tt.wantErr)
			}
		})
	}

	// These fall back to ambient credentials or cache tokens per credential, so none of them
	// can be configured per request.
	// Vertex is served per request with a caller-supplied access token only.
	for _, provider := range []schemas.ModelProvider{schemas.BedrockMantle, schemas.Databricks, schemas.GithubCopilot} {
		instance, err := b.createBaseProvider(provider, requestScopedProviderConfig(false))
		if err != nil {
			t.Fatalf("createBaseProvider(%s): %v", provider, err)
		}
		if _, ok := instance.(schemas.RequestScopedProvider); ok {
			t.Errorf("%s must not implement schemas.RequestScopedProvider", provider)
		}
	}
}

// requestScopedTestPlugin gives requests request-scoped configuration: configure runs in
// PreRequestHook and attempt in PreLLMHook with the attempt's fallback index. It counts the
// hook calls and records the errors PostLLMHook observes.
type requestScopedTestPlugin struct {
	configure func(ctx *schemas.BifrostContext, req *schemas.BifrostRequest)
	attempt   func(fallbackIndex int, req *schemas.BifrostRequest)

	preRequestCalls, preLLMCalls, postLLMCalls atomic.Int64

	mu         sync.Mutex
	postErrors []string
}

func (p *requestScopedTestPlugin) GetName() string { return "request-scoped-test" }
func (p *requestScopedTestPlugin) Cleanup() error  { return nil }
func (p *requestScopedTestPlugin) PreRequestHook(ctx *schemas.BifrostContext, req *schemas.BifrostRequest) error {
	p.preRequestCalls.Add(1)
	if p.configure != nil {
		p.configure(ctx, req)
	}
	return nil
}
func (p *requestScopedTestPlugin) PreLLMHook(ctx *schemas.BifrostContext, req *schemas.BifrostRequest) (*schemas.BifrostRequest, *schemas.LLMPluginShortCircuit, error) {
	p.preLLMCalls.Add(1)
	if p.attempt != nil {
		fallbackIndex, _ := ctx.Value(schemas.BifrostContextKeyFallbackIndex).(int)
		p.attempt(fallbackIndex, req)
	}
	return req, nil, nil
}
func (p *requestScopedTestPlugin) PostLLMHook(_ *schemas.BifrostContext, resp *schemas.BifrostResponse, bifrostErr *schemas.BifrostError) (*schemas.BifrostResponse, *schemas.BifrostError, error) {
	p.postLLMCalls.Add(1)
	if bifrostErr != nil && bifrostErr.Error != nil {
		p.mu.Lock()
		p.postErrors = append(p.postErrors, requestScopedErrorText(bifrostErr))
		p.mu.Unlock()
	}
	return resp, bifrostErr, nil
}

// requestScopedErrorText is an error's message followed by its underlying error, where dial
// failures carry their reason.
func requestScopedErrorText(err *schemas.BifrostError) string {
	if err == nil || err.Error == nil {
		return ""
	}
	if err.Error.Error != nil {
		return err.Error.Message + ": " + err.Error.Error.Error()
	}
	return err.Error.Message
}

// observedErrors returns the errors PostLLMHook has observed so far.
func (p *requestScopedTestPlugin) observedErrors() []string {
	p.mu.Lock()
	defer p.mu.Unlock()
	return append([]string(nil), p.postErrors...)
}

// mustConfigureOpenAI gives req request-scoped OpenAI configuration and fails the test on error.
func mustConfigureOpenAI(t *testing.T, req *schemas.BifrostRequest, key, baseURL string) {
	t.Helper()
	if err := req.UpdateProviderKey(schemas.OpenAI, schemas.Key{Value: *schemas.NewSecretVar(key)}); err != nil {
		t.Errorf("UpdateProviderKey: %v", err)
	}
	if err := req.UpdateProviderBaseURL(schemas.OpenAI, baseURL); err != nil {
		t.Errorf("UpdateProviderBaseURL: %v", err)
	}
	if err := req.UpdateProviderAllowPrivateNetwork(schemas.OpenAI, true); err != nil {
		t.Errorf("UpdateProviderAllowPrivateNetwork: %v", err)
	}
}

// requestScopedTestCall is one upstream request the test server received.
type requestScopedTestCall struct {
	path, auth string
}

// requestScopedTestServer answers OpenAI chat, responses, embeddings and model listing in JSON
// or SSE, and records the path and credential of every request.
func requestScopedTestServer(t *testing.T) (*httptest.Server, func() []requestScopedTestCall) {
	t.Helper()
	var mu sync.Mutex
	var calls []requestScopedTestCall
	chatStream := sseHandler(`{"id":"c1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":"hello"},"finish_reason":null}]}`,
		`{"id":"c1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`)
	responsesStream := func(w http.ResponseWriter) {
		w.Header().Set("Content-Type", "text/event-stream")
		for _, event := range []string{
			`{"type":"response.created","sequence_number":0,"response":{"id":"resp_1","object":"response","created_at":1,"status":"in_progress","model":"m","output":[]}}`,
			`{"type":"response.output_text.delta","sequence_number":1,"item_id":"msg_1","output_index":0,"content_index":0,"delta":"hello"}`,
			`{"type":"response.completed","sequence_number":2,"response":{"id":"resp_1","object":"response","created_at":1,"status":"completed","model":"m","output":[],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}`,
		} {
			_, _ = fmt.Fprintf(w, "data: %s\n\n", event)
		}
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		mu.Lock()
		calls = append(calls, requestScopedTestCall{path: r.URL.Path, auth: r.Header.Get("Authorization")})
		mu.Unlock()
		stream := strings.Contains(string(body), `"stream":true`)
		switch {
		case strings.HasSuffix(r.URL.Path, "/v1/chat/completions") && stream:
			chatStream(w, r)
		case strings.HasSuffix(r.URL.Path, "/v1/chat/completions"):
			w.Header().Set("Content-Type", "application/json")
			_, _ = fmt.Fprint(w, `{"id":"c1","object":"chat.completion","created":1,"model":"m",`+
				`"choices":[{"index":0,"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}],`+
				`"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
		case strings.HasSuffix(r.URL.Path, "/v1/responses") && stream:
			responsesStream(w)
		case strings.HasSuffix(r.URL.Path, "/v1/responses"):
			w.Header().Set("Content-Type", "application/json")
			_, _ = fmt.Fprint(w, `{"id":"resp_1","object":"response","created_at":1,"status":"completed","model":"m",`+
				`"output":[{"type":"message","id":"msg_1","status":"completed","role":"assistant","content":[{"type":"output_text","text":"hello","annotations":[]}]}],`+
				`"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}`)
		case strings.HasSuffix(r.URL.Path, "/v1/embeddings"):
			w.Header().Set("Content-Type", "application/json")
			_, _ = fmt.Fprint(w, `{"object":"list","data":[{"object":"embedding","index":0,"embedding":[0.1,0.2]}],"model":"m","usage":{"prompt_tokens":1,"total_tokens":1}}`)
		case strings.HasSuffix(r.URL.Path, "/v1/models"):
			openAIStyleModelsHandler("m")(w, r)
		case strings.HasSuffix(r.URL.Path, "/v1/batches"):
			w.Header().Set("Content-Type", "application/json")
			_, _ = fmt.Fprint(w, `{"id":"batch_1","object":"batch","endpoint":"/v1/chat/completions","input_file_id":"file-1",`+
				`"completion_window":"24h","status":"validating","created_at":1}`)
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(server.Close)
	return server, func() []requestScopedTestCall {
		mu.Lock()
		defer mu.Unlock()
		return append([]requestScopedTestCall(nil), calls...)
	}
}

// newRequestScopedTestClient starts a client with plugin and the given account and KV store.
func newRequestScopedTestClient(t *testing.T, account schemas.Account, plugin schemas.LLMPlugin, kvStore schemas.KVStore) *Bifrost {
	t.Helper()
	client, err := Init(context.Background(), schemas.BifrostConfig{
		Account:    account,
		LLMPlugins: []schemas.LLMPlugin{plugin},
		KVStore:    kvStore,
		Logger:     NewDefaultLogger(schemas.LogLevelError),
	})
	if err != nil {
		t.Fatalf("Init: %v", err)
	}
	t.Cleanup(client.Shutdown)
	return client
}

// requestScopedChat is a minimal chat request for provider and model.
func requestScopedChat(provider schemas.ModelProvider, model string, fallbacks ...schemas.Fallback) *schemas.BifrostChatRequest {
	content := "hello"
	return &schemas.BifrostChatRequest{
		Provider:  provider,
		Model:     model,
		Input:     []schemas.ChatMessage{{Role: schemas.ChatMessageRoleUser, Content: &schemas.ChatMessageContent{ContentStr: &content}}},
		Params:    &schemas.ChatParameters{},
		Fallbacks: fallbacks,
	}
}

// scopedDrainStream reads a stream to its end and returns the error chunks it carried.
func scopedDrainStream(stream chan *schemas.BifrostStreamChunk) []string {
	var errs []string
	for chunk := range stream {
		if chunk.BifrostError != nil && chunk.BifrostError.Error != nil {
			errs = append(errs, chunk.BifrostError.Error.Message)
		}
	}
	return errs
}

// countingAccount counts provider config lookups on top of MockAccount.
type countingAccount struct {
	*MockAccount
	configLookups atomic.Int64
}

func (a *countingAccount) GetConfigForProvider(provider schemas.ModelProvider) (*schemas.ProviderConfig, error) {
	a.configLookups.Add(1)
	return a.MockAccount.GetConfigForProvider(provider)
}

// TestRequestScopedConfiguration_UnconfiguredProvider sends each kind of request to a provider the
// account does not configure, with the credential and endpoint supplied by a plugin.
func TestRequestScopedConfiguration_UnconfiguredProvider(t *testing.T) {
	server, calls := requestScopedTestServer(t)
	plugin := &requestScopedTestPlugin{configure: func(_ *schemas.BifrostContext, req *schemas.BifrostRequest) {
		mustConfigureOpenAI(t, req, "sk-scoped", server.URL)
	}}
	client := newRequestScopedTestClient(t, NewMockAccount(), plugin, nil)
	input := "hello"

	tests := []struct {
		name string
		path string
		call func(ctx *schemas.BifrostContext) *schemas.BifrostError
	}{
		{"chat", "/v1/chat/completions", func(ctx *schemas.BifrostContext) *schemas.BifrostError {
			_, err := client.ChatCompletionRequest(ctx, requestScopedChat(schemas.OpenAI, "m"))
			return err
		}},
		{"chat stream", "/v1/chat/completions", func(ctx *schemas.BifrostContext) *schemas.BifrostError {
			stream, err := client.ChatCompletionStreamRequest(ctx, requestScopedChat(schemas.OpenAI, "m"))
			if err != nil {
				return err
			}
			if errs := scopedDrainStream(stream); len(errs) > 0 {
				t.Errorf("stream errors: %v", errs)
			}
			return nil
		}},
		{"responses", "/v1/responses", func(ctx *schemas.BifrostContext) *schemas.BifrostError {
			_, err := client.ResponsesRequest(ctx, &schemas.BifrostResponsesRequest{Provider: schemas.OpenAI, Model: "m",
				Input: []schemas.ResponsesMessage{{Role: schemas.Ptr(schemas.ResponsesInputMessageRoleUser), Content: &schemas.ResponsesMessageContent{ContentStr: &input}}}})
			return err
		}},
		{"responses stream", "/v1/responses", func(ctx *schemas.BifrostContext) *schemas.BifrostError {
			stream, err := client.ResponsesStreamRequest(ctx, &schemas.BifrostResponsesRequest{Provider: schemas.OpenAI, Model: "m",
				Input: []schemas.ResponsesMessage{{Role: schemas.Ptr(schemas.ResponsesInputMessageRoleUser), Content: &schemas.ResponsesMessageContent{ContentStr: &input}}}})
			if err != nil {
				return err
			}
			if errs := scopedDrainStream(stream); len(errs) > 0 {
				t.Errorf("stream errors: %v", errs)
			}
			return nil
		}},
		{"embedding", "/v1/embeddings", func(ctx *schemas.BifrostContext) *schemas.BifrostError {
			_, err := client.EmbeddingRequest(ctx, &schemas.BifrostEmbeddingRequest{Provider: schemas.OpenAI, Model: "m",
				Input: []schemas.EmbeddingInputItem{{Content: schemas.EmbeddingContent{{Type: schemas.EmbeddingContentPartTypeText, Text: &input}}}}})
			return err
		}},
		{"list models", "/v1/models", func(ctx *schemas.BifrostContext) *schemas.BifrostError {
			_, err := client.ListModelsRequest(ctx, &schemas.BifrostListModelsRequest{Provider: schemas.OpenAI})
			return err
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			before := len(calls())
			if err := tt.call(schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)); err != nil {
				t.Fatalf("request failed: %s", err.GetErrorString())
			}
			got := calls()[before:]
			if len(got) != 1 || got[0].path != tt.path || got[0].auth != "Bearer sk-scoped" {
				t.Fatalf("upstream calls = %+v, want one %s call with the request-scoped key", got, tt.path)
			}
		})
	}
}

// TestRequestScopedConfiguration_LeavesNoState pins that request-scoped attempts register no
// provider, queue or worker, never consult the account, and leave nothing for session affinity,
// however many distinct credentials and endpoints they carry.
func TestRequestScopedConfiguration_LeavesNoState(t *testing.T) {
	server, calls := requestScopedTestServer(t)
	plugin := &requestScopedTestPlugin{configure: func(ctx *schemas.BifrostContext, req *schemas.BifrostRequest) {
		n := ctx.Value(schemas.BifrostContextKeySessionID).(string)
		mustConfigureOpenAI(t, req, "sk-"+n, server.URL+"/"+n)
	}}
	account := &countingAccount{MockAccount: NewMockAccount()}
	kvStore := newMockKVStore()
	client := newRequestScopedTestClient(t, account, plugin, kvStore)

	for i := range 50 {
		n := fmt.Sprintf("caller-%d", i)
		ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
		ctx.SetValue(schemas.BifrostContextKeySessionID, n)
		if i%2 == 0 {
			if _, err := client.ChatCompletionRequest(ctx, requestScopedChat(schemas.OpenAI, "m")); err != nil {
				t.Fatalf("request %d: %s", i, err.GetErrorString())
			}
		} else {
			stream, err := client.ChatCompletionStreamRequest(ctx, requestScopedChat(schemas.OpenAI, "m"))
			if err != nil {
				t.Fatalf("request %d: %s", i, err.GetErrorString())
			}
			scopedDrainStream(stream)
		}
		if got := calls()[i]; got.path != "/"+n+"/v1/chat/completions" || got.auth != "Bearer sk-"+n {
			t.Fatalf("request %d reached %+v", i, got)
		}
	}

	countEntries := func(m *sync.Map) int {
		n := 0
		m.Range(func(_, _ any) bool { n++; return true })
		return n
	}
	if n := countEntries(&client.requestQueues); n != 0 {
		t.Errorf("%d provider queues were created", n)
	}
	if n := countEntries(&client.waitGroups); n != 0 {
		t.Errorf("%d worker groups were created", n)
	}
	if providers := client.providers.Load(); providers != nil && len(*providers) != 0 {
		t.Errorf("%d providers were registered", len(*providers))
	}
	if instances := client.requestScopedProviders.Load(); instances == nil || len(*instances) != 1 {
		t.Errorf("want exactly one request-scoped instance for one provider and policy, got %v", instances)
	}
	if n := account.KeyLookupCount(schemas.OpenAI); n != 0 {
		t.Errorf("the account was asked for keys %d times", n)
	}
	if n := account.configLookups.Load(); n != 0 {
		t.Errorf("the account was asked for provider config %d times", n)
	}
	kvStore.mu.RLock()
	defer kvStore.mu.RUnlock()
	if len(kvStore.data) != 0 {
		t.Errorf("session affinity stored %d entries", len(kvStore.data))
	}
}

// TestRequestScopedConfiguration_NoGoroutineLeak pins that request-scoped attempts, which run in
// the caller's goroutine, start nothing that outlives them.
func TestRequestScopedConfiguration_NoGoroutineLeak(t *testing.T) {
	server, _ := requestScopedTestServer(t)
	// Without keep-alives the server holds no goroutine per pooled connection, so the count
	// cannot move with how many connections the client happens to keep open.
	server.Config.SetKeepAlivesEnabled(false)
	plugin := &requestScopedTestPlugin{configure: func(_ *schemas.BifrostContext, req *schemas.BifrostRequest) {
		mustConfigureOpenAI(t, req, "sk-scoped", server.URL)
	}}
	client := newRequestScopedTestClient(t, NewMockAccount(), plugin, nil)
	run := func() {
		ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
		if _, err := client.ChatCompletionRequest(ctx, requestScopedChat(schemas.OpenAI, "m")); err != nil {
			t.Fatalf("request: %s", err.GetErrorString())
		}
		stream, err := client.ChatCompletionStreamRequest(ctx, requestScopedChat(schemas.OpenAI, "m"))
		if err != nil {
			t.Fatalf("stream request: %s", err.GetErrorString())
		}
		scopedDrainStream(stream)
	}
	// Warm up: the first request builds the instance and opens the pooled connection.
	run()
	memtest.AssertNoGoroutineLeak(t, func() {
		for range 10 {
			run()
		}
	})
}

// reorderingAffinity moves the last route of every chain to the front and counts how often it
// is consulted.
type reorderingAffinity struct {
	resolves, observes atomic.Int64
}

func (a *reorderingAffinity) ResolveRoute(_ *schemas.BifrostContext, _ schemas.Route, chain []schemas.Route) []schemas.Route {
	a.resolves.Add(1)
	return append([]schemas.Route{chain[len(chain)-1]}, chain[:len(chain)-1]...)
}
func (a *reorderingAffinity) ResolveKey(*schemas.BifrostContext, schemas.ModelProvider, string, []schemas.Key) (schemas.Key, bool) {
	return schemas.Key{}, false
}
func (a *reorderingAffinity) Observe(*schemas.BifrostContext, schemas.Route, schemas.RouteOutcome) {
	a.observes.Add(1)
}

// TestRequestScopedConfiguration_SkipsSessionAffinity pins that session affinity neither reroutes
// nor learns from a request carrying request-scoped configuration: its routes are the plugin's.
func TestRequestScopedConfiguration_SkipsSessionAffinity(t *testing.T) {
	server, calls := requestScopedTestServer(t)
	affinity := &reorderingAffinity{}
	client, err := Init(context.Background(), schemas.BifrostConfig{
		Account: NewMockAccount(),
		LLMPlugins: []schemas.LLMPlugin{&requestScopedTestPlugin{configure: func(_ *schemas.BifrostContext, req *schemas.BifrostRequest) {
			mustConfigureOpenAI(t, req, "sk-scoped", server.URL)
		}}},
		SessionAffinity: affinity,
		Logger:          NewDefaultLogger(schemas.LogLevelError),
	})
	if err != nil {
		t.Fatalf("Init: %v", err)
	}
	defer client.Shutdown()

	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	ctx.SetValue(schemas.BifrostContextKeySessionID, "session-1")
	if _, err := client.ChatCompletionRequest(ctx, requestScopedChat(schemas.OpenAI, "m", schemas.Fallback{Provider: schemas.Anthropic, Model: "claude"})); err != nil {
		t.Fatalf("request: %s", err.GetErrorString())
	}
	if got := calls(); len(got) != 1 {
		t.Fatalf("upstream calls = %+v, want the request-scoped primary only", got)
	}
	if n, m := affinity.resolves.Load(), affinity.observes.Load(); n != 0 || m != 0 {
		t.Fatalf("session affinity was consulted: %d route resolutions, %d observations", n, m)
	}
}

// TestRequestScopedConfiguration_ReleasesOverride pins that nothing Bifrost keeps holds a
// request's provider configuration, and with it the credential, once the request is over.
func TestRequestScopedConfiguration_ReleasesOverride(t *testing.T) {
	server, _ := requestScopedTestServer(t)
	var captured *schemas.ProviderOverride
	plugin := &requestScopedTestPlugin{configure: func(_ *schemas.BifrostContext, req *schemas.BifrostRequest) {
		mustConfigureOpenAI(t, req, "sk-released", server.URL)
		captured = req.ProviderOverrideFor(schemas.OpenAI)
	}}
	client := newRequestScopedTestClient(t, NewMockAccount(), plugin, nil)

	// A pooled request drops its overrides rather than keeping their backing array for reuse.
	pooled := &schemas.BifrostRequest{}
	mustConfigureOpenAI(t, pooled, "sk-pooled", server.URL)
	resetBifrostRequest(pooled)
	if pooled.ProviderOverrides != nil {
		t.Fatal("resetBifrostRequest kept the request's provider overrides")
	}

	for _, stream := range []bool{false, true} {
		memtest.AssertReleased(t, fmt.Sprintf("the provider override (stream=%v)", stream), func() *schemas.ProviderOverride {
			ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
			if stream {
				ch, err := client.ChatCompletionStreamRequest(ctx, requestScopedChat(schemas.OpenAI, "m"))
				if err != nil {
					t.Fatalf("stream request: %s", err.GetErrorString())
				}
				scopedDrainStream(ch)
			} else if _, err := client.ChatCompletionRequest(ctx, requestScopedChat(schemas.OpenAI, "m")); err != nil {
				t.Fatalf("request: %s", err.GetErrorString())
			}
			override := captured
			captured = nil
			return override
		})
	}
}

// TestRequestScopedConfiguration_PerAttemptCredentials sets credentials from PreLLMHook per
// attempt of a request that PreRequestHook gave request-scoped configuration: two attempts
// against the same provider type use different keys and endpoints, a fallback to another
// provider uses its own, and a fallback to an unconfigured provider without request-scoped
// configuration fails closed without reaching anything.
func TestRequestScopedConfiguration_PerAttemptCredentials(t *testing.T) {
	type seen struct {
		mu   sync.Mutex
		keys []string
	}
	record := func(s *seen, status int, body func(w http.ResponseWriter, r *http.Request)) *httptest.Server {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			s.mu.Lock()
			s.keys = append(s.keys, r.Header.Get("Authorization")+r.Header.Get("x-api-key"))
			s.mu.Unlock()
			if status != http.StatusOK {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(status)
				_, _ = fmt.Fprint(w, `{"error":{"message":"upstream failed","type":"server_error"}}`)
				return
			}
			body(w, r)
		}))
		t.Cleanup(server.Close)
		return server
	}
	var seenA, seenB, seenC seen
	serverA := record(&seenA, http.StatusInternalServerError, nil)
	serverB := record(&seenB, http.StatusInternalServerError, nil)
	serverC := record(&seenC, http.StatusOK, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"msg_1","type":"message","role":"assistant","model":"claude-test","content":[{"type":"text","text":"hello"}],`+
			`"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
	})

	plugin := &requestScopedTestPlugin{configure: func(_ *schemas.BifrostContext, req *schemas.BifrostRequest) {
		mustConfigureOpenAI(t, req, "sk-request", serverA.URL)
	}, attempt: func(fallbackIndex int, req *schemas.BifrostRequest) {
		switch fallbackIndex {
		case 0:
			mustConfigureOpenAI(t, req, "sk-a", serverA.URL)
		case 1:
			mustConfigureOpenAI(t, req, "sk-b", serverB.URL)
		case 3:
			if err := req.UpdateProviderKey(schemas.Anthropic, schemas.Key{Value: *schemas.NewSecretVar("sk-c")}); err != nil {
				t.Errorf("UpdateProviderKey: %v", err)
			}
			if err := req.UpdateProviderBaseURL(schemas.Anthropic, serverC.URL); err != nil {
				t.Errorf("UpdateProviderBaseURL: %v", err)
			}
			if err := req.UpdateProviderAllowPrivateNetwork(schemas.Anthropic, true); err != nil {
				t.Errorf("UpdateProviderAllowPrivateNetwork: %v", err)
			}
		}
	}}
	client := newRequestScopedTestClient(t, NewMockAccount(), plugin, nil)

	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	resp, err := client.ChatCompletionRequest(ctx, requestScopedChat(schemas.OpenAI, "model-a",
		schemas.Fallback{Provider: schemas.OpenAI, Model: "model-b"},
		schemas.Fallback{Provider: schemas.Groq, Model: "unconfigured"},
		schemas.Fallback{Provider: schemas.Anthropic, Model: "claude-test"},
	))
	if err != nil {
		t.Fatalf("request: %s", err.GetErrorString())
	}
	if resp.ExtraFields.RoutingInfo.Provider != schemas.Anthropic {
		t.Fatalf("served by %s, want the anthropic fallback", resp.ExtraFields.RoutingInfo.Provider)
	}
	for _, tc := range []struct {
		name string
		seen *seen
		want string
	}{{"first openai attempt", &seenA, "Bearer sk-a"}, {"second openai attempt", &seenB, "Bearer sk-b"}, {"anthropic fallback", &seenC, "sk-c"}} {
		tc.seen.mu.Lock()
		if len(tc.seen.keys) != 1 || tc.seen.keys[0] != tc.want {
			t.Errorf("%s sent credentials %q, want exactly [%q]", tc.name, tc.seen.keys, tc.want)
		}
		tc.seen.mu.Unlock()
	}
	if errs := plugin.observedErrors(); !slices.ContainsFunc(errs, func(e string) bool { return strings.Contains(e, "failed to get config for provider groq") }) {
		t.Errorf("the unconfigured fallback must fail through the post-hooks, observed %q", errs)
	}
}

// TestRequestScopedConfiguration_OverridesConfiguredProvider pins that request-scoped
// configuration replaces a configured provider's keys and endpoint for the request.
func TestRequestScopedConfiguration_OverridesConfiguredProvider(t *testing.T) {
	configured, configuredCalls := requestScopedTestServer(t)
	scoped, scopedCalls := requestScopedTestServer(t)
	account := NewMockAccount()
	account.AddProviderWithBaseURL(schemas.OpenAI, 1, 1, configured.URL)
	plugin := &requestScopedTestPlugin{configure: func(_ *schemas.BifrostContext, req *schemas.BifrostRequest) {
		mustConfigureOpenAI(t, req, "sk-scoped", scoped.URL)
	}}
	client := newRequestScopedTestClient(t, account, plugin, nil)

	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	if _, err := client.ChatCompletionRequest(ctx, requestScopedChat(schemas.OpenAI, "m")); err != nil {
		t.Fatalf("request: %s", err.GetErrorString())
	}
	if got := scopedCalls(); len(got) != 1 || got[0].auth != "Bearer sk-scoped" {
		t.Errorf("request-scoped endpoint saw %+v", got)
	}
	if got := configuredCalls(); len(got) != 0 {
		t.Errorf("configured endpoint was reached: %+v", got)
	}
	if n := account.KeyLookupCount(schemas.OpenAI); n != 0 {
		t.Errorf("configured keys were looked up %d times", n)
	}
	if selected, _ := ctx.Value(schemas.BifrostContextKeySelectedKeyID).(string); selected != requestScopedKeyID {
		t.Errorf("selected key ID = %q, want %q for a key without an ID", selected, requestScopedKeyID)
	}
}

// TestRequestScopedConfiguration_FailsClosed pins that configuration Bifrost cannot serve without
// ambient credentials or a permitted destination fails the attempt, and that the post-hooks see
// the failure.
func TestRequestScopedConfiguration_FailsClosed(t *testing.T) {
	server, calls := requestScopedTestServer(t)
	tests := []struct {
		name      string
		provider  schemas.ModelProvider
		configure func(req *schemas.BifrostRequest) error
		attempt   func(req *schemas.BifrostRequest)
		wantErr   string
	}{
		{name: "provider without support", provider: schemas.BedrockMantle, configure: func(req *schemas.BifrostRequest) error {
			return req.UpdateProviderKey(schemas.BedrockMantle, schemas.Key{Value: *schemas.NewSecretVar("token")})
		}, wantErr: "does not support request-scoped configuration"},
		{name: "vertex without an access token", provider: schemas.Vertex, configure: func(req *schemas.BifrostRequest) error {
			return req.UpdateProviderKey(schemas.Vertex, schemas.Key{VertexKeyConfig: &schemas.VertexKeyConfig{
				ProjectID: *schemas.NewSecretVar("my-project"), Region: *schemas.NewSecretVar("us-central1"),
			}})
		}, wantErr: "non-empty vertex_key_config access token"},
		{name: "unknown provider", provider: "not-a-provider", configure: func(req *schemas.BifrostRequest) error {
			return req.UpdateProviderKey("not-a-provider", schemas.Key{Value: *schemas.NewSecretVar("k")})
		}, wantErr: "unsupported provider"},
		{name: "base URL the provider cannot honor", provider: schemas.VLLM, configure: func(req *schemas.BifrostRequest) error {
			return req.UpdateProviderBaseURL(schemas.VLLM, "https://vllm.example.com")
		}, wantErr: "vllm_key_config.url"},
		{name: "loopback without private networks", provider: schemas.OpenAI, configure: func(req *schemas.BifrostRequest) error {
			return req.UpdateProviderBaseURL(schemas.OpenAI, server.URL)
		}, wantErr: "private IP 127.0.0.1"},
		{name: "RFC 1918 without private networks", provider: schemas.OpenAI, configure: func(req *schemas.BifrostRequest) error {
			return req.UpdateProviderBaseURL(schemas.OpenAI, "http://10.0.0.1:8000")
		}, wantErr: "private IP 10.0.0.1"},
		{name: "link-local with private networks", provider: schemas.OpenAI, configure: func(req *schemas.BifrostRequest) error {
			if err := req.UpdateProviderAllowPrivateNetwork(schemas.OpenAI, true); err != nil {
				return err
			}
			return req.UpdateProviderBaseURL(schemas.OpenAI, "http://169.254.169.254")
		}, wantErr: "link-local IP 169.254.169.254"},
		{name: "override removed in PreLLMHook", provider: schemas.OpenAI, configure: func(req *schemas.BifrostRequest) error {
			return req.UpdateProviderBaseURL(schemas.OpenAI, server.URL)
		}, attempt: func(req *schemas.BifrostRequest) { req.ProviderOverrides = nil }, wantErr: "failed to get config for provider openai"},
	}
	for _, tt := range tests {
		for _, stream := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/stream=%v", tt.name, stream), func(t *testing.T) {
				plugin := &requestScopedTestPlugin{configure: func(_ *schemas.BifrostContext, req *schemas.BifrostRequest) {
					if err := tt.configure(req); err != nil {
						t.Errorf("configure: %v", err)
					}
				}}
				if tt.attempt != nil {
					plugin.attempt = func(_ int, req *schemas.BifrostRequest) { tt.attempt(req) }
				}
				client := newRequestScopedTestClient(t, NewMockAccount(), plugin, nil)
				before := len(calls())

				ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
				var bifrostErr *schemas.BifrostError
				if stream {
					var ch chan *schemas.BifrostStreamChunk
					if ch, bifrostErr = client.ChatCompletionStreamRequest(ctx, requestScopedChat(tt.provider, "m")); ch != nil {
						if errs := scopedDrainStream(ch); len(errs) > 0 {
							bifrostErr = &schemas.BifrostError{Error: &schemas.ErrorField{Message: errs[0]}}
						}
					}
				} else {
					_, bifrostErr = client.ChatCompletionRequest(ctx, requestScopedChat(tt.provider, "m"))
				}
				if !strings.Contains(requestScopedErrorText(bifrostErr), tt.wantErr) {
					t.Fatalf("error = %v, want one containing %q", bifrostErr, tt.wantErr)
				}
				if errs := plugin.observedErrors(); !slices.ContainsFunc(errs, func(e string) bool { return strings.Contains(e, tt.wantErr) }) {
					t.Errorf("PostLLMHook did not observe the failure, saw %q", errs)
				}
				if got := calls()[before:]; len(got) != 0 {
					t.Errorf("upstream was reached: %+v", got)
				}
			})
		}
	}
}

// TestRequestScopedConfiguration_InstanceLookupAllocations pins that finding the instance for a
// request-scoped attempt allocates nothing once it exists.
func TestRequestScopedConfiguration_InstanceLookupAllocations(t *testing.T) {
	client := newRequestScopedTestClient(t, NewMockAccount(), &requestScopedTestPlugin{}, nil)
	class := requestScopedClass{provider: schemas.OpenAI, allowPrivateNetwork: true}
	if _, err := client.getRequestScopedProvider(class); err != nil {
		t.Fatal(err)
	}
	if allocs := testing.AllocsPerRun(100, func() {
		if _, err := client.getRequestScopedProvider(class); err != nil {
			t.Fatal(err)
		}
	}); allocs != 0 {
		t.Errorf("getRequestScopedProvider for an existing instance: %v allocs, want 0", allocs)
	}
}

// BenchmarkRequestScopedChatCompletion compares a chat completion against a configured provider
// with the same request carrying request-scoped configuration, unary and streamed, end to end
// against a local server.
func BenchmarkRequestScopedChatCompletion(b *testing.B) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		if strings.Contains(string(body), `"stream":true`) {
			sseHandler(`{"id":"c1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":"stop"}]}`)(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"c1","object":"chat.completion","created":1,"model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}]}`)
	}))
	defer server.Close()

	account := NewMockAccount()
	account.AddProviderWithBaseURL(schemas.OpenAI, 4, 16, server.URL)
	account.SetKeysForProvider(schemas.OpenAI, []schemas.Key{{ID: "k", Value: *schemas.NewSecretVar("sk"), Models: schemas.WhiteList{"*"}, Weight: 1}})
	clients := []struct {
		name   string
		client func() (*Bifrost, error)
	}{
		{"configured", func() (*Bifrost, error) {
			return Init(context.Background(), schemas.BifrostConfig{Account: account, LLMPlugins: []schemas.LLMPlugin{&requestScopedTestPlugin{}}, Logger: NewDefaultLogger(schemas.LogLevelError)})
		}},
		{"request-scoped", func() (*Bifrost, error) {
			plugin := &requestScopedTestPlugin{configure: func(_ *schemas.BifrostContext, req *schemas.BifrostRequest) {
				_ = req.UpdateProviderKey(schemas.OpenAI, schemas.Key{Value: *schemas.NewSecretVar("sk")})
				_ = req.UpdateProviderBaseURL(schemas.OpenAI, server.URL)
				_ = req.UpdateProviderAllowPrivateNetwork(schemas.OpenAI, true)
			}}
			return Init(context.Background(), schemas.BifrostConfig{Account: NewMockAccount(), LLMPlugins: []schemas.LLMPlugin{plugin}, Logger: NewDefaultLogger(schemas.LogLevelError)})
		}},
	}
	for _, c := range clients {
		client, err := c.client()
		if err != nil {
			b.Fatal(err)
		}
		b.Run(c.name+"/unary", func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				if _, err := client.ChatCompletionRequest(schemas.NewBifrostContext(context.Background(), schemas.NoDeadline), requestScopedChat(schemas.OpenAI, "m")); err != nil {
					b.Fatal(err.GetErrorString())
				}
			}
		})
		b.Run(c.name+"/stream", func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				stream, err := client.ChatCompletionStreamRequest(schemas.NewBifrostContext(context.Background(), schemas.NoDeadline), requestScopedChat(schemas.OpenAI, "m"))
				if err != nil {
					b.Fatal(err.GetErrorString())
				}
				scopedDrainStream(stream)
			}
		})
		client.Shutdown()
	}
}

// requestScopedCallerKey carries a test goroutine's identity to the plugin.
type requestScopedCallerKey struct{}

// TestRequestScopedConfiguration_Concurrent runs request-scoped requests from many goroutines,
// each with its own credential and endpoint, and pins that no credential reaches another's
// endpoint. Run with -race.
func TestRequestScopedConfiguration_Concurrent(t *testing.T) {
	var mismatches atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		caller, _, _ := strings.Cut(strings.TrimPrefix(r.URL.Path, "/"), "/")
		if r.Header.Get("Authorization") != "Bearer sk-"+caller {
			mismatches.Add(1)
		}
		body, _ := io.ReadAll(r.Body)
		if strings.Contains(string(body), `"stream":true`) {
			sseHandler(`{"id":"c1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":"stop"}]}`)(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"c1","object":"chat.completion","created":1,"model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}]}`)
	}))
	defer server.Close()
	plugin := &requestScopedTestPlugin{configure: func(ctx *schemas.BifrostContext, req *schemas.BifrostRequest) {
		caller := ctx.Value(requestScopedCallerKey{}).(string)
		mustConfigureOpenAI(t, req, "sk-"+caller, server.URL+"/"+caller)
	}}
	client := newRequestScopedTestClient(t, NewMockAccount(), plugin, nil)

	var wg sync.WaitGroup
	for g := range 64 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range 8 {
				ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
				ctx.SetValue(requestScopedCallerKey{}, fmt.Sprintf("g%d-%d", g, i))
				if i%2 == 0 {
					if _, err := client.ChatCompletionRequest(ctx, requestScopedChat(schemas.OpenAI, "m")); err != nil {
						t.Errorf("goroutine %d: %s", g, err.GetErrorString())
					}
					continue
				}
				stream, err := client.ChatCompletionStreamRequest(ctx, requestScopedChat(schemas.OpenAI, "m"))
				if err != nil {
					t.Errorf("goroutine %d: %s", g, err.GetErrorString())
					continue
				}
				scopedDrainStream(stream)
			}
		}()
	}
	wg.Wait()
	if n := mismatches.Load(); n != 0 {
		t.Fatalf("%d requests reached an endpoint with another request's credential", n)
	}
}

// TestRequestScopedConfiguration_UnconfiguredProviderWithoutOverrides pins that a request carrying
// no request-scoped configuration resolves its provider before PreLLMHook, as it always has: an
// unconfigured provider fails there, without running the LLM hooks, even if PreLLMHook would
// have configured the attempt.
func TestRequestScopedConfiguration_UnconfiguredProviderWithoutOverrides(t *testing.T) {
	server, calls := requestScopedTestServer(t)
	for _, stream := range []bool{false, true} {
		t.Run(fmt.Sprintf("stream=%v", stream), func(t *testing.T) {
			plugin := &requestScopedTestPlugin{attempt: func(_ int, req *schemas.BifrostRequest) {
				mustConfigureOpenAI(t, req, "sk-too-late", server.URL)
			}}
			client := newRequestScopedTestClient(t, NewMockAccount(), plugin, nil)
			ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
			var bifrostErr *schemas.BifrostError
			if stream {
				var ch chan *schemas.BifrostStreamChunk
				if ch, bifrostErr = client.ChatCompletionStreamRequest(ctx, requestScopedChat(schemas.OpenAI, "m")); ch != nil {
					t.Fatalf("got a stream, want an error: %v", scopedDrainStream(ch))
				}
			} else {
				_, bifrostErr = client.ChatCompletionRequest(ctx, requestScopedChat(schemas.OpenAI, "m"))
			}
			if !strings.Contains(requestScopedErrorText(bifrostErr), "failed to get config for provider openai") {
				t.Fatalf("error = %v, want the provider lookup failure", bifrostErr)
			}
			if bifrostErr.ExtraFields.Provider != schemas.OpenAI {
				t.Errorf("error reports provider %q, want openai", bifrostErr.ExtraFields.Provider)
			}
			if pre, post := plugin.preLLMCalls.Load(), plugin.postLLMCalls.Load(); pre != 0 || post != 0 {
				t.Errorf("LLM hooks ran (%d pre, %d post), want none before the lookup failure", pre, post)
			}
			if got := calls(); len(got) != 0 {
				t.Errorf("upstream was reached: %+v", got)
			}
		})
	}
}

// shutdownOrderPlugin gives every request request-scoped configuration and records, in order,
// each PostLLMHook and Cleanup call. A PostLLMHook is recorded as it returns, after inPost.
type shutdownOrderPlugin struct {
	requestScopedTestPlugin
	inPost func()

	eventsMu sync.Mutex
	events   []string
}

func (p *shutdownOrderPlugin) record(event string) {
	p.eventsMu.Lock()
	p.events = append(p.events, event)
	p.eventsMu.Unlock()
}
func (p *shutdownOrderPlugin) Cleanup() error {
	p.record("cleanup")
	return nil
}
func (p *shutdownOrderPlugin) PostLLMHook(ctx *schemas.BifrostContext, resp *schemas.BifrostResponse, bifrostErr *schemas.BifrostError) (*schemas.BifrostResponse, *schemas.BifrostError, error) {
	if p.inPost != nil {
		p.inPost()
	}
	defer p.record("post")
	return p.requestScopedTestPlugin.PostLLMHook(ctx, resp, bifrostErr)
}

// assertNoHookAfterCleanup fails when Cleanup did not run or a PostLLMHook ran after it.
func (p *shutdownOrderPlugin) assertNoHookAfterCleanup(t *testing.T) {
	t.Helper()
	p.eventsMu.Lock()
	defer p.eventsMu.Unlock()
	cleanup := slices.Index(p.events, "cleanup")
	if cleanup < 0 {
		t.Fatalf("plugin Cleanup did not run: events %v", p.events)
	}
	if slices.Contains(p.events[cleanup:], "post") {
		t.Fatalf("PostLLMHook ran after plugin Cleanup: events %v", p.events)
	}
}

// gatedScopedServer is an OpenAI chat endpoint that holds every request while it is closed.
// arrived receives once per request as it comes in.
type gatedScopedServer struct {
	*httptest.Server
	arrived chan struct{}
	calls   atomic.Int64

	mu      sync.Mutex
	release chan struct{} // closed while the gate is open
}

func newGatedScopedServer(t *testing.T) *gatedScopedServer {
	t.Helper()
	s := &gatedScopedServer{arrived: make(chan struct{}, 16)}
	s.close()
	chatStream := sseHandler(`{"id":"c1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":"hello"},"finish_reason":null}]}`,
		`{"id":"c1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`)
	s.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		s.calls.Add(1)
		s.arrived <- struct{}{}
		s.mu.Lock()
		release := s.release
		s.mu.Unlock()
		<-release
		if strings.Contains(string(body), `"stream":true`) {
			chatStream(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"c1","object":"chat.completion","created":1,"model":"m",`+
			`"choices":[{"index":0,"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}],`+
			`"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
	}))
	// Without keep-alives no connection goroutine outlives a request.
	s.Config.SetKeepAlivesEnabled(false)
	// Registered after the server, so it runs first: Close waits for held handlers.
	t.Cleanup(s.Close)
	t.Cleanup(s.open)
	return s
}

// open lets every held and future request through.
func (s *gatedScopedServer) open() {
	s.mu.Lock()
	defer s.mu.Unlock()
	select {
	case <-s.release:
	default:
		close(s.release)
	}
}

// close holds every request that arrives from now on.
func (s *gatedScopedServer) close() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.release == nil {
		s.release = make(chan struct{})
		return
	}
	select {
	case <-s.release:
		s.release = make(chan struct{})
	default:
	}
}

// TestRequestScopedConfiguration_Shutdown pins that Shutdown drains request-scoped attempts
// before it cleans up plugins: an attempt in flight finishes, PostLLMHook included, an attempt
// that arrives once shutdown has begun fails with the shutdown error
// without reaching the provider, and a stream nobody reads does not hold Shutdown up.
func TestRequestScopedConfiguration_Shutdown(t *testing.T) {
	const shutdownErr = "provider is shutting down"
	tests := []struct {
		name     string
		upstream int64 // calls the provider must receive
		// handedBack is set when the attempt hands a stream back before Shutdown: as on the
		// queued path, its chunks run the post-hooks as they are read, so they may follow
		// Cleanup.
		handedBack bool
		run        func(t *testing.T, client *Bifrost, plugin *shutdownOrderPlugin, server *gatedScopedServer)
	}{
		{
			name:     "in-flight attempt finishes before cleanup",
			upstream: 1,
			run: func(t *testing.T, client *Bifrost, _ *shutdownOrderPlugin, server *gatedScopedServer) {
				result := make(chan *schemas.BifrostError, 1)
				go func() {
					_, err := client.ChatCompletionRequest(schemas.NewBifrostContext(context.Background(), schemas.NoDeadline), requestScopedChat(schemas.OpenAI, "m"))
					result <- err
				}()
				<-server.arrived
				shutdown := make(chan struct{})
				go func() {
					client.Shutdown()
					close(shutdown)
				}()
				select {
				case <-shutdown:
					t.Error("Shutdown returned while a request-scoped attempt was in flight")
				case <-time.After(100 * time.Millisecond):
				}
				server.open()
				if err := <-result; err != nil {
					t.Errorf("in-flight attempt: %s", err.GetErrorString())
				}
				<-shutdown
			},
		},
		{
			name: "attempt after shutdown is rejected",
			run: func(t *testing.T, client *Bifrost, _ *shutdownOrderPlugin, _ *gatedScopedServer) {
				client.Shutdown()
				ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
				if _, err := client.ChatCompletionRequest(ctx, requestScopedChat(schemas.OpenAI, "m")); !strings.Contains(requestScopedErrorText(err), shutdownErr) {
					t.Errorf("error = %q, want %q", requestScopedErrorText(err), shutdownErr)
				}
				stream, err := client.ChatCompletionStreamRequest(ctx, requestScopedChat(schemas.OpenAI, "m"))
				if stream != nil {
					t.Fatalf("got a stream, want an error: %v", scopedDrainStream(stream))
				}
				if !strings.Contains(requestScopedErrorText(err), shutdownErr) {
					t.Errorf("stream error = %q, want %q", requestScopedErrorText(err), shutdownErr)
				}
			},
		},
		{
			name:       "unread stream does not hold shutdown",
			upstream:   1,
			handedBack: true,
			run: func(t *testing.T, client *Bifrost, _ *shutdownOrderPlugin, server *gatedScopedServer) {
				server.open()
				stream, err := client.ChatCompletionStreamRequest(schemas.NewBifrostContext(context.Background(), schemas.NoDeadline), requestScopedChat(schemas.OpenAI, "m"))
				if err != nil {
					t.Fatalf("stream request: %s", err.GetErrorString())
				}
				shutdown := make(chan struct{})
				go func() {
					client.Shutdown()
					close(shutdown)
				}()
				select {
				case <-shutdown:
				case <-time.After(5 * time.Second):
					t.Error("Shutdown waited on a stream that was handed back")
				}
				scopedDrainStream(stream)
				<-shutdown
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := newGatedScopedServer(t)
			plugin := &shutdownOrderPlugin{}
			plugin.configure = func(_ *schemas.BifrostContext, req *schemas.BifrostRequest) {
				mustConfigureOpenAI(t, req, "sk-scoped", server.URL)
			}
			client, err := Init(context.Background(), schemas.BifrostConfig{
				Account:    NewMockAccount(),
				LLMPlugins: []schemas.LLMPlugin{plugin},
				Logger:     NewDefaultLogger(schemas.LogLevelError),
			})
			if err != nil {
				t.Fatalf("Init: %v", err)
			}
			tt.run(t, client, plugin, server)
			if !tt.handedBack {
				plugin.assertNoHookAfterCleanup(t)
			}
			if n := server.calls.Load(); n != tt.upstream {
				t.Errorf("upstream calls = %d, want %d", n, tt.upstream)
			}
		})
	}
}

// TestRequestScopedConfiguration_ShutdownNoGoroutineLeak pins that draining request-scoped
// attempts at shutdown leaves no goroutine behind: two attempts in flight both end with
// Shutdown.
func TestRequestScopedConfiguration_ShutdownNoGoroutineLeak(t *testing.T) {
	server := newGatedScopedServer(t)
	plugin := &shutdownOrderPlugin{}
	plugin.configure = func(_ *schemas.BifrostContext, req *schemas.BifrostRequest) {
		mustConfigureOpenAI(t, req, "sk-scoped", server.URL)
	}
	client, err := Init(context.Background(), schemas.BifrostConfig{
		Account:    NewMockAccount(),
		LLMPlugins: []schemas.LLMPlugin{plugin},
		Logger:     NewDefaultLogger(schemas.LogLevelError),
	})
	if err != nil {
		t.Fatalf("Init: %v", err)
	}
	// Warm up outside the measurement: the first request starts the HTTP client's own
	// long-lived goroutines.
	server.open()
	if _, err := client.ChatCompletionRequest(schemas.NewBifrostContext(context.Background(), schemas.NoDeadline), requestScopedChat(schemas.OpenAI, "m")); err != nil {
		t.Fatalf("warm-up: %s", err.GetErrorString())
	}
	<-server.arrived
	server.close()

	memtest.AssertNoGoroutineLeak(t, func() {
		results := make(chan *schemas.BifrostError, 2)
		request := func() {
			_, err := client.ChatCompletionRequest(schemas.NewBifrostContext(context.Background(), schemas.NoDeadline), requestScopedChat(schemas.OpenAI, "m"))
			results <- err
		}
		go request()
		go request()
		<-server.arrived
		<-server.arrived
		shutdown := make(chan struct{})
		go func() {
			client.Shutdown()
			close(shutdown)
		}()
		server.open()
		for range 2 {
			<-results
		}
		<-shutdown
	})
}

// TestRequestScopedConfiguration_ShutdownRunsPostHooks pins that an attempt whose PreLLMHook
// ran gets its PostLLMHook, with the shutdown error, before Shutdown cleans up plugins when
// shutdown begins before the attempt is admitted.
func TestRequestScopedConfiguration_ShutdownRunsPostHooks(t *testing.T) {
	const shutdownErr = "provider is shutting down"
	for _, stream := range []bool{false, true} {
		name := "admission"
		if stream {
			name += " stream"
		}
		t.Run(name, func(t *testing.T) {
			server := newGatedScopedServer(t)
			memtest.AssertNoGoroutineLeak(t, func() {
				plugin := &shutdownOrderPlugin{}
				plugin.configure = func(_ *schemas.BifrostContext, req *schemas.BifrostRequest) {
					mustConfigureOpenAI(t, req, "sk-scoped", server.URL)
				}
				client, err := Init(context.Background(), schemas.BifrostConfig{
					Account:    NewMockAccount(),
					LLMPlugins: []schemas.LLMPlugin{plugin},
					Logger:     NewDefaultLogger(schemas.LogLevelError),
				})
				if err != nil {
					t.Fatalf("Init: %v", err)
				}
				// Hold Shutdown after it begins and before it waits for request-scoped
				// attempts. The PostLLMHook lets it go on and then lingers, so a Cleanup
				// that does not wait for the hook lands before it in the events.
				client.oldWorkerCleanups.Add(1)
				var release sync.Once
				plugin.inPost = func() {
					release.Do(client.oldWorkerCleanups.Done)
					time.Sleep(50 * time.Millisecond)
				}
				shutdown := make(chan struct{})
				startShutdown := func() {
					go func() {
						client.Shutdown()
						close(shutdown)
					}()
					<-client.ctx.Done()
				}
				// Shutdown begins after the PreLLMHook and before the attempt is admitted.
				plugin.attempt = func(int, *schemas.BifrostRequest) { startShutdown() }
				result := make(chan *schemas.BifrostError, 1)
				go func() {
					ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
					var err *schemas.BifrostError
					if stream {
						var chunks chan *schemas.BifrostStreamChunk
						if chunks, err = client.ChatCompletionStreamRequest(ctx, requestScopedChat(schemas.OpenAI, "m")); chunks != nil {
							scopedDrainStream(chunks)
						}
					} else {
						_, err = client.ChatCompletionRequest(ctx, requestScopedChat(schemas.OpenAI, "m"))
					}
					result <- err
				}()
				if err := <-result; !strings.Contains(requestScopedErrorText(err), shutdownErr) {
					t.Errorf("error = %q, want %q", requestScopedErrorText(err), shutdownErr)
				}
				release.Do(client.oldWorkerCleanups.Done)
				<-shutdown
				if n := plugin.postLLMCalls.Load(); n != 1 {
					t.Errorf("PostLLMHook calls = %d, want 1", n)
				}
				if errs := plugin.observedErrors(); len(errs) != 1 || !strings.Contains(errs[0], shutdownErr) {
					t.Errorf("PostLLMHook errors = %q, want one %q", errs, shutdownErr)
				}
				plugin.assertNoHookAfterCleanup(t)
				if n := server.calls.Load(); n != 0 {
					t.Errorf("upstream calls = %d, want 0", n)
				}
			})
		})
	}
}

// TestRequestScopedConfiguration_BatchCreateSkipsProviderCheck pins BatchCreateRequest's
// provider check: it runs after PreRequestHook, so a batch create the hook configured per request
// runs on the request-scoped instance and initializes no provider, even for a dynamically
// configurable provider the check would otherwise initialize with default settings.
func TestRequestScopedConfiguration_BatchCreateSkipsProviderCheck(t *testing.T) {
	server, calls := requestScopedTestServer(t)
	plugin := &requestScopedTestPlugin{configure: func(_ *schemas.BifrostContext, req *schemas.BifrostRequest) {
		mustConfigureOpenAI(t, req, "sk-batch", server.URL)
	}}
	account := &countingAccount{MockAccount: NewMockAccount()}
	client := newRequestScopedTestClient(t, account, plugin, nil)

	resp, bifrostErr := client.BatchCreateRequest(schemas.NewBifrostContext(context.Background(), schemas.NoDeadline), &schemas.BifrostBatchCreateRequest{
		Provider:    schemas.OpenAI,
		InputFileID: "file-1",
		Endpoint:    schemas.BatchEndpointChatCompletions,
	})
	if bifrostErr != nil {
		t.Fatalf("BatchCreateRequest: %s", bifrostErr.GetErrorString())
	}
	if resp == nil || resp.ID != "batch_1" {
		t.Fatalf("response = %+v, want batch_1", resp)
	}
	if got := calls(); len(got) != 1 || got[0].path != "/v1/batches" || got[0].auth != "Bearer sk-batch" {
		t.Fatalf("upstream calls = %+v, want one POST /v1/batches with the request-scoped key", got)
	}
	if providers := client.providers.Load(); providers != nil && len(*providers) != 0 {
		t.Errorf("%d providers were registered", len(*providers))
	}
	if _, ok := client.requestQueues.Load(schemas.OpenAI); ok {
		t.Error("a provider queue was created for openai")
	}
	if n := account.configLookups.Load(); n != 0 {
		t.Errorf("the account was asked for provider config %d times", n)
	}
}

// TestRequestScopedConfiguration_BatchCreateChecksProvider pins that a batch create without
// request-scoped configuration keeps BatchCreateRequest's provider check: a provider that is
// neither configured nor dynamically configurable fails with the same error, after
// PreRequestHook and before the LLM hooks or any upstream call.
func TestRequestScopedConfiguration_BatchCreateChecksProvider(t *testing.T) {
	_, calls := requestScopedTestServer(t)
	plugin := &requestScopedTestPlugin{}
	client := newRequestScopedTestClient(t, NewMockAccount(), plugin, nil)

	_, bifrostErr := client.BatchCreateRequest(schemas.NewBifrostContext(context.Background(), schemas.NoDeadline), &schemas.BifrostBatchCreateRequest{
		Provider: schemas.Ollama,
		Requests: []schemas.BatchRequestItem{{CustomID: "1", Body: map[string]any{"model": "m"}}},
	})
	if bifrostErr == nil || bifrostErr.Error == nil || bifrostErr.Error.Message != errBatchCreateProviderNotFound {
		t.Fatalf("error = %v, want the batch create provider check", bifrostErr)
	}
	if n := plugin.preRequestCalls.Load(); n != 1 {
		t.Errorf("PreRequestHook ran %d times, want once before the check", n)
	}
	if pre, post := plugin.preLLMCalls.Load(), plugin.postLLMCalls.Load(); pre != 0 || post != 0 {
		t.Errorf("LLM hooks ran (%d pre, %d post), want none", pre, post)
	}
	if got := calls(); len(got) != 0 {
		t.Errorf("upstream was reached: %+v", got)
	}
	if providers := client.providers.Load(); providers != nil && len(*providers) != 0 {
		t.Errorf("%d providers were registered", len(*providers))
	}
}

// TestRequestScopedConfiguration_DirectKeyDoesNotOutliveTheAttempt pins that the attempt's
// key is on the context only while the attempt runs: afterwards the context holds what it
// held before, so a later request that shares the context never finds another request's key.
func TestRequestScopedConfiguration_DirectKeyDoesNotOutliveTheAttempt(t *testing.T) {
	server, _ := requestScopedTestServer(t)
	plugin := &requestScopedTestPlugin{configure: func(_ *schemas.BifrostContext, req *schemas.BifrostRequest) {
		mustConfigureOpenAI(t, req, "sk-scoped", server.URL)
	}}
	client := newRequestScopedTestClient(t, NewMockAccount(), plugin, nil)

	for _, stream := range []bool{false, true} {
		t.Run(fmt.Sprintf("stream=%v/no key before", stream), func(t *testing.T) {
			ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
			runRequestScopedChat(t, client, ctx, stream)
			if key, ok := ctx.Value(schemas.BifrostContextKeyDirectKey).(schemas.Key); ok {
				t.Fatalf("the attempt left key %q on the context", key.ID)
			}
		})
		t.Run(fmt.Sprintf("stream=%v/caller key before", stream), func(t *testing.T) {
			ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
			ctx.SetValue(schemas.BifrostContextKeyDirectKey, schemas.Key{ID: "caller", Value: *schemas.NewSecretVar("sk-caller")})
			runRequestScopedChat(t, client, ctx, stream)
			key, ok := ctx.Value(schemas.BifrostContextKeyDirectKey).(schemas.Key)
			if !ok || key.ID != "caller" || key.Value.GetValue() != "sk-caller" {
				t.Fatalf("context key after the attempt = %+v (present %v), want the caller's own key back", key, ok)
			}
		})
	}
}

func runRequestScopedChat(t *testing.T, client *Bifrost, ctx *schemas.BifrostContext, stream bool) {
	t.Helper()
	if stream {
		chunks, err := client.ChatCompletionStreamRequest(ctx, requestScopedChat(schemas.OpenAI, "m"))
		if err != nil {
			t.Fatalf("stream request: %s", err.GetErrorString())
		}
		scopedDrainStream(chunks)
		return
	}
	if _, err := client.ChatCompletionRequest(ctx, requestScopedChat(schemas.OpenAI, "m")); err != nil {
		t.Fatalf("request: %s", err.GetErrorString())
	}
}

// TestRequestScopedConfiguration_FallbackConfigurationIsItsOwn pins that what one fallback's
// PreLLMHook sets for its attempt does not reach the next fallback: each starts from the
// configuration the primary attempt ran with.
func TestRequestScopedConfiguration_FallbackConfigurationIsItsOwn(t *testing.T) {
	var mu sync.Mutex
	var seen []string
	failing := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		seen = append(seen, r.Header.Get("Authorization"))
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = fmt.Fprint(w, `{"error":{"message":"upstream failed","type":"server_error"}}`)
	}))
	t.Cleanup(failing.Close)

	plugin := &requestScopedTestPlugin{configure: func(_ *schemas.BifrostContext, req *schemas.BifrostRequest) {
		mustConfigureOpenAI(t, req, "sk-request", failing.URL)
	}, attempt: func(fallbackIndex int, req *schemas.BifrostRequest) {
		if fallbackIndex == 1 {
			mustConfigureOpenAI(t, req, "sk-first-fallback", failing.URL)
		}
	}}
	client := newRequestScopedTestClient(t, NewMockAccount(), plugin, nil)

	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	_, err := client.ChatCompletionRequest(ctx, requestScopedChat(schemas.OpenAI, "primary",
		schemas.Fallback{Provider: schemas.OpenAI, Model: "first"},
		schemas.Fallback{Provider: schemas.OpenAI, Model: "second"},
	))
	if err == nil {
		t.Fatal("every attempt failed upstream, want an error")
	}
	mu.Lock()
	defer mu.Unlock()
	want := []string{"Bearer sk-request", "Bearer sk-first-fallback", "Bearer sk-request"}
	if !slices.Equal(seen, want) {
		t.Fatalf("credentials sent = %q, want %q: the second fallback must not inherit the first fallback's key", seen, want)
	}
}

// TestRequestScopedConfiguration_ConfiguredRequestIsRefusedBeforeItsLLMHooksAfterShutdown pins
// that a request already carrying request-scoped configuration is admitted before its LLM
// hooks run: after Shutdown it is refused without running them.
func TestRequestScopedConfiguration_ConfiguredRequestIsRefusedBeforeItsLLMHooksAfterShutdown(t *testing.T) {
	server, calls := requestScopedTestServer(t)
	plugin := &requestScopedTestPlugin{configure: func(_ *schemas.BifrostContext, req *schemas.BifrostRequest) {
		mustConfigureOpenAI(t, req, "sk-scoped", server.URL)
	}}
	client, err := Init(context.Background(), schemas.BifrostConfig{
		Account:    NewMockAccount(),
		LLMPlugins: []schemas.LLMPlugin{plugin},
		Logger:     NewDefaultLogger(schemas.LogLevelError),
	})
	if err != nil {
		t.Fatalf("Init: %v", err)
	}
	client.Shutdown()

	for _, stream := range []bool{false, true} {
		ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
		var bifrostErr *schemas.BifrostError
		if stream {
			var chunks chan *schemas.BifrostStreamChunk
			if chunks, bifrostErr = client.ChatCompletionStreamRequest(ctx, requestScopedChat(schemas.OpenAI, "m")); chunks != nil {
				t.Fatalf("got a stream after shutdown: %v", scopedDrainStream(chunks))
			}
		} else {
			_, bifrostErr = client.ChatCompletionRequest(ctx, requestScopedChat(schemas.OpenAI, "m"))
		}
		if !strings.Contains(requestScopedErrorText(bifrostErr), "provider is shutting down") {
			t.Errorf("stream=%v: error = %q, want the shutdown refusal", stream, requestScopedErrorText(bifrostErr))
		}
	}
	if pre, post := plugin.preLLMCalls.Load(), plugin.postLLMCalls.Load(); pre != 0 || post != 0 {
		t.Errorf("LLM hooks ran after shutdown (%d pre, %d post), want none", pre, post)
	}
	if got := calls(); len(got) != 0 {
		t.Errorf("upstream calls after shutdown = %+v, want none", got)
	}
}

// countingAffinity leaves every chain as it is and counts how often it is consulted.
type countingAffinity struct {
	resolves, observes atomic.Int64
}

func (a *countingAffinity) ResolveRoute(_ *schemas.BifrostContext, _ schemas.Route, chain []schemas.Route) []schemas.Route {
	a.resolves.Add(1)
	return chain
}
func (a *countingAffinity) ResolveKey(*schemas.BifrostContext, schemas.ModelProvider, string, []schemas.Key) (schemas.Key, bool) {
	return schemas.Key{}, false
}
func (a *countingAffinity) Observe(*schemas.BifrostContext, schemas.Route, schemas.RouteOutcome) {
	a.observes.Add(1)
}

// TestRequestScopedConfiguration_FallbackConfiguredInItsHookIsNotRemembered pins that a
// session learns nothing from a request whose serving attempt got its own provider
// configuration, also when that attempt is a fallback configured only in its PreLLMHook.
func TestRequestScopedConfiguration_FallbackConfiguredInItsHookIsNotRemembered(t *testing.T) {
	failing := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = fmt.Fprint(w, `{"type":"error","error":{"type":"api_error","message":"upstream failed"}}`)
	}))
	t.Cleanup(failing.Close)
	server, calls := requestScopedTestServer(t)

	account := NewMockAccount()
	account.AddProviderWithBaseURL(schemas.Anthropic, 1, 1, failing.URL)
	account.SetKeysForProvider(schemas.Anthropic, []schemas.Key{{ID: "configured", Value: *schemas.NewSecretVar("sk-configured"), Models: schemas.WhiteList{"*"}, Weight: 1}})
	// The fallback's provider is configured too, or a request without request-scoped
	// configuration would skip the fallback before its PreLLMHook could configure it.
	account.AddProviderWithBaseURL(schemas.OpenAI, 1, 1, failing.URL)
	account.SetKeysForProvider(schemas.OpenAI, []schemas.Key{{ID: "configured-openai", Value: *schemas.NewSecretVar("sk-configured-openai"), Models: schemas.WhiteList{"*"}, Weight: 1}})
	affinity := &countingAffinity{}
	plugin := &requestScopedTestPlugin{attempt: func(fallbackIndex int, req *schemas.BifrostRequest) {
		if fallbackIndex == 1 {
			mustConfigureOpenAI(t, req, "sk-fallback", server.URL)
		}
	}}
	client, err := Init(context.Background(), schemas.BifrostConfig{
		Account:         account,
		LLMPlugins:      []schemas.LLMPlugin{plugin},
		SessionAffinity: affinity,
		Logger:          NewDefaultLogger(schemas.LogLevelError),
	})
	if err != nil {
		t.Fatalf("Init: %v", err)
	}
	defer client.Shutdown()

	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	ctx.SetValue(schemas.BifrostContextKeySessionID, "session-1")
	if _, bifrostErr := client.ChatCompletionRequest(ctx, requestScopedChat(schemas.Anthropic, "claude", schemas.Fallback{Provider: schemas.OpenAI, Model: "m"})); bifrostErr != nil {
		t.Fatalf("request: %s", bifrostErr.GetErrorString())
	}
	if got := calls(); len(got) != 1 || got[0].auth != "Bearer sk-fallback" {
		t.Fatalf("fallback calls = %+v, want one with the key its PreLLMHook set", got)
	}
	if n := affinity.observes.Load(); n != 0 {
		t.Fatalf("session affinity observed %d outcomes of a request served with request-scoped configuration", n)
	}
}

// TestRequestScopedConfiguration_AgentContinuationIsRefused pins that a later turn of an MCP
// agent loop is not run for a request whose first turn ran with request-scoped configuration:
// the continuation request carries none, and must not reach a configured provider instead.
func TestRequestScopedConfiguration_AgentContinuationIsRefused(t *testing.T) {
	server, calls := requestScopedTestServer(t)
	configure := true
	plugin := &requestScopedTestPlugin{configure: func(_ *schemas.BifrostContext, req *schemas.BifrostRequest) {
		if configure {
			mustConfigureOpenAI(t, req, "sk-scoped", server.URL)
		}
	}}
	account := NewMockAccount()
	account.AddProviderWithBaseURL(schemas.OpenAI, 1, 1, server.URL)
	account.SetKeysForProvider(schemas.OpenAI, []schemas.Key{{ID: "configured", Value: *schemas.NewSecretVar("sk-configured"), Models: schemas.WhiteList{"*"}, Weight: 1}})
	client := newRequestScopedTestClient(t, account, plugin, nil)

	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	if _, bifrostErr := client.ChatCompletionRequest(ctx, requestScopedChat(schemas.OpenAI, "m")); bifrostErr != nil {
		t.Fatalf("first turn: %s", bifrostErr.GetErrorString())
	}

	// The agent loop's next turn: a new request on the same context, pinned to the served route.
	configure = false
	ctx.SetValue(schemas.BifrostContextKeyMCPAgentServedRoute, &schemas.RoutingInfo{Provider: schemas.OpenAI, Model: "m"})
	_, bifrostErr := client.makeChatCompletionRequest(ctx, requestScopedChat(schemas.OpenAI, "m"))
	if !strings.Contains(requestScopedErrorText(bifrostErr), "MCP agent continuation is not supported") {
		t.Fatalf("continuation error = %q, want the refusal", requestScopedErrorText(bifrostErr))
	}
	if got := calls(); len(got) != 1 || got[0].auth != "Bearer sk-scoped" {
		t.Fatalf("upstream calls = %+v, want only the first turn with its own key", got)
	}
}

// TestRequestScopedConfiguration_RefusedAgentContinuationIsNotObservedByTheSession pins that
// the refused later turn of an agent loop, whose own request carries no configuration, tells
// the session nothing: the first turn ran with request-scoped configuration.
func TestRequestScopedConfiguration_RefusedAgentContinuationIsNotObservedByTheSession(t *testing.T) {
	server, _ := requestScopedTestServer(t)
	configure := true
	plugin := &requestScopedTestPlugin{configure: func(_ *schemas.BifrostContext, req *schemas.BifrostRequest) {
		if configure {
			mustConfigureOpenAI(t, req, "sk-scoped", server.URL)
		}
	}}
	affinity := &countingAffinity{}
	client, err := Init(context.Background(), schemas.BifrostConfig{
		Account:         NewMockAccount(),
		LLMPlugins:      []schemas.LLMPlugin{plugin},
		SessionAffinity: affinity,
		Logger:          NewDefaultLogger(schemas.LogLevelError),
	})
	if err != nil {
		t.Fatalf("Init: %v", err)
	}
	defer client.Shutdown()

	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	ctx.SetValue(schemas.BifrostContextKeySessionID, "session-1")
	if _, bifrostErr := client.ChatCompletionRequest(ctx, requestScopedChat(schemas.OpenAI, "m")); bifrostErr != nil {
		t.Fatalf("first turn: %s", bifrostErr.GetErrorString())
	}
	configure = false
	ctx.SetValue(schemas.BifrostContextKeyMCPAgentServedRoute, &schemas.RoutingInfo{Provider: schemas.OpenAI, Model: "m"})
	if _, bifrostErr := client.makeChatCompletionRequest(ctx, requestScopedChat(schemas.OpenAI, "m")); bifrostErr == nil {
		t.Fatal("the continuation ran, want the refusal")
	}
	if n := affinity.observes.Load(); n != 0 {
		t.Fatalf("session affinity observed %d outcomes across a request-scoped turn and its refused continuation", n)
	}
}
