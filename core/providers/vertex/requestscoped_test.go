package vertex

import (
	"context"
	"crypto/tls"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	schemas "github.com/maximhq/bifrost/core/schemas"
)

// requestScopedTestLogger discards everything the provider logs.
type requestScopedTestLogger struct{}

func (requestScopedTestLogger) Debug(string, ...any)                   {}
func (requestScopedTestLogger) Info(string, ...any)                    {}
func (requestScopedTestLogger) Warn(string, ...any)                    {}
func (requestScopedTestLogger) Error(string, ...any)                   {}
func (requestScopedTestLogger) Fatal(string, ...any)                   {}
func (requestScopedTestLogger) SetLevel(schemas.LogLevel)              {}
func (requestScopedTestLogger) SetOutputType(schemas.LoggerOutputType) {}
func (requestScopedTestLogger) LogHTTPRequest(schemas.LogLevel, string) schemas.LogEventBuilder {
	return schemas.NoopLogEvent
}

// requestScopedVertexCall is one request the fake Google endpoint received.
type requestScopedVertexCall struct {
	host, path, auth, query string
}

// newRequestScopedVertexProvider builds a Vertex provider the way Bifrost builds a
// request-scoped instance, runs ForRequest with key, and points both HTTP clients at a fake
// TLS endpoint that answers every call with a 401, so each operation stops after its first
// upstream call. It returns the provider and a function reporting the calls received.
func newRequestScopedVertexProvider(t *testing.T, key schemas.Key) (*VertexProvider, func() []requestScopedVertexCall) {
	t.Helper()
	var mu sync.Mutex
	var calls []requestScopedVertexCall
	server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		calls = append(calls, requestScopedVertexCall{host: r.Host, path: r.URL.Path, auth: r.Header.Get("Authorization"), query: r.URL.RawQuery})
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte(`{"error":{"code":401,"message":"fake endpoint","status":"UNAUTHENTICATED"}}`))
	}))
	t.Cleanup(server.Close)

	config := &schemas.ProviderConfig{NetworkConfig: schemas.DefaultNetworkConfig}
	config.NetworkConfig.DefaultRequestTimeoutInSeconds = 10
	config.NetworkConfig.LoopbackIsPrivate = true
	config.CheckAndSetDefaults()
	instance, err := NewVertexProvider(config, requestScopedTestLogger{})
	if err != nil {
		t.Fatalf("NewVertexProvider: %v", err)
	}
	scoped, err := instance.ForRequest(schemas.ChatCompletionRequest, key, "")
	if err != nil {
		t.Fatalf("ForRequest: %v", err)
	}
	provider := scoped.(*VertexProvider)
	if provider != instance {
		t.Fatal("ForRequest must return the receiver: VertexProvider holds a sync.Map and is never copied")
	}

	// Every Google host resolves to the fake endpoint.
	addr := server.Listener.Addr().String()
	dial := func(string) (net.Conn, error) { return net.Dial("tcp", addr) }
	provider.client.Dial = dial
	provider.client.TLSConfig = &tls.Config{InsecureSkipVerify: true} //nolint:gosec // test endpoint
	provider.streamingClient.Dial = dial
	provider.streamingClient.TLSConfig = &tls.Config{InsecureSkipVerify: true} //nolint:gosec // test endpoint

	return provider, func() []requestScopedVertexCall {
		mu.Lock()
		defer mu.Unlock()
		return append([]requestScopedVertexCall(nil), calls...)
	}
}

// requestScopedVertexKey is a request-scoped Vertex key carrying a caller-supplied token.
func requestScopedVertexKey(token string) schemas.Key {
	return schemas.Key{VertexKeyConfig: &schemas.VertexKeyConfig{
		ProjectID:   *schemas.NewSecretVar("my-project"),
		Region:      *schemas.NewSecretVar("us-central1"),
		AccessToken: token,
	}}
}

// TestRequestScopedVertex_EveryOperationUsesCallerToken pins that every Vertex operation a
// request-scoped key can reach authenticates with the caller-supplied access token: Gemini and
// Claude chat (unary and streaming), Responses, embeddings, rerank, image generation, count
// tokens, list models, cached contents, files, batches, video and passthrough. Application
// Default Credentials point at a missing file, so an operation that consulted them would fail
// before calling the endpoint, and no token source may be cached.
func TestRequestScopedVertex_EveryOperationUsesCallerToken(t *testing.T) {
	t.Setenv("GOOGLE_APPLICATION_CREDENTIALS", "/nonexistent/bifrost-test-adc.json")
	const token = "ya29.caller-minted"
	key := requestScopedVertexKey(token)
	keys := []schemas.Key{key}
	text := "hello"
	model := "gemini-2.5-flash"
	claude := "claude-sonnet-4@20250514"
	chat := func(m string) *schemas.BifrostChatRequest {
		return &schemas.BifrostChatRequest{Provider: schemas.Vertex, Model: m, Input: []schemas.ChatMessage{{
			Role: schemas.ChatMessageRoleUser, Content: &schemas.ChatMessageContent{ContentStr: &text},
		}}}
	}
	responses := func(m string) *schemas.BifrostResponsesRequest {
		role := schemas.ResponsesInputMessageRoleUser
		return &schemas.BifrostResponsesRequest{Provider: schemas.Vertex, Model: m, Input: []schemas.ResponsesMessage{{
			Role: &role, Content: &schemas.ResponsesMessageContent{ContentStr: &text},
		}}}
	}
	gcs := &schemas.FileStorageConfig{GCS: &schemas.GCSStorageConfig{Bucket: "my-bucket"}}
	noopPostHook := func(_ *schemas.BifrostContext, r *schemas.BifrostResponse, e *schemas.BifrostError) (*schemas.BifrostResponse, *schemas.BifrostError) {
		return r, e
	}
	drain := func(ch chan *schemas.BifrostStreamChunk, err *schemas.BifrostError) *schemas.BifrostError {
		if ch == nil {
			return err
		}
		for range ch {
		}
		return err
	}

	ops := []struct {
		name string
		run  func(p *VertexProvider, ctx *schemas.BifrostContext) *schemas.BifrostError
	}{
		{"chat gemini", func(p *VertexProvider, ctx *schemas.BifrostContext) *schemas.BifrostError {
			_, err := p.ChatCompletion(ctx, key, chat(model))
			return err
		}},
		{"chat claude", func(p *VertexProvider, ctx *schemas.BifrostContext) *schemas.BifrostError {
			_, err := p.ChatCompletion(ctx, key, chat(claude))
			return err
		}},
		{"chat stream gemini", func(p *VertexProvider, ctx *schemas.BifrostContext) *schemas.BifrostError {
			return drain(p.ChatCompletionStream(ctx, noopPostHook, nil, key, chat(model)))
		}},
		{"chat stream claude", func(p *VertexProvider, ctx *schemas.BifrostContext) *schemas.BifrostError {
			return drain(p.ChatCompletionStream(ctx, noopPostHook, nil, key, chat(claude)))
		}},
		{"responses gemini", func(p *VertexProvider, ctx *schemas.BifrostContext) *schemas.BifrostError {
			_, err := p.Responses(ctx, key, responses(model))
			return err
		}},
		{"responses claude", func(p *VertexProvider, ctx *schemas.BifrostContext) *schemas.BifrostError {
			_, err := p.Responses(ctx, key, responses(claude))
			return err
		}},
		{"responses stream claude", func(p *VertexProvider, ctx *schemas.BifrostContext) *schemas.BifrostError {
			return drain(p.ResponsesStream(ctx, noopPostHook, nil, key, responses(claude)))
		}},
		{"count tokens", func(p *VertexProvider, ctx *schemas.BifrostContext) *schemas.BifrostError {
			_, err := p.CountTokens(ctx, key, responses(model))
			return err
		}},
		{"embedding", func(p *VertexProvider, ctx *schemas.BifrostContext) *schemas.BifrostError {
			_, err := p.Embedding(ctx, key, &schemas.BifrostEmbeddingRequest{Provider: schemas.Vertex, Model: "text-embedding-005", Input: []schemas.EmbeddingInputItem{{
				Content: schemas.EmbeddingContent{{Type: schemas.EmbeddingContentPartTypeText, Text: &text}},
			}}})
			return err
		}},
		{"rerank", func(p *VertexProvider, ctx *schemas.BifrostContext) *schemas.BifrostError {
			_, err := p.Rerank(ctx, key, &schemas.BifrostRerankRequest{Provider: schemas.Vertex, Model: "semantic-ranker-default-004", Query: "q", Documents: []schemas.RerankDocument{{Text: "d"}}})
			return err
		}},
		{"image generation", func(p *VertexProvider, ctx *schemas.BifrostContext) *schemas.BifrostError {
			_, err := p.ImageGeneration(ctx, key, &schemas.BifrostImageGenerationRequest{Provider: schemas.Vertex, Model: "imagen-4.0-generate-001", Input: &schemas.ImageGenerationInput{Prompt: "p"}})
			return err
		}},
		{"video generation", func(p *VertexProvider, ctx *schemas.BifrostContext) *schemas.BifrostError {
			_, err := p.VideoGeneration(ctx, key, &schemas.BifrostVideoGenerationRequest{Provider: schemas.Vertex, Model: "veo-3.0-generate-001", Input: &schemas.VideoGenerationInput{Prompt: "p"}})
			return err
		}},
		{"list models", func(p *VertexProvider, ctx *schemas.BifrostContext) *schemas.BifrostError {
			_, err := p.ListModels(ctx, keys, &schemas.BifrostListModelsRequest{Provider: schemas.Vertex, Unfiltered: true})
			return err
		}},
		{"cached content create", func(p *VertexProvider, ctx *schemas.BifrostContext) *schemas.BifrostError {
			_, err := p.CachedContentCreate(ctx, key, &schemas.BifrostCachedContentCreateRequest{Provider: schemas.Vertex, Model: model})
			return err
		}},
		{"cached content list", func(p *VertexProvider, ctx *schemas.BifrostContext) *schemas.BifrostError {
			_, err := p.CachedContentList(ctx, keys, &schemas.BifrostCachedContentListRequest{Provider: schemas.Vertex})
			return err
		}},
		{"file upload", func(p *VertexProvider, ctx *schemas.BifrostContext) *schemas.BifrostError {
			_, err := p.FileUpload(ctx, key, &schemas.BifrostFileUploadRequest{Provider: schemas.Vertex, File: []byte("{}\n"), Filename: "in.jsonl", Purpose: "batch", StorageConfig: gcs})
			return err
		}},
		{"file list", func(p *VertexProvider, ctx *schemas.BifrostContext) *schemas.BifrostError {
			_, err := p.FileList(ctx, keys, &schemas.BifrostFileListRequest{Provider: schemas.Vertex, StorageConfig: gcs})
			return err
		}},
		{"file retrieve", func(p *VertexProvider, ctx *schemas.BifrostContext) *schemas.BifrostError {
			_, err := p.FileRetrieve(ctx, keys, &schemas.BifrostFileRetrieveRequest{Provider: schemas.Vertex, FileID: "gs://my-bucket/in.jsonl"})
			return err
		}},
		{"file content", func(p *VertexProvider, ctx *schemas.BifrostContext) *schemas.BifrostError {
			_, err := p.FileContent(ctx, keys, &schemas.BifrostFileContentRequest{Provider: schemas.Vertex, FileID: "gs://my-bucket/in.jsonl"})
			return err
		}},
		{"file delete", func(p *VertexProvider, ctx *schemas.BifrostContext) *schemas.BifrostError {
			_, err := p.FileDelete(ctx, keys, &schemas.BifrostFileDeleteRequest{Provider: schemas.Vertex, FileID: "gs://my-bucket/in.jsonl"})
			return err
		}},
		{"batch create", func(p *VertexProvider, ctx *schemas.BifrostContext) *schemas.BifrostError {
			_, err := p.BatchCreate(ctx, key, &schemas.BifrostBatchCreateRequest{Provider: schemas.Vertex, Model: &model, InputFileID: "gs://my-bucket/in.jsonl",
				OutputFolder: &schemas.BatchOutputFolder{URL: "gs://my-bucket/out/"}})
			return err
		}},
		{"batch list", func(p *VertexProvider, ctx *schemas.BifrostContext) *schemas.BifrostError {
			_, err := p.BatchList(ctx, keys, &schemas.BifrostBatchListRequest{Provider: schemas.Vertex})
			return err
		}},
		{"batch retrieve", func(p *VertexProvider, ctx *schemas.BifrostContext) *schemas.BifrostError {
			_, err := p.BatchRetrieve(ctx, keys, &schemas.BifrostBatchRetrieveRequest{Provider: schemas.Vertex, BatchID: "123"})
			return err
		}},
		{"batch cancel", func(p *VertexProvider, ctx *schemas.BifrostContext) *schemas.BifrostError {
			_, err := p.BatchCancel(ctx, keys, &schemas.BifrostBatchCancelRequest{Provider: schemas.Vertex, BatchID: "123"})
			return err
		}},
		{"batch delete", func(p *VertexProvider, ctx *schemas.BifrostContext) *schemas.BifrostError {
			_, err := p.BatchDelete(ctx, keys, &schemas.BifrostBatchDeleteRequest{Provider: schemas.Vertex, BatchID: "123"})
			return err
		}},
		{"batch results", func(p *VertexProvider, ctx *schemas.BifrostContext) *schemas.BifrostError {
			_, err := p.BatchResults(ctx, keys, &schemas.BifrostBatchResultsRequest{Provider: schemas.Vertex, BatchID: "123"})
			return err
		}},
		{"passthrough", func(p *VertexProvider, ctx *schemas.BifrostContext) *schemas.BifrostError {
			_, err := p.Passthrough(ctx, key, &schemas.BifrostPassthroughRequest{Provider: schemas.Vertex, Model: model, Method: http.MethodPost,
				Path: "/v1/projects/my-project/locations/us-central1/publishers/google/models/" + model + ":generateContent", Body: []byte(`{}`)})
			return err
		}},
	}

	for _, op := range ops {
		t.Run(op.name, func(t *testing.T) {
			provider, calls := newRequestScopedVertexProvider(t, key)
			ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
			bifrostErr := op.run(provider, ctx)
			got := calls()
			if len(got) == 0 {
				t.Fatalf("no upstream call was made (error: %s)", errorText(bifrostErr))
			}
			for _, call := range got {
				if call.auth != "Bearer "+token {
					t.Errorf("%s%s carried Authorization %q, want the caller's token", call.host, call.path, call.auth)
				}
				if strings.Contains(call.query, "key=") {
					t.Errorf("%s%s carried an API key query: %q", call.host, call.path, call.query)
				}
				if !strings.HasSuffix(strings.Split(call.host, ":")[0], "googleapis.com") {
					t.Errorf("call went to %q, want a googleapis.com host", call.host)
				}
			}
			cached := 0
			provider.tokenSources.Range(func(_, _ any) bool { cached++; return true })
			if cached != 0 {
				t.Errorf("%d token sources were cached for a caller-supplied token", cached)
			}
		})
	}
}

// errorText renders a BifrostError for a test failure message.
func errorText(err *schemas.BifrostError) string {
	if err == nil {
		return "<nil>"
	}
	return err.GetErrorString()
}

// TestRequestScopedVertex_TokenBypassesTokenSourceCache pins that getAuthTokenSource returns
// the caller's token as given for any key carrying one, ignoring service-account JSON on the
// same key, and that evicting such a key leaves the provider's cached sources alone.
func TestRequestScopedVertex_TokenBypassesTokenSourceCache(t *testing.T) {
	t.Setenv("GOOGLE_APPLICATION_CREDENTIALS", "/nonexistent/bifrost-test-adc.json")
	provider := &VertexProvider{}
	key := requestScopedVertexKey("ya29.direct")
	key.VertexKeyConfig.AuthCredentials = *schemas.NewSecretVar(`{"type":"service_account"}`)

	source, err := provider.getAuthTokenSource(key)
	if err != nil {
		t.Fatalf("getAuthTokenSource: %v", err)
	}
	tok, err := source.Token()
	if err != nil || tok.AccessToken != "ya29.direct" || tok.Type() != "Bearer" {
		t.Fatalf("token = %+v, %v; want the caller's bearer token", tok, err)
	}

	provider.tokenSources.Store(defaultCredentialsCacheKey, source)
	provider.removeVertexClient(key)
	if _, ok := provider.tokenSources.Load(defaultCredentialsCacheKey); !ok {
		t.Error("removeVertexClient for a caller-token key evicted another credential's source")
	}

	// Without a token the same key goes through its configured credential, which is cached.
	key.VertexKeyConfig.AccessToken = ""
	if _, err := provider.getAuthTokenSource(key); err != nil {
		t.Fatalf("getAuthTokenSource without a token: %v", err)
	}
	if _, ok := provider.tokenSources.Load(getClientKey(key.VertexKeyConfig.AuthCredentials.GetValue())); !ok {
		t.Error("a key without a caller token did not take the cached credential path")
	}
}
