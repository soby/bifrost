package handlers

import (
	"bufio"
	"context"
	"io"
	"net"
	"net/http"
	"runtime"
	"strings"
	"testing"

	"github.com/fasthttp/router"
	bifrost "github.com/maximhq/bifrost/core"
	"github.com/maximhq/bifrost/core/schemas"
	"github.com/maximhq/bifrost/framework/configstore"
	"github.com/maximhq/bifrost/transports/v2/bifrost-http/integrations"
	"github.com/maximhq/bifrost/transports/v2/bifrost-http/lib"
	"github.com/stretchr/testify/require"
	"github.com/valyala/fasthttp"
)

// Early SSE commitment (lib.StreamSetup): a plugin that holds a streaming request before
// the provider call signals schemas.StreamAdmissionWait. Without the signal the response
// stays uncommitted until setup ends, so a failure is an ordinary HTTP status. With it the
// transport commits a 200 event stream at once, the stream heartbeat keeps the connection
// alive, and a later failure is delivered as an error event.
//
// The tests gate setup on channels: a held setup is released only after the client has read
// a heartbeat comment, which proves the response was committed while setup was still running.

type earlyCommitSetup func(ctx *schemas.BifrostContext) (chan *schemas.BifrostStreamChunk, *schemas.BifrostError)

// earlyCommitServer serves handleStreamingResponse with the given setup on a local socket.
// setup runs as core would: it receives the request's BifrostContext, which carries the
// transport's wait signal.
func earlyCommitServer(t *testing.T, requestType schemas.RequestType, setup earlyCommitSetup) string {
	t.Helper()
	SetLogger(&mockLogger{})
	handler := func(reqCtx *fasthttp.RequestCtx) {
		h := &CompletionHandler{config: &lib.Config{}}
		bifrostCtx, cancel := schemas.NewBifrostContextWithCancel(t.Context())
		getStream := func() (chan *schemas.BifrostStreamChunk, *schemas.BifrostError) {
			return setup(bifrostCtx)
		}
		h.handleStreamingResponse(reqCtx, bifrostCtx, requestType, getStream, cancel)
	}
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	server := &fasthttp.Server{Handler: handler}
	go server.Serve(ln) //nolint:errcheck
	t.Cleanup(func() { _ = ln.Close() })
	return "http://" + ln.Addr().String() + "/v1/chat/completions"
}

func earlyCommitPost(t *testing.T, url string) *http.Response {
	t.Helper()
	resp, err := http.Post(url, "application/json", strings.NewReader(`{}`))
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	t.Cleanup(func() { _ = resp.Body.Close() })
	return resp
}

// readUntilHeartbeat reads the stream up to and including its first heartbeat comment and
// returns what preceded it.
func readUntilHeartbeat(t *testing.T, br *bufio.Reader) string {
	t.Helper()
	var before strings.Builder
	for {
		line, err := br.ReadString('\n')
		if err != nil {
			t.Fatalf("stream ended before a heartbeat: %v (read %q)", err, before.String())
		}
		if line == ": heartbeat\n" {
			return before.String()
		}
		before.WriteString(line)
	}
}

func readRest(t *testing.T, br *bufio.Reader) string {
	t.Helper()
	rest, err := io.ReadAll(br)
	if err != nil {
		t.Fatalf("read stream: %v", err)
	}
	return string(rest)
}

func earlyCommitReadyStream() chan *schemas.BifrostStreamChunk {
	stream := make(chan *schemas.BifrostStreamChunk, 1)
	terminal := contentChunk("hello")
	terminal.BifrostChatResponse.Choices[0].FinishReason = schemas.Ptr("stop")
	stream <- terminal
	close(stream)
	return stream
}

func earlyCommitUnavailableError() *schemas.BifrostError {
	status := fasthttp.StatusServiceUnavailable
	return &schemas.BifrostError{
		IsBifrostError: false,
		StatusCode:     &status,
		Error:          &schemas.ErrorField{Message: "lookup service unavailable", Code: schemas.Ptr("LOOKUP_UNAVAILABLE")},
	}
}

// heldSetup signals the wait, blocks until release is closed, then returns result.
func heldSetup(release <-chan struct{}, result func() (chan *schemas.BifrostStreamChunk, *schemas.BifrostError)) earlyCommitSetup {
	return func(ctx *schemas.BifrostContext) (chan *schemas.BifrostStreamChunk, *schemas.BifrostError) {
		schemas.StreamAdmissionWaitFrom(ctx).Begin()
		<-release
		return result()
	}
}

func TestEarlyCommitSetupErrorWithoutSignalIsHTTPStatus(t *testing.T) {
	release := make(chan struct{})
	url := earlyCommitServer(t, schemas.ChatCompletionStreamRequest, func(ctx *schemas.BifrostContext) (chan *schemas.BifrostStreamChunk, *schemas.BifrostError) {
		if schemas.StreamAdmissionWaitFrom(ctx) == nil {
			t.Error("the /v1 stream handler must offer the wait signal")
		}
		<-release
		return nil, earlyCommitUnavailableError()
	})
	// Nothing can reach the client while an unsignalled setup runs, so release it first.
	close(release)
	resp := earlyCommitPost(t, url)
	body := readRest(t, bufio.NewReader(resp.Body))
	if resp.StatusCode != fasthttp.StatusServiceUnavailable {
		t.Fatalf("status=%d, want the setup error's own status", resp.StatusCode)
	}
	if strings.Contains(resp.Header.Get("Content-Type"), "text/event-stream") || !strings.Contains(body, "LOOKUP_UNAVAILABLE") {
		t.Fatalf("want a JSON error body, got content-type=%q body=%q", resp.Header.Get("Content-Type"), body)
	}
}

func TestEarlyCommitKeepsTheConnectionAliveThenStreams(t *testing.T) {
	release := make(chan struct{})
	url := earlyCommitServer(t, schemas.ChatCompletionStreamRequest, heldSetup(release, func() (chan *schemas.BifrostStreamChunk, *schemas.BifrostError) {
		return earlyCommitReadyStream(), nil
	}))
	resp := earlyCommitPost(t, url)
	if resp.StatusCode != fasthttp.StatusOK || !strings.Contains(resp.Header.Get("Content-Type"), "text/event-stream") {
		t.Fatalf("status=%d content-type=%q, want a committed event stream", resp.StatusCode, resp.Header.Get("Content-Type"))
	}
	br := bufio.NewReader(resp.Body)
	if before := readUntilHeartbeat(t, br); before != "" {
		t.Fatalf("nothing but heartbeats may be sent while setup is held, got %q", before)
	}
	close(release)
	body := readRest(t, br)
	if !strings.Contains(body, `"hello"`) || !strings.HasSuffix(body, "data: [DONE]\n\n") {
		t.Fatalf("want the content then [DONE]:\n%s", body)
	}
}

func TestEarlyCommitErrorAfterCommitIsInStreamErrorWithoutDone(t *testing.T) {
	release := make(chan struct{})
	url := earlyCommitServer(t, schemas.ChatCompletionStreamRequest, heldSetup(release, func() (chan *schemas.BifrostStreamChunk, *schemas.BifrostError) {
		return nil, earlyCommitUnavailableError()
	}))
	resp := earlyCommitPost(t, url)
	if resp.StatusCode != fasthttp.StatusOK {
		t.Fatalf("status=%d, a committed stream keeps 200", resp.StatusCode)
	}
	br := bufio.NewReader(resp.Body)
	readUntilHeartbeat(t, br)
	close(release)
	body := readRest(t, br)
	if !strings.Contains(body, "LOOKUP_UNAVAILABLE") {
		t.Fatalf("want the error event:\n%s", body)
	}
	if strings.Contains(body, "[DONE]") {
		t.Fatalf("an error-terminated stream must not end with [DONE]:\n%s", body)
	}
}

func TestEarlyCommitResponsesErrorUsesErrorEvent(t *testing.T) {
	release := make(chan struct{})
	url := earlyCommitServer(t, schemas.ResponsesStreamRequest, heldSetup(release, func() (chan *schemas.BifrostStreamChunk, *schemas.BifrostError) {
		return nil, earlyCommitUnavailableError()
	}))
	br := bufio.NewReader(earlyCommitPost(t, url).Body)
	readUntilHeartbeat(t, br)
	close(release)
	if body := readRest(t, br); !strings.Contains(body, "event: error\n") {
		t.Fatalf("Responses streams carry errors as an error event:\n%s", body)
	}
}

func TestEarlyCommitSanitizesInternalErrors(t *testing.T) {
	release := make(chan struct{})
	url := earlyCommitServer(t, schemas.ChatCompletionStreamRequest, heldSetup(release, func() (chan *schemas.BifrostStreamChunk, *schemas.BifrostError) {
		return nil, &schemas.BifrostError{IsBifrostError: true, Error: &schemas.ErrorField{Message: "panic: secret detail at handler.go:42"}}
	}))
	br := bufio.NewReader(earlyCommitPost(t, url).Body)
	readUntilHeartbeat(t, br)
	close(release)
	body := readRest(t, br)
	if strings.Contains(body, "secret detail") || !strings.Contains(body, lib.ClientSafeInternalErrorMessage) {
		t.Fatalf("internal detail must be sanitized as on the pre-commit path:\n%s", body)
	}
}

func TestEarlyCommitSetupPanicAfterCommitIsInStreamError(t *testing.T) {
	release := make(chan struct{})
	url := earlyCommitServer(t, schemas.ChatCompletionStreamRequest, heldSetup(release, func() (chan *schemas.BifrostStreamChunk, *schemas.BifrostError) {
		panic("setup failed with secret detail")
	}))
	br := bufio.NewReader(earlyCommitPost(t, url).Body)
	readUntilHeartbeat(t, br)
	close(release)
	body := readRest(t, br)
	if strings.Contains(body, "secret detail") || !strings.Contains(body, lib.ClientSafeInternalErrorMessage) || strings.Contains(body, "[DONE]") {
		t.Fatalf("a setup panic after commitment ends the stream with a sanitized error event:\n%s", body)
	}
}

func TestEarlyCommitClientDisconnectCancelsTheWait(t *testing.T) {
	setupDone := make(chan struct{})
	url := earlyCommitServer(t, schemas.ChatCompletionStreamRequest, func(ctx *schemas.BifrostContext) (chan *schemas.BifrostStreamChunk, *schemas.BifrostError) {
		defer close(setupDone)
		schemas.StreamAdmissionWaitFrom(ctx).Begin()
		// A plugin's wait ends when the request context is cancelled.
		<-ctx.Done()
		return nil, &schemas.BifrostError{IsBifrostError: true, Error: &schemas.ErrorField{Message: "request cancelled"}}
	})
	resp := earlyCommitPost(t, url)
	readUntilHeartbeat(t, bufio.NewReader(resp.Body))
	_ = resp.Body.Close()
	// The heartbeat's next failed write cancels the request, which ends the wait.
	<-setupDone
}

func TestEarlyCommitPanicBeforeCommitReachesHandlerGoroutine(t *testing.T) {
	setup := lib.StartStreamSetup(schemas.NewBifrostContext(t.Context(), schemas.NoDeadline), true, func() (chan *schemas.BifrostStreamChunk, *schemas.BifrostError) {
		panic("setup failed")
	})
	defer func() {
		if recovered := recover(); recovered != "setup failed" {
			t.Fatalf("recovered %v, want the setup panic re-raised for the recovery middleware", recovered)
		}
	}()
	setup.Await()
	t.Fatal("Await must re-raise a setup panic")
}

func TestEarlyCommitIsNotOfferedWhenTheRouteCannotCarryComments(t *testing.T) {
	ctx := schemas.NewBifrostContext(t.Context(), schemas.NoDeadline)
	offered := true
	setup := lib.StartStreamSetup(ctx, false, func() (chan *schemas.BifrostStreamChunk, *schemas.BifrostError) {
		offered = schemas.StreamAdmissionWaitFrom(ctx) != nil
		return earlyCommitReadyStream(), nil
	})
	stream, bifrostErr, committed := setup.Await()
	if committed || bifrostErr != nil || stream == nil {
		t.Fatalf("Await() = (%v, %v, %v), want setup's own result", stream, bifrostErr, committed)
	}
	if offered {
		t.Fatal("no wait signal may be offered on such a route; the wait stays silent")
	}
}

// holdingPlugin is an LLM plugin that holds every request in PreLLMHook until release is
// closed, after telling the transport it is waiting.
type holdingPlugin struct {
	release chan struct{}
	offered chan bool
}

func (p *holdingPlugin) GetName() string { return "holding" }
func (p *holdingPlugin) Cleanup() error  { return nil }
func (p *holdingPlugin) PreRequestHook(*schemas.BifrostContext, *schemas.BifrostRequest) error {
	return nil
}

func (p *holdingPlugin) PreLLMHook(ctx *schemas.BifrostContext, req *schemas.BifrostRequest) (*schemas.BifrostRequest, *schemas.LLMPluginShortCircuit, error) {
	wait := schemas.StreamAdmissionWaitFrom(ctx)
	p.offered <- wait != nil
	wait.Begin()
	<-p.release
	return req, nil, nil
}

func (p *holdingPlugin) PostLLMHook(_ *schemas.BifrostContext, resp *schemas.BifrostResponse, bifrostErr *schemas.BifrostError) (*schemas.BifrostResponse, *schemas.BifrostError, error) {
	return resp, bifrostErr, nil
}

// TestEarlyCommitPluginHoldsStreamBeforeProviderCall drives a streamed chat completion
// through the served routes with a real Bifrost client whose plugin holds the request in
// PreLLMHook: the client receives the committed stream and a heartbeat while the request
// is still held, then the provider's stream once it is released.
func TestEarlyCommitPluginHoldsStreamBeforeProviderCall(t *testing.T) {
	SetLogger(&mockLogger{})
	for _, path := range []string{"/v1/chat/completions", "/openai/v1/chat/completions"} {
		t.Run(path, func(t *testing.T) {
			upstream := sharedContextUpstream(t)
			plugin := &holdingPlugin{release: make(chan struct{}), offered: make(chan bool, 1)}
			client, err := bifrost.Init(context.Background(), schemas.BifrostConfig{
				Account:    sharedContextAccount{baseURL: upstream.URL},
				LLMPlugins: []schemas.LLMPlugin{plugin},
				Logger:     bifrost.NewDefaultLogger(schemas.LogLevelError),
			})
			require.NoError(t, err)
			defer client.Shutdown()

			config := &lib.Config{ClientConfig: &configstore.ClientConfig{}}
			r := router.New()
			r.POST("/v1/chat/completions", NewInferenceHandler(nil, client, config).chatCompletion)
			integrations.NewOpenAIRouter(client, config, nil, &mockLogger{}).RegisterRoutes(r)
			ln, err := net.Listen("tcp", "127.0.0.1:0")
			require.NoError(t, err)
			server := &fasthttp.Server{Handler: r.Handler}
			go server.Serve(ln) //nolint:errcheck
			defer ln.Close()    //nolint:errcheck

			resp, err := http.Post("http://"+ln.Addr().String()+path, "application/json",
				strings.NewReader(`{"model":"openai/gpt-4o","stream":true,"messages":[{"role":"user","content":"hello"}]}`))
			require.NoError(t, err)
			defer resp.Body.Close()
			require.Equal(t, http.StatusOK, resp.StatusCode)
			require.Contains(t, resp.Header.Get("Content-Type"), "text/event-stream")
			require.True(t, <-plugin.offered, "the plugin finds the wait signal on its hook context")
			br := bufio.NewReader(resp.Body)
			require.Empty(t, readUntilHeartbeat(t, br), "nothing but heartbeats is sent while the request is held")
			close(plugin.release)
			body := readRest(t, br)
			require.Contains(t, body, `"content":"hi"`)
			require.True(t, strings.HasSuffix(body, "data: [DONE]\n\n"), body)
		})
	}
}

// loggingHoldPlugin writes plugin logs from PreLLMHook before and while it holds the request,
// and from its transport hooks.
type loggingHoldPlugin struct {
	release chan struct{}
	held    chan struct{}
}

func (p *loggingHoldPlugin) GetName() string { return "logging-hold" }
func (p *loggingHoldPlugin) Cleanup() error  { return nil }
func (p *loggingHoldPlugin) PreRequestHook(*schemas.BifrostContext, *schemas.BifrostRequest) error {
	return nil
}

func (p *loggingHoldPlugin) PreLLMHook(ctx *schemas.BifrostContext, req *schemas.BifrostRequest) (*schemas.BifrostRequest, *schemas.LLMPluginShortCircuit, error) {
	ctx.Log(schemas.LogLevelInfo, "llm: before the hold")
	schemas.StreamAdmissionWaitFrom(ctx).Begin()
	close(p.held)
	for {
		select {
		case <-p.release:
			ctx.Log(schemas.LogLevelInfo, "llm: released")
			return req, nil, nil
		default:
			// Keep logging for as long as the request is held, while the handler
			// goroutine commits the stream and runs the transport hooks.
			ctx.Log(schemas.LogLevelInfo, "llm: holding")
			runtime.Gosched()
		}
	}
}

func (p *loggingHoldPlugin) PostLLMHook(ctx *schemas.BifrostContext, resp *schemas.BifrostResponse, bifrostErr *schemas.BifrostError) (*schemas.BifrostResponse, *schemas.BifrostError, error) {
	ctx.Log(schemas.LogLevelInfo, "llm: post")
	return resp, bifrostErr, nil
}

func (p *loggingHoldPlugin) HTTPTransportPreAuthHook(*schemas.BifrostContext, *schemas.HTTPRequest) (*schemas.HTTPResponse, error) {
	return nil, nil
}

func (p *loggingHoldPlugin) HTTPTransportPreHook(ctx *schemas.BifrostContext, _ *schemas.HTTPRequest) (*schemas.HTTPResponse, error) {
	ctx.Log(schemas.LogLevelInfo, "transport: pre")
	return nil, nil
}

func (p *loggingHoldPlugin) HTTPTransportResponseHeadersHook(ctx *schemas.BifrostContext, _ *schemas.HTTPRequest, _ *schemas.HTTPResponseMetadata) error {
	ctx.Log(schemas.LogLevelInfo, "transport: response headers")
	return nil
}

func (p *loggingHoldPlugin) HTTPTransportStreamChunkHook(_ *schemas.BifrostContext, _ *schemas.HTTPRequest, chunk *schemas.BifrostStreamChunk) (*schemas.BifrostStreamChunk, error) {
	return chunk, nil
}

func (p *loggingHoldPlugin) HTTPTransportPostHook(ctx *schemas.BifrostContext, _ *schemas.HTTPRequest, _ *schemas.HTTPResponse) error {
	ctx.Log(schemas.LogLevelInfo, "transport: post")
	return nil
}

// TestEarlyCommitTransportHooksDoNotShareThePluginLogStore commits a stream while a plugin is
// still inside PreLLMHook writing plugin logs. The handler goroutine runs the transport
// response-headers hook at that moment, and both sides log. The transport hooks must log to
// a store of their own: core drains and recycles the handler context's store from its own
// goroutines, so a transport-side drain of it would take the LLM hook's logs and hand the
// store back for reuse while the hook still writes to it.
func TestEarlyCommitTransportHooksDoNotShareThePluginLogStore(t *testing.T) {
	SetLogger(&mockLogger{})
	for _, path := range []string{"/v1/chat/completions", "/openai/v1/chat/completions"} {
		t.Run(path, func(t *testing.T) {
			upstream := sharedContextUpstream(t)
			plugin := &loggingHoldPlugin{release: make(chan struct{}), held: make(chan struct{})}
			client, err := bifrost.Init(context.Background(), schemas.BifrostConfig{
				Account:    sharedContextAccount{baseURL: upstream.URL},
				LLMPlugins: []schemas.LLMPlugin{plugin},
				Logger:     bifrost.NewDefaultLogger(schemas.LogLevelError),
			})
			require.NoError(t, err)
			defer client.Shutdown()

			config := responseHeadersTestConfig(t, plugin)
			config.ClientConfig = &configstore.ClientConfig{}
			// The trace completer receives the transport plugin logs when the stream is torn down.
			transportLogs := make(chan []schemas.PluginLogEntry, 1)
			tracing := func(next fasthttp.RequestHandler) fasthttp.RequestHandler {
				return func(ctx *fasthttp.RequestCtx) {
					ctx.SetUserValue(schemas.BifrostContextKeyTraceCompleter, func(logs []schemas.PluginLogEntry) { transportLogs <- logs })
					next(ctx)
				}
			}
			transport := TransportInterceptorMiddleware(config)
			r := router.New()
			r.POST("/v1/chat/completions", tracing(transport(NewInferenceHandler(nil, client, config).chatCompletion)))
			integrations.NewOpenAIRouter(client, config, nil, &mockLogger{}).RegisterRoutes(r, tracing, transport)
			ln, err := net.Listen("tcp", "127.0.0.1:0")
			require.NoError(t, err)
			server := &fasthttp.Server{Handler: r.Handler}
			go server.Serve(ln) //nolint:errcheck
			defer ln.Close()    //nolint:errcheck

			resp, err := http.Post("http://"+ln.Addr().String()+path, "application/json",
				strings.NewReader(`{"model":"openai/gpt-4o","stream":true,"messages":[{"role":"user","content":"hello"}]}`))
			require.NoError(t, err)
			defer resp.Body.Close()
			require.Equal(t, http.StatusOK, resp.StatusCode)
			<-plugin.held
			// The response headers are on the wire: the handler goroutine has run the
			// response-headers hook and returned while PreLLMHook is still logging.
			br := bufio.NewReader(resp.Body)
			readUntilHeartbeat(t, br)
			close(plugin.release)
			body := readRest(t, br)
			require.Contains(t, body, `"content":"hi"`)

			var messages []string
			for _, entry := range <-transportLogs {
				messages = append(messages, entry.Message)
			}
			require.Equal(t, []string{"transport: pre", "transport: response headers", "transport: post"}, messages,
				"the transport hooks' logs, all of them and nothing an LLM hook wrote")
		})
	}
}
