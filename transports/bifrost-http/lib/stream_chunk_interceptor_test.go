package lib

import (
	"context"
	"errors"
	"testing"

	"github.com/maximhq/bifrost/core/schemas"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// testStreamPlugin is an HTTP transport plugin whose stream hook is a function.
type testStreamPlugin struct {
	name string
	hook func(chunk *schemas.BifrostStreamChunk) (*schemas.BifrostStreamChunk, error)
}

func (p *testStreamPlugin) GetName() string { return p.name }
func (p *testStreamPlugin) Cleanup() error  { return nil }
func (p *testStreamPlugin) HTTPTransportPreAuthHook(*schemas.BifrostContext, *schemas.HTTPRequest) (*schemas.HTTPResponse, error) {
	return nil, nil
}
func (p *testStreamPlugin) HTTPTransportPreHook(*schemas.BifrostContext, *schemas.HTTPRequest) (*schemas.HTTPResponse, error) {
	return nil, nil
}
func (p *testStreamPlugin) HTTPTransportPostHook(*schemas.BifrostContext, *schemas.HTTPRequest, *schemas.HTTPResponse) error {
	return nil
}
func (p *testStreamPlugin) HTTPTransportResponseHeadersHook(*schemas.BifrostContext, *schemas.HTTPRequest, *schemas.HTTPResponseMetadata) error {
	return nil
}
func (p *testStreamPlugin) HTTPTransportStreamChunkHook(_ *schemas.BifrostContext, _ *schemas.HTTPRequest, chunk *schemas.BifrostStreamChunk) (*schemas.BifrostStreamChunk, error) {
	return p.hook(chunk)
}

// testExpandingPlugin also implements the multi-chunk hook.
type testExpandingPlugin struct {
	testStreamPlugin
	expand func(chunk *schemas.BifrostStreamChunk) ([]*schemas.BifrostStreamChunk, error)
}

func (p *testExpandingPlugin) HTTPTransportStreamChunksHook(_ *schemas.BifrostContext, _ *schemas.HTTPRequest, chunk *schemas.BifrostStreamChunk) ([]*schemas.BifrostStreamChunk, error) {
	return p.expand(chunk)
}

func textChunk(text string) *schemas.BifrostStreamChunk {
	return &schemas.BifrostStreamChunk{BifrostResponsesStreamResponse: &schemas.BifrostResponsesStreamResponse{
		Type: schemas.ResponsesStreamResponseTypeOutputTextDelta, Delta: &text,
	}}
}

func chunkTexts(chunks []*schemas.BifrostStreamChunk) []string {
	texts := make([]string, 0, len(chunks))
	for _, chunk := range chunks {
		texts = append(texts, *chunk.BifrostResponsesStreamResponse.Delta)
	}
	return texts
}

func suffix(tag string) func(*schemas.BifrostStreamChunk) (*schemas.BifrostStreamChunk, error) {
	return func(chunk *schemas.BifrostStreamChunk) (*schemas.BifrostStreamChunk, error) {
		return textChunk(*chunk.BifrostResponsesStreamResponse.Delta + tag), nil
	}
}

func testContext() *schemas.BifrostContext {
	return schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
}

// Plugins run in reverse order. An expanding plugin's outputs each pass
// through the plugins that run after it, in order.
func TestInterceptChunk_ExpandedChunksContinueThroughLaterPlugins(t *testing.T) {
	expanding := &testExpandingPlugin{
		testStreamPlugin: testStreamPlugin{name: "expanding"},
		expand: func(chunk *schemas.BifrostStreamChunk) ([]*schemas.BifrostStreamChunk, error) {
			text := *chunk.BifrostResponsesStreamResponse.Delta
			return []*schemas.BifrostStreamChunk{textChunk(text + "-a"), nil, textChunk(text + "-b")}, nil
		},
	}
	later := &testStreamPlugin{name: "later", hook: suffix("!")}
	first := &testStreamPlugin{name: "first", hook: suffix("1")}
	// Reverse order: first runs, then expanding, then later.
	interceptor := &pluginChunkInterceptor{plugins: []schemas.HTTPTransportPlugin{later, expanding, first}}

	chunks, err := interceptor.InterceptChunk(testContext(), nil, textChunk("x"))
	require.NoError(t, err)
	assert.Equal(t, []string{"x1-a!", "x1-b!"}, chunkTexts(chunks))
}

func TestInterceptChunk_SingleChunkPluginsKeepTheirContract(t *testing.T) {
	skip := &testStreamPlugin{name: "skip", hook: func(*schemas.BifrostStreamChunk) (*schemas.BifrostStreamChunk, error) { return nil, nil }}
	chunks, err := (&pluginChunkInterceptor{plugins: []schemas.HTTPTransportPlugin{skip}}).InterceptChunk(testContext(), nil, textChunk("x"))
	require.NoError(t, err)
	assert.Empty(t, chunks, "a nil chunk skips the input")

	stop := &testStreamPlugin{name: "stop", hook: func(*schemas.BifrostStreamChunk) (*schemas.BifrostStreamChunk, error) {
		return nil, errors.New("blocked")
	}}
	chunks, err = (&pluginChunkInterceptor{plugins: []schemas.HTTPTransportPlugin{stop}}).InterceptChunk(testContext(), nil, textChunk("x"))
	assert.Error(t, err)
	assert.Empty(t, chunks, "an error without a chunk stops the stream")

	never := &testStreamPlugin{name: "never", hook: func(*schemas.BifrostStreamChunk) (*schemas.BifrostStreamChunk, error) {
		t.Fatal("a warning skips the remaining plugins")
		return nil, nil
	}}
	warn := &testStreamPlugin{name: "warn", hook: func(chunk *schemas.BifrostStreamChunk) (*schemas.BifrostStreamChunk, error) {
		return textChunk("kept"), errors.New("degraded")
	}}
	chunks, err = (&pluginChunkInterceptor{plugins: []schemas.HTTPTransportPlugin{never, warn}}).InterceptChunk(testContext(), nil, textChunk("x"))
	assert.Error(t, err)
	assert.Equal(t, []string{"kept"}, chunkTexts(chunks), "a warning keeps the modified chunk")
}

func TestInterceptChunk_ExpandingPluginCanHoldAndRelease(t *testing.T) {
	var held []*schemas.BifrostStreamChunk
	gate := &testExpandingPlugin{
		testStreamPlugin: testStreamPlugin{name: "gate"},
		expand: func(chunk *schemas.BifrostStreamChunk) ([]*schemas.BifrostStreamChunk, error) {
			held = append(held, chunk)
			if *chunk.BifrostResponsesStreamResponse.Delta != "end" {
				return nil, nil
			}
			released := held
			held = nil
			return released, nil
		},
	}
	interceptor := &pluginChunkInterceptor{plugins: []schemas.HTTPTransportPlugin{gate}}
	for _, text := range []string{"a", "b"} {
		chunks, err := interceptor.InterceptChunk(testContext(), nil, textChunk(text))
		require.NoError(t, err)
		assert.Empty(t, chunks)
	}
	chunks, err := interceptor.InterceptChunk(testContext(), nil, textChunk("end"))
	require.NoError(t, err)
	assert.Equal(t, []string{"a", "b", "end"}, chunkTexts(chunks))
}

// A warning on one expanded chunk ends that chunk's chain only; its siblings
// still pass through every remaining plugin.
func TestInterceptChunk_WarningOnOneExpandedChunkLeavesSiblingsProcessed(t *testing.T) {
	split := &testExpandingPlugin{
		testStreamPlugin: testStreamPlugin{name: "split"},
		expand: func(chunk *schemas.BifrostStreamChunk) ([]*schemas.BifrostStreamChunk, error) {
			return []*schemas.BifrostStreamChunk{textChunk("a"), textChunk("b")}, nil
		},
	}
	middle := &testStreamPlugin{name: "middle", hook: func(chunk *schemas.BifrostStreamChunk) (*schemas.BifrostStreamChunk, error) {
		if *chunk.BifrostResponsesStreamResponse.Delta == "a" {
			return textChunk("a-warned"), errors.New("degraded")
		}
		return textChunk("b-middle"), nil
	}}
	last := &testStreamPlugin{name: "last", hook: suffix("!")}
	// Reverse order: split, then middle, then last.
	interceptor := &pluginChunkInterceptor{plugins: []schemas.HTTPTransportPlugin{last, middle, split}}

	chunks, err := interceptor.InterceptChunk(testContext(), nil, textChunk("x"))
	assert.Error(t, err, "the warning is reported")
	assert.Equal(t, []string{"a-warned", "b-middle!"}, chunkTexts(chunks))
}
