package schemas

import (
	"context"
	"testing"
)

func TestWriteStreamChunkWithoutAWriterReportsFalse(t *testing.T) {
	chunk := &BifrostStreamChunk{}
	if WriteStreamChunk(nil, chunk) { //nolint:staticcheck // nil context is the case under test
		t.Fatal("a nil context has no writer")
	}
	ctx := NewBifrostContext(context.Background(), NoDeadline)
	if WriteStreamChunk(ctx, chunk) {
		t.Fatal("a request without a writer writes nothing")
	}
	ctx.SetValue(BifrostContextKeyStreamChunkWriter, "not a writer")
	if WriteStreamChunk(ctx, chunk) {
		t.Fatal("a value of another type is not a writer")
	}
}

func TestWriteStreamChunkCallsTheWriterOfTheCallsContext(t *testing.T) {
	ctx := NewBifrostContext(context.Background(), NoDeadline)
	var got []*BifrostStreamChunk
	write := StreamChunkWriter(func(chunk *BifrostStreamChunk) bool {
		got = append(got, chunk)
		return len(got) == 1
	})

	// The writer is handed to one hook call's context and to nothing else.
	name := "plugin"
	scoped := ctx.WithPluginScopeValue(&name, BifrostContextKeyStreamChunkWriter, write)
	defer scoped.ReleasePluginScope()

	first, second := &BifrostStreamChunk{}, &BifrostStreamChunk{}
	if !WriteStreamChunk(scoped, first) {
		t.Fatal("the writer's result is returned")
	}
	if WriteStreamChunk(scoped, second) {
		t.Fatal("the writer's result is returned")
	}
	if len(got) != 2 || got[0] != first || got[1] != second {
		t.Fatalf("the writer received %v", got)
	}

	// The request context, the scope's root and another scope of the request do not carry it.
	other := ctx.WithPluginScope(&name)
	defer other.ReleasePluginScope()
	for _, c := range []*BifrostContext{ctx, scoped.Root(), other} {
		if WriteStreamChunk(c, first) {
			t.Fatal("only the context the writer was handed to reaches it")
		}
	}
	if len(got) != 2 {
		t.Fatalf("the writer was reached from another context: %v", got)
	}
}

func TestWithPluginScopeValueShadowsOneKeyForOneScope(t *testing.T) {
	type key string
	ctx := NewBifrostContext(context.Background(), NoDeadline)
	ctx.SetValue(key("shared"), "request")
	ctx.SetValue(key("own"), "request")

	name := "plugin"
	scoped := ctx.WithPluginScopeValue(&name, key("own"), "scope")
	defer scoped.ReleasePluginScope()

	if got := scoped.Value(key("own")); got != "scope" {
		t.Fatalf("the scope reads its own value, got %v", got)
	}
	if got := scoped.Value(key("shared")); got != "request" {
		t.Fatalf("every other key is the request's, got %v", got)
	}
	if got := ctx.Value(key("own")); got != "request" {
		t.Fatalf("the request context is untouched, got %v", got)
	}
}
