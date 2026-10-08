package lib

import (
	"bytes"
	"context"
	"io"
	"sync"
	"testing"

	"github.com/maximhq/bifrost/core/schemas"
)

// recordingTap records each Write as one frame.
type recordingTap struct {
	mu     sync.Mutex
	frames [][]byte
}

func (tap *recordingTap) Write(p []byte) (int, error) {
	tap.mu.Lock()
	defer tap.mu.Unlock()
	tap.frames = append(tap.frames, append([]byte(nil), p...))
	return len(p), nil
}

func (tap *recordingTap) joined() []byte {
	tap.mu.Lock()
	defer tap.mu.Unlock()
	return bytes.Join(tap.frames, nil)
}

func drainReader(t *testing.T, reader *SSEStreamReader) <-chan []byte {
	t.Helper()
	out := make(chan []byte, 1)
	go func() {
		data, _ := io.ReadAll(reader)
		out <- data
	}()
	return out
}

func TestSSEStreamReaderFrameTapReceivesEveryReleasedByte(t *testing.T) {
	reader := NewSSEStreamReader()
	tap := &recordingTap{}
	reader.SetFrameTap(tap)
	received := drainReader(t, reader)

	if !reader.SendEvent("", []byte(`{"n":1}`)) ||
		!reader.SendHeartbeat() ||
		!reader.SendEvent("message", []byte(`{"n":2}`)) ||
		!reader.Send([]byte("data: raw")) ||
		!reader.Send(nil) ||
		!reader.Send([]byte("\n\n")) ||
		!reader.SendError([]byte(`{"error":"x"}`)) ||
		!reader.SendDone() {
		t.Fatal("send reported a closed reader")
	}
	reader.Done()
	wire := <-received

	if got := tap.joined(); !bytes.Equal(got, wire) {
		t.Fatalf("tap bytes differ from released bytes:\n tap: %q\nwire: %q", got, wire)
	}
	want := []string{
		"data: {\"n\":1}\n\n", ": heartbeat\n", "event: message\ndata: {\"n\":2}\n\n",
		"data: raw", "\n\n", "event: error\ndata: {\"error\":\"x\"}\n\n", "data: [DONE]\n\n",
	}
	if len(tap.frames) != len(want) {
		t.Fatalf("tap writes = %d %q, want one per non-empty release (%d)", len(tap.frames), tap.frames, len(want))
	}
	for i := range want {
		if string(tap.frames[i]) != want[i] {
			t.Errorf("tap write %d = %q, want %q", i, tap.frames[i], want[i])
		}
	}
}

func TestSSEStreamReaderFrameTapSkipsUnreleasedBytes(t *testing.T) {
	reader := NewSSEStreamReader()
	tap := &recordingTap{}
	reader.SetFrameTap(tap)
	received := drainReader(t, reader)
	reader.SendEvent("", []byte(`{"n":1}`))
	reader.Send([]byte("data: partial"))
	// A heartbeat mid-line is skipped, so it is not released and not tapped.
	if !reader.SendHeartbeat() {
		t.Fatal("skipped heartbeat reported a disconnect")
	}
	reader.Send([]byte("\n\n"))
	reader.Done()
	<-received

	after := NewSSEStreamReader()
	afterTap := &recordingTap{}
	after.SetFrameTap(afterTap)
	_ = after.Close()
	if after.SendEvent("", []byte(`{"n":2}`)) || after.SendHeartbeat() || after.SendDone() {
		t.Fatal("send on a closed reader reported success")
	}
	if len(afterTap.frames) != 0 {
		t.Fatalf("closed reader tapped %q", afterTap.frames)
	}
	if got, want := string(tap.joined()), "data: {\"n\":1}\n\ndata: partial\n\n"; got != want {
		t.Fatalf("tap = %q, want %q", got, want)
	}
}

func TestStreamFrameTapReadsTheRequestContext(t *testing.T) {
	if StreamFrameTap(nil) != nil {
		t.Fatal("nil context returned a tap")
	}
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	defer ctx.Cancel()
	if StreamFrameTap(ctx) != nil {
		t.Fatal("context without a tap returned one")
	}
	tap := &recordingTap{}
	ctx.SetValue(schemas.BifrostContextKeyStreamFrameTap, io.Writer(tap))
	if StreamFrameTap(ctx) != io.Writer(tap) {
		t.Fatal("context tap was not returned")
	}
	ctx.SetValue(schemas.BifrostContextKeyStreamFrameTap, "not a writer")
	if StreamFrameTap(ctx) != nil {
		t.Fatal("a non-writer value was returned as a tap")
	}
}

func TestSSEStreamReaderNilFrameTapAllocatesNothing(t *testing.T) {
	reader := NewSSEStreamReader()
	reader.SetFrameTap(nil)
	event := []byte("data: {}\n\n")
	go func() {
		buffer := make([]byte, 64)
		for {
			if _, err := reader.Read(buffer); err != nil {
				return
			}
		}
	}()
	allocations := testing.AllocsPerRun(1000, func() { reader.Send(event) })
	reader.Done()
	if allocations != 0 {
		t.Fatalf("Send with a nil tap allocated %v times per call, want 0", allocations)
	}
}

func benchmarkSSEStreamReaderSend(b *testing.B, tap io.Writer) {
	reader := NewSSEStreamReader()
	reader.SetFrameTap(tap)
	done := make(chan struct{})
	go func() {
		defer close(done)
		buffer := make([]byte, 64)
		for {
			if _, err := reader.Read(buffer); err != nil {
				return
			}
		}
	}()
	event := []byte("data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n\n")
	b.ReportAllocs()
	for b.Loop() {
		reader.Send(event)
	}
	reader.Done()
	<-done
}

func BenchmarkSSEStreamReaderSendNilTap(b *testing.B) { benchmarkSSEStreamReaderSend(b, nil) }

func BenchmarkSSEStreamReaderSendWithTap(b *testing.B) { benchmarkSSEStreamReaderSend(b, io.Discard) }
