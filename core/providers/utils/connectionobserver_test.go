package utils

import (
	"context"
	"crypto/tls"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/maximhq/bifrost/core/schemas"
	"github.com/valyala/fasthttp"
)

type connectionEvent struct {
	kind   string
	reused bool
	wait   time.Duration
}

// recordingConnectionObserver keeps the events for one address, so connections other tests
// in the package open while it is installed do not reach the assertions.
type recordingConnectionObserver struct {
	addr   string
	mu     sync.Mutex
	events []connectionEvent
}

func (o *recordingConnectionObserver) record(addr string, event connectionEvent) {
	if addr != o.addr {
		return
	}
	o.mu.Lock()
	defer o.mu.Unlock()
	o.events = append(o.events, event)
}

func (o *recordingConnectionObserver) ConnectionAcquired(addr string, reused bool, wait time.Duration) {
	o.record(addr, connectionEvent{kind: "acquired", reused: reused, wait: wait})
}
func (o *recordingConnectionObserver) ConnectionOpened(addr string) {
	o.record(addr, connectionEvent{kind: "opened"})
}
func (o *recordingConnectionObserver) ConnectionClosed(addr string) {
	o.record(addr, connectionEvent{kind: "closed"})
}

func (o *recordingConnectionObserver) snapshot() []connectionEvent {
	o.mu.Lock()
	defer o.mu.Unlock()
	return append([]connectionEvent(nil), o.events...)
}

func installRecordingConnectionObserver(t *testing.T, serverURL string) *recordingConnectionObserver {
	t.Helper()
	observer := &recordingConnectionObserver{addr: strings.TrimPrefix(serverURL, "http://")}
	schemas.SetConnectionObserver(observer)
	t.Cleanup(func() { schemas.SetConnectionObserver(nil) })
	return observer
}

func assertConnectionLifecycle(t *testing.T, events []connectionEvent) {
	t.Helper()
	var kinds []string
	for _, event := range events {
		kinds = append(kinds, event.kind)
	}
	want := []string{"opened", "acquired", "acquired", "closed"}
	if strings.Join(kinds, ",") != strings.Join(want, ",") {
		t.Fatalf("events = %v, want %v", kinds, want)
	}
	if events[1].reused || events[1].wait <= 0 {
		t.Errorf("first acquisition = %+v, want a new connection with a positive wait", events[1])
	}
	if !events[2].reused {
		t.Errorf("second acquisition = %+v, want the pooled connection", events[2])
	}
}

// Two sequential requests on a request-scoped fasthttp client: one dial, a new then a
// reused acquisition, and the close once idle connections are dropped.
func TestConnectionObserver_FastHTTP(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte("ok"))
	}))
	defer server.Close()
	observer := installRecordingConnectionObserver(t, server.URL)

	client := ConfigureDialerFor(&fasthttp.Client{ReadTimeout: 5 * time.Second}, schemas.NetworkConfig{LoopbackIsPrivate: true, AllowPrivateNetwork: true})
	for range 2 {
		req, resp := fasthttp.AcquireRequest(), fasthttp.AcquireResponse()
		req.SetRequestURI(server.URL)
		_, bifrostErr, wait := MakeRequestWithContext(context.Background(), client, req, resp)
		wait()
		if bifrostErr != nil {
			t.Fatalf("request: %v", bifrostErr.Error)
		}
		fasthttp.ReleaseRequest(req)
		fasthttp.ReleaseResponse(resp)
	}
	client.CloseIdleConnections()
	assertConnectionLifecycle(t, observer.snapshot())
}

// The same lifecycle on a net/http client dialing through ObservedDialContext.
func TestConnectionObserver_NetHTTP(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte("ok"))
	}))
	defer server.Close()
	observer := installRecordingConnectionObserver(t, server.URL)

	transport := &http.Transport{DialContext: ObservedDialContext(NetHTTPDialContextFor(schemas.NetworkConfig{LoopbackIsPrivate: true, AllowPrivateNetwork: true}, 5*time.Second))}
	client := &http.Client{Transport: transport}
	for range 2 {
		req, err := http.NewRequest(http.MethodGet, server.URL, nil)
		if err != nil {
			t.Fatal(err)
		}
		resp, err := DoHTTPRequest(client, req)
		if err != nil {
			t.Fatalf("request: %v", err)
		}
		_, _ = resp.Body.Read(make([]byte, 16))
		_ = resp.Body.Close()
	}
	transport.CloseIdleConnections()
	assertConnectionLifecycle(t, observer.snapshot())
}

// Without an observer a dialed connection is returned as is.
func TestConnectionObserver_UnsetWrapsNothing(t *testing.T) {
	schemas.SetConnectionObserver(nil)
	server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	defer server.Close()
	client := ConfigureDialerFor(&fasthttp.Client{ReadTimeout: 5 * time.Second}, schemas.NetworkConfig{LoopbackIsPrivate: true, AllowPrivateNetwork: true})
	conn, err := client.Dial(strings.TrimPrefix(server.URL, "http://"))
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = conn.Close() }()
	if _, wrapped := conn.(*observedConn); wrapped {
		t.Fatal("connection wrapped with no observer installed")
	}
}

// A custom dialer that returns an established TLS connection keeps working with an observer
// installed: the wrapper keeps Handshake, so fasthttp does not add a second TLS layer.
func TestConnectionObserver_PreEstablishedTLS(t *testing.T) {
	server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte("ok"))
	}))
	defer server.Close()
	addr := strings.TrimPrefix(server.URL, "https://")
	tlsConfig := server.Client().Transport.(*http.Transport).TLSClientConfig
	for _, observed := range []bool{false, true} {
		if observed {
			installRecordingConnectionObserver(t, "http://"+addr)
		}
		client := ConfigureDialer(&fasthttp.Client{
			ReadTimeout: 5 * time.Second,
			Dial: func(target string) (net.Conn, error) {
				raw, err := net.Dial("tcp", target)
				if err != nil {
					return nil, err
				}
				config := tlsConfig.Clone()
				config.ServerName = "example.com" // httptest's certificate names example.com
				conn := tls.Client(raw, config)
				if err := conn.Handshake(); err != nil {
					_ = raw.Close()
					return nil, err
				}
				return conn, nil
			},
		}, true)
		req, resp := fasthttp.AcquireRequest(), fasthttp.AcquireResponse()
		req.SetRequestURI(server.URL)
		_, bifrostErr, wait := MakeRequestWithContext(context.Background(), client, req, resp)
		wait()
		if bifrostErr != nil {
			t.Fatalf("observer installed=%v: %v", observed, bifrostErr.Error)
		}
		if string(resp.Body()) != "ok" {
			t.Fatalf("observer installed=%v: body %q", observed, resp.Body())
		}
		fasthttp.ReleaseRequest(req)
		fasthttp.ReleaseResponse(resp)
		client.CloseIdleConnections()
	}
}
