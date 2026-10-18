package utils

import (
	"context"
	"net"
	"net/http"
	"net/http/httptrace"
	"sync/atomic"
	"time"

	"github.com/maximhq/bifrost/core/schemas"
)

// This file reports provider connections to the schemas.ConnectionObserver, when one is
// installed: a connection dialed by a provider client (ConfigureDialer/ConfigureDialerFor for
// fasthttp, ObservedDialContext for net/http) is wrapped so its Close is reported,
// contextTransport reports each fasthttp connection acquisition, and DoHTTPRequest traces
// net/http ones. With no observer installed nothing is wrapped, traced or timed.

// observedConn reports its first Close to the observer that saw it open.
type observedConn struct {
	net.Conn
	observer schemas.ConnectionObserver
	addr     string
	closed   atomic.Bool
}

func (c *observedConn) Close() error {
	if c.closed.CompareAndSwap(false, true) {
		c.observer.ConnectionClosed(c.addr)
	}
	return c.Conn.Close()
}

// observedHandshakeConn keeps the Handshake method of a dialed connection that already speaks
// TLS, which clients check for before adding their own TLS layer.
type observedHandshakeConn struct {
	*observedConn
	handshaker interface{ Handshake() error }
}

func (c *observedHandshakeConn) Handshake() error { return c.handshaker.Handshake() }

// observeDialedConn reports a connection just dialed to addr and returns it wrapped so its
// close is reported too. Without an observer it returns conn unchanged.
func observeDialedConn(conn net.Conn, addr string) net.Conn {
	observer := schemas.GetConnectionObserver()
	if observer == nil || conn == nil {
		return conn
	}
	observer.ConnectionOpened(addr)
	observed := &observedConn{Conn: conn, observer: observer, addr: addr}
	if handshaker, ok := conn.(interface{ Handshake() error }); ok {
		return &observedHandshakeConn{observedConn: observed, handshaker: handshaker}
	}
	return observed
}

// ObservedDialContext returns dial wrapped so the connections it opens are reported to the
// connection observer; a nil dial is net/http's default (a zero net.Dialer). Through a proxy
// the reported address is the proxy's, since that is the connection dialed.
func ObservedDialContext(dial func(ctx context.Context, network, addr string) (net.Conn, error)) func(ctx context.Context, network, addr string) (net.Conn, error) {
	if dial == nil {
		dial = (&net.Dialer{}).DialContext
	}
	return func(ctx context.Context, network, addr string) (net.Conn, error) {
		conn, err := dial(ctx, network, addr)
		if err != nil {
			return nil, err
		}
		return observeDialedConn(conn, addr), nil
	}
}

// connectionAcquireStart returns the time an acquisition starts, or the zero time when no
// observer is installed, so an unobserved request reads no clock.
func connectionAcquireStart() time.Time {
	if schemas.GetConnectionObserver() == nil {
		return time.Time{}
	}
	return time.Now()
}

// observeConnectionAcquired reports a connection to addr obtained since start.
func observeConnectionAcquired(addr string, reused bool, start time.Time) {
	if start.IsZero() {
		return
	}
	if observer := schemas.GetConnectionObserver(); observer != nil {
		observer.ConnectionAcquired(addr, reused, time.Since(start))
	}
}

// withConnectionTrace returns req with a trace reporting its net/http connection acquisition,
// or req unchanged without an observer.
func withConnectionTrace(req *http.Request) *http.Request {
	if schemas.GetConnectionObserver() == nil {
		return req
	}
	var addr string
	var start time.Time
	trace := &httptrace.ClientTrace{
		GetConn: func(hostPort string) {
			addr, start = hostPort, time.Now()
		},
		GotConn: func(info httptrace.GotConnInfo) {
			if !start.IsZero() {
				observeConnectionAcquired(addr, info.Reused, start)
			}
		},
	}
	return req.WithContext(httptrace.WithClientTrace(req.Context(), trace))
}
