package schemas

import (
	"sync/atomic"
	"time"
)

// ConnectionObserver receives events about the outbound HTTP connections that provider
// clients open to upstream hosts, so an embedding application can export connection metrics.
// addr is the "host:port" the client connected to. Implementations must be safe for
// concurrent use and must return quickly: every method runs on the request path.
type ConnectionObserver interface {
	// ConnectionAcquired runs once per successful connection acquisition, including a
	// transport retry after a stale pooled connection; a failed acquisition reports nothing.
	// reused reports a connection taken from the pool; wait is the time spent obtaining it,
	// which for a new one includes the dial and TLS handshake. Every client that sends
	// through Bifrost's transports reports acquisitions, auxiliary ones (auth, media fetches)
	// included.
	ConnectionAcquired(addr string, reused bool, wait time.Duration)
	// ConnectionOpened runs when a provider inference client has dialed a new connection, and
	// ConnectionClosed exactly once when that connection is closed. Through a proxy, addr is
	// the proxy's for a net/http client and the target's for a fasthttp one. Auxiliary clients'
	// connections are not reported.
	ConnectionOpened(addr string)
	ConnectionClosed(addr string)
}

type connectionObserverHolder struct{ observer ConnectionObserver }

var connectionObserver atomic.Pointer[connectionObserverHolder]

// SetConnectionObserver installs observer for every provider client in the process; nil
// removes it. Set it before the first request: a connection dialed while no observer is set
// reports no ConnectionClosed later.
func SetConnectionObserver(observer ConnectionObserver) {
	if observer == nil {
		connectionObserver.Store(nil)
		return
	}
	connectionObserver.Store(&connectionObserverHolder{observer: observer})
}

// GetConnectionObserver returns the installed observer, or nil.
func GetConnectionObserver() ConnectionObserver {
	if holder := connectionObserver.Load(); holder != nil {
		return holder.observer
	}
	return nil
}
