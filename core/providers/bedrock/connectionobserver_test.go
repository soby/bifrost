package bedrock

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	providerUtils "github.com/maximhq/bifrost/core/providers/utils"
	"github.com/maximhq/bifrost/core/schemas"
	"github.com/stretchr/testify/require"
)

type bedrockConnectionCounter struct {
	addr                     string
	mu                       sync.Mutex
	opened, closed, acquired int
}

func (c *bedrockConnectionCounter) ConnectionAcquired(addr string, _ bool, _ time.Duration) {
	c.count(addr, &c.acquired)
}
func (c *bedrockConnectionCounter) ConnectionOpened(addr string) { c.count(addr, &c.opened) }
func (c *bedrockConnectionCounter) ConnectionClosed(addr string) { c.count(addr, &c.closed) }

func (c *bedrockConnectionCounter) count(addr string, n *int) {
	if addr != c.addr {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	*n++
}

// Bedrock's net/http client reports the whole connection lifecycle, opened and closed as well
// as acquired, for a configured provider and for a request-scoped instance alike.
func TestBedrockClientReportsConnectionLifecycle(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte("ok"))
	}))
	defer server.Close()

	for name, network := range map[string]schemas.NetworkConfig{
		"configured":     {DefaultRequestTimeoutInSeconds: 30},
		"request-scoped": {DefaultRequestTimeoutInSeconds: 30, LoopbackIsPrivate: true, AllowPrivateNetwork: true, ContextBoundReads: true},
	} {
		t.Run(name, func(t *testing.T) {
			counter := &bedrockConnectionCounter{addr: strings.TrimPrefix(server.URL, "http://")}
			schemas.SetConnectionObserver(counter)
			t.Cleanup(func() { schemas.SetConnectionObserver(nil) })

			config := &schemas.ProviderConfig{NetworkConfig: network}
			config.CheckAndSetDefaults()
			provider, err := NewBedrockProvider(config, noopLogger{})
			require.NoError(t, err)

			for range 2 {
				req, err := http.NewRequest(http.MethodGet, server.URL, nil)
				require.NoError(t, err)
				resp, err := providerUtils.DoHTTPRequest(provider.client, req)
				require.NoError(t, err)
				_, _ = resp.Body.Read(make([]byte, 16))
				require.NoError(t, resp.Body.Close())
			}
			provider.client.CloseIdleConnections()

			counter.mu.Lock()
			defer counter.mu.Unlock()
			require.Equal(t, 1, counter.opened, "opened")
			require.Equal(t, 2, counter.acquired, "acquired")
			require.Equal(t, 1, counter.closed, "closed")
		})
	}
}
