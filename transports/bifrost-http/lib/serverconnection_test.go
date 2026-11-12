package lib

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

// Not set, the inbound connection settings keep fasthttp's behavior: buffers are kept
// between requests and neither reading a request nor an idle connection times out.
func TestServerConfigConnectionSettingsDefaultToNoTimeouts(t *testing.T) {
	var unset *ServerConfig
	require.Zero(t, unset.ConnectionReadTimeout())
	require.Zero(t, unset.ConnectionIdleTimeout())

	var cfg ServerConfig
	require.NoError(t, json.Unmarshal([]byte(`{"read_buffer_size":65536}`), &cfg))
	require.False(t, cfg.ReduceMemoryUsage)
	require.Zero(t, cfg.ConnectionReadTimeout())
	require.Zero(t, cfg.ConnectionIdleTimeout())
}

func TestServerConfigConnectionSettingsAreConfigurable(t *testing.T) {
	var cfg ServerConfig
	require.NoError(t, json.Unmarshal([]byte(`{"read_buffer_size":65536,"reduce_memory_usage":true,"read_timeout_seconds":300,"idle_timeout_seconds":90}`), &cfg))
	require.True(t, cfg.ReduceMemoryUsage)
	require.Equal(t, 300*time.Second, cfg.ConnectionReadTimeout())
	require.Equal(t, 90*time.Second, cfg.ConnectionIdleTimeout())

	// A negative value is not a timeout; it means none, like 0.
	negative := ServerConfig{ReadTimeoutSeconds: -1, IdleTimeoutSeconds: -1}
	require.Zero(t, negative.ConnectionReadTimeout())
	require.Zero(t, negative.ConnectionIdleTimeout())
}

// config.schema.json declares the three settings with the defaults the server applies.
func TestServerConnectionSettingsSchemaMatchesDefaults(t *testing.T) {
	data, err := os.ReadFile(filepath.Join(repoRoot(t), "transports", "config.schema.json"))
	require.NoError(t, err)
	var schema struct {
		Properties struct {
			Server struct {
				Properties map[string]struct {
					Type    string `json:"type"`
					Default any    `json:"default"`
				} `json:"properties"`
			} `json:"server"`
		} `json:"properties"`
	}
	require.NoError(t, json.Unmarshal(data, &schema))
	server := schema.Properties.Server.Properties
	require.Equal(t, "boolean", server["reduce_memory_usage"].Type)
	require.Equal(t, false, server["reduce_memory_usage"].Default)
	for _, key := range []string{"read_timeout_seconds", "idle_timeout_seconds"} {
		require.Equal(t, "integer", server[key].Type, key)
		require.Equal(t, float64(0), server[key].Default, "%s must default to no timeout", key)
	}
}
