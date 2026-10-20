package modelcatalog

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"sync"
	"testing"

	providerUtils "github.com/maximhq/bifrost/core/providers/utils"
	"github.com/maximhq/bifrost/core/schemas"
)

// infoRecorder keeps the Info messages a test catalog logs.
type infoRecorder struct {
	noOpLogger
	mu       sync.Mutex
	messages []string
}

func (r *infoRecorder) Info(format string, args ...any) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.messages = append(r.messages, fmt.Sprintf(format, args...))
}

// Without a config store the catalog loads the datasheet into memory, and the
// capability records providers read must come from that same sheet: a model's
// max_output_tokens is the output limit used when a request omits one. The load
// reports how many model-parameter rows it read, as the stored load does.
func TestInitWithoutConfigStoreServesCapabilitiesFromTheLoadedSheet(t *testing.T) {
	dir := t.TempDir()
	write := func(name, body string) string {
		path := filepath.Join(dir, name)
		if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
			t.Fatal(err)
		}
		return "file://" + path
	}
	// A model no built-in table knows, so only the sheet can supply its limit.
	pricingURL := write("pricing.json", `{"claude-example-9":{"provider":"anthropic","mode":"chat","input_cost_per_token":0.000003,"output_cost_per_token":0.000015}}`)
	paramsURL := write("model-parameters.json", `{"claude-example-9":{"provider":"anthropic","mode":"chat","max_output_tokens":128000,"supported_endpoints":["/v1/chat/completions"]}}`)
	disabledInterval := int64(0)
	logs := &infoRecorder{}

	catalog, err := Init(context.Background(), &Config{
		PricingURL:             &pricingURL,
		ModelParametersURL:     &paramsURL,
		MCPLibrarySyncInterval: &disabledInterval,
	}, nil, logs)
	if err != nil {
		t.Fatalf("Init() without a config store: %v", err)
	}
	t.Cleanup(func() { _ = catalog.Cleanup() })
	logs.mu.Lock()
	logged := slices.Contains(logs.messages, "successfully synced 1 model parameters records")
	logs.mu.Unlock()
	if !logged {
		t.Fatalf("Info logs = %q, want the model-parameter row count", logs.messages)
	}

	caps := catalog.GetModelCapabilities(schemas.Anthropic, "claude-example-9")
	if caps == nil || caps.MaxOutputTokens == nil || *caps.MaxOutputTokens != 128000 {
		t.Fatalf("capabilities = %+v, want the sheet's max_output_tokens", caps)
	}
	if got := providerUtils.GetMaxOutputTokensOrDefault(schemas.Anthropic, "claude-example-9", 4096); got != 128000 {
		t.Fatalf("default output limit = %d, want the sheet's 128000", got)
	}
	params, err := catalog.ResolveModelParameters(context.Background(), "claude-example-9")
	if err != nil || params == nil || params.Model != "claude-example-9" {
		t.Fatalf("ResolveModelParameters = (%+v, %v), want the sheet's row", params, err)
	}
	if caps := catalog.GetModelCapabilities(schemas.OpenAI, "claude-example-9"); caps != nil {
		t.Fatalf("another provider's lookup was served this row: %+v", caps)
	}
}
