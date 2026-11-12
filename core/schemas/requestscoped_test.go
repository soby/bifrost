package schemas

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

func TestBifrostRequestUpdateProviderKey(t *testing.T) {
	var nilReq *BifrostRequest
	if err := nilReq.UpdateProviderKey(OpenAI, Key{}); err == nil {
		t.Fatal("UpdateProviderKey on a nil request: want error")
	}
	req := &BifrostRequest{}
	if err := req.UpdateProviderKey("", Key{}); err == nil {
		t.Fatal("UpdateProviderKey with an empty provider: want error")
	}
	if len(req.ProviderOverrides) != 0 {
		t.Fatalf("a rejected call added %d overrides", len(req.ProviderOverrides))
	}

	if err := req.UpdateProviderKey(OpenAI, Key{ID: "a", Value: SecretVar{Val: "sk-a"}}); err != nil {
		t.Fatalf("UpdateProviderKey: %v", err)
	}
	if err := req.UpdateProviderKey(Anthropic, Key{Value: SecretVar{Val: "sk-ant"}}); err != nil {
		t.Fatalf("UpdateProviderKey: %v", err)
	}
	// A second call for the same provider rewrites its entry in place.
	if err := req.UpdateProviderKey(OpenAI, Key{ID: "b", Value: SecretVar{Val: "sk-b"}}); err != nil {
		t.Fatalf("UpdateProviderKey: %v", err)
	}
	if len(req.ProviderOverrides) != 2 {
		t.Fatalf("got %d overrides, want 2", len(req.ProviderOverrides))
	}
	if got := req.ProviderOverrideFor(OpenAI); got == nil || got.Key.ID != "b" || got.Key.Value.GetValue() != "sk-b" {
		t.Fatalf("OpenAI override = %+v, want key b", got)
	}
	if got := req.ProviderOverrideFor(Anthropic); got == nil || got.Key.Value.GetValue() != "sk-ant" {
		t.Fatalf("Anthropic override = %+v, want sk-ant", got)
	}

	key := Key{ID: "c", Value: SecretVar{Val: "sk-c"}}
	if allocs := testing.AllocsPerRun(100, func() {
		if err := req.UpdateProviderKey(OpenAI, key); err != nil {
			t.Fatal(err)
		}
	}); allocs != 0 {
		t.Errorf("rewriting an existing override: %v allocs, want 0", allocs)
	}
}

func TestBifrostRequestUpdateProviderKeyRejectsSecretReferences(t *testing.T) {
	envRef := NewSecretVar("env.BIFROST_TEST_REQUEST_SCOPED_KEY")
	vaultRef := &SecretVar{ref: "vault.secret/path", SecretType: SecretTypeVault}

	tests := []struct {
		name string
		key  Key
	}{
		{"value from env", Key{Value: *envRef}},
		{"value from vault", Key{Value: *vaultRef}},
		{"azure endpoint", Key{Value: SecretVar{Val: "k"}, AzureKeyConfig: &AzureKeyConfig{Endpoint: *envRef}}},
		{"azure client secret", Key{AzureKeyConfig: &AzureKeyConfig{Endpoint: SecretVar{Val: "https://x"}, ClientSecret: vaultRef}}},
		{"vertex credentials", Key{VertexKeyConfig: &VertexKeyConfig{AuthCredentials: *envRef}}},
		{"bedrock region", Key{Value: SecretVar{Val: "k"}, BedrockKeyConfig: &BedrockKeyConfig{Region: envRef}}},
		{"bedrock endpoint", Key{BedrockKeyConfig: &BedrockKeyConfig{Endpoints: &BedrockEndpoints{Runtime: vaultRef}}}},
		{"bedrock mantle secret key", Key{BedrockMantleKeyConfig: &BedrockMantleKeyConfig{SecretKey: *envRef}}},
		{"vllm url", Key{VLLMKeyConfig: &VLLMKeyConfig{URL: *envRef}}},
		{"ollama url", Key{OllamaKeyConfig: &OllamaKeyConfig{URL: *envRef}}},
		{"sgl url", Key{SGLKeyConfig: &SGLKeyConfig{URL: *envRef}}},
		{"databricks workspace", Key{DatabricksKeyConfig: &DatabricksKeyConfig{WorkspaceURL: *envRef}}},
		{"alias region", Key{Aliases: KeyAliases{"m": {ModelID: "m", Region: envRef}}}},
		{"alias azure endpoint", Key{Aliases: KeyAliases{"m": {ModelID: "m", AzureAliasCfg: &AzureAliasCfg{Endpoint: envRef}}}}},
		{"alias bedrock profile", Key{Aliases: KeyAliases{"m": {ModelID: "m", BedrockAliasCfg: &BedrockAliasCfg{InferenceProfileARN: vaultRef}}}}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := &BifrostRequest{}
			if err := req.UpdateProviderKey(OpenAI, tt.key); err == nil {
				t.Fatal("want error for a key holding a secret reference")
			}
			if len(req.ProviderOverrides) != 0 {
				t.Fatalf("a rejected key added %d overrides", len(req.ProviderOverrides))
			}
		})
	}

	literal := Key{
		Value:          SecretVar{Val: "sk"},
		AzureKeyConfig: &AzureKeyConfig{Endpoint: *NewSecretVar("https://example.openai.azure.com")},
		Aliases:        KeyAliases{"m": {ModelID: "deployment"}},
	}
	if err := (&BifrostRequest{}).UpdateProviderKey(Azure, literal); err != nil {
		t.Fatalf("literal key rejected: %v", err)
	}
}

// secretVarSetter sets one SecretVar reachable from a value of some type, allocating the
// pointers, maps and slices on the way.
type secretVarSetter struct {
	path string
	set  func(v reflect.Value, ref SecretVar)
}

// secretVarSetters returns a setter for every SecretVar reachable from a value of type t
// through exported fields, pointers, maps and slices.
func secretVarSetters(t reflect.Type, path string, visiting map[reflect.Type]bool) []secretVarSetter {
	if t == reflect.TypeFor[SecretVar]() {
		return []secretVarSetter{{path, func(v reflect.Value, ref SecretVar) { v.Set(reflect.ValueOf(ref)) }}}
	}
	if visiting[t] {
		return nil
	}
	visiting[t] = true
	defer delete(visiting, t)
	var setters []secretVarSetter
	switch t.Kind() {
	case reflect.Pointer:
		for _, s := range secretVarSetters(t.Elem(), path, visiting) {
			setters = append(setters, secretVarSetter{s.path, func(v reflect.Value, ref SecretVar) {
				p := reflect.New(t.Elem())
				s.set(p.Elem(), ref)
				v.Set(p)
			}})
		}
	case reflect.Struct:
		for i := range t.NumField() {
			if field := t.Field(i); field.IsExported() {
				for _, s := range secretVarSetters(field.Type, path+"."+field.Name, visiting) {
					setters = append(setters, secretVarSetter{s.path, func(v reflect.Value, ref SecretVar) { s.set(v.Field(i), ref) }})
				}
			}
		}
	case reflect.Map:
		for _, s := range secretVarSetters(t.Elem(), path+"[k]", visiting) {
			setters = append(setters, secretVarSetter{s.path, func(v reflect.Value, ref SecretVar) {
				elem := reflect.New(t.Elem()).Elem()
				s.set(elem, ref)
				m := reflect.MakeMap(t)
				m.SetMapIndex(reflect.ValueOf("k").Convert(t.Key()), elem)
				v.Set(m)
			}})
		}
	case reflect.Slice, reflect.Array:
		for _, s := range secretVarSetters(t.Elem(), path+"[0]", visiting) {
			setters = append(setters, secretVarSetter{s.path, func(v reflect.Value, ref SecretVar) {
				if t.Kind() == reflect.Slice {
					v.Set(reflect.MakeSlice(t, 1, 1))
				}
				s.set(v.Index(0), ref)
			}})
		}
	}
	return setters
}

// TestKeyHasSecretReferenceCoversEveryField pins that a reference in any SecretVar of a Key,
// including ones added to Key and its configs later, makes UpdateProviderKey reject the key.
func TestKeyHasSecretReferenceCoversEveryField(t *testing.T) {
	setters := secretVarSetters(reflect.TypeFor[Key](), "Key", map[reflect.Type]bool{})
	if len(setters) < 20 {
		t.Fatalf("found only %d SecretVar fields in Key; the walk is broken", len(setters))
	}
	for _, ref := range []*SecretVar{NewSecretVar("env.BIFROST_TEST_REQUEST_SCOPED_KEY"), {ref: "vault.secret/path", SecretType: SecretTypeVault}} {
		for _, s := range setters {
			var key Key
			s.set(reflect.ValueOf(&key).Elem(), *ref)
			if !key.hasSecretReference() {
				t.Errorf("%s holding %s reference is not detected", s.path, ref.Type())
			}
		}
	}
}

func TestBifrostRequestUpdateProviderBaseURL(t *testing.T) {
	tests := []struct {
		name    string
		baseURL string
		want    string
		wantErr bool
	}{
		{name: "https", baseURL: "https://api.example.com", want: "https://api.example.com"},
		{name: "http with port", baseURL: "http://10.0.0.5:8000", want: "http://10.0.0.5:8000"},
		{name: "path kept", baseURL: "https://example.com/v1beta", want: "https://example.com/v1beta"},
		{name: "trailing slashes trimmed", baseURL: "https://example.com/openai//", want: "https://example.com/openai"},
		{name: "ipv6 literal", baseURL: "http://[::1]:11434", want: "http://[::1]:11434"},
		{name: "empty", baseURL: "", wantErr: true},
		{name: "unsupported scheme", baseURL: "ftp://example.com", wantErr: true},
		{name: "relative", baseURL: "/v1", wantErr: true},
		{name: "scheme only", baseURL: "https://", wantErr: true},
		{name: "no host", baseURL: "https:///v1", wantErr: true},
		{name: "port only", baseURL: "https://:443", wantErr: true},
		{name: "user info", baseURL: "https://user:pass@example.com", wantErr: true},
		{name: "query", baseURL: "https://example.com/v1?key=x", wantErr: true},
		{name: "fragment", baseURL: "https://example.com/#x", wantErr: true},
		{name: "backslash", baseURL: `https://example.com\@evil.com`, wantErr: true},
		{name: "whitespace", baseURL: "https://example.com /v1", wantErr: true},
		{name: "control character", baseURL: "https://example.com\n", wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := &BifrostRequest{}
			err := req.UpdateProviderBaseURL(OpenAI, tt.baseURL)
			if tt.wantErr {
				if err == nil {
					t.Fatalf("UpdateProviderBaseURL(%q): want error", tt.baseURL)
				}
				// The URL may carry a credential, so the error must not quote it.
				if (tt.baseURL != "" && strings.Contains(err.Error(), tt.baseURL)) || len(req.ProviderOverrides) != 0 {
					t.Fatalf("rejected URL: error %q, %d overrides added", err, len(req.ProviderOverrides))
				}
				return
			}
			if err != nil {
				t.Fatalf("UpdateProviderBaseURL(%q): %v", tt.baseURL, err)
			}
			if got := req.ProviderOverrideFor(OpenAI).BaseURL; got != tt.want {
				t.Fatalf("BaseURL = %q, want %q", got, tt.want)
			}
		})
	}

	req := &BifrostRequest{}
	if err := req.UpdateProviderBaseURL(OpenAI, "https://a.example.com"); err != nil {
		t.Fatal(err)
	}
	if allocs := testing.AllocsPerRun(100, func() {
		if err := req.UpdateProviderBaseURL(OpenAI, "https://b.example.com/"); err != nil {
			t.Fatal(err)
		}
	}); allocs != 0 {
		t.Errorf("UpdateProviderBaseURL on an existing override: %v allocs, want 0", allocs)
	}
}

func TestBifrostRequestUpdateProviderAllowPrivateNetwork(t *testing.T) {
	req := &BifrostRequest{}
	if err := req.UpdateProviderAllowPrivateNetwork(Ollama, true); err != nil {
		t.Fatal(err)
	}
	if err := req.UpdateProviderBaseURL(Ollama, "http://10.0.0.5:11434"); err != nil {
		t.Fatal(err)
	}
	got := req.ProviderOverrideFor(Ollama)
	if got == nil || !got.AllowPrivateNetwork || got.BaseURL != "http://10.0.0.5:11434" || len(req.ProviderOverrides) != 1 {
		t.Fatalf("override = %+v (%d entries), want one entry with both settings", got, len(req.ProviderOverrides))
	}
	if err := req.UpdateProviderAllowPrivateNetwork("", true); err == nil {
		t.Fatal("empty provider: want error")
	}
}

func TestBifrostRequestProviderOverrideFor(t *testing.T) {
	var nilReq *BifrostRequest
	if nilReq.ProviderOverrideFor(OpenAI) != nil {
		t.Fatal("nil request: want nil override")
	}
	req := &BifrostRequest{ChatRequest: &BifrostChatRequest{Provider: OpenAI}}
	if allocs := testing.AllocsPerRun(100, func() {
		if req.ProviderOverrideFor(OpenAI) != nil {
			t.Fatal("want nil override")
		}
	}); allocs != 0 {
		t.Errorf("ProviderOverrideFor without overrides: %v allocs, want 0", allocs)
	}
	if err := req.UpdateProviderKey(Gemini, Key{Value: SecretVar{Val: "g"}}); err != nil {
		t.Fatal(err)
	}
	if req.ProviderOverrideFor(OpenAI) != nil {
		t.Fatal("override for another provider returned")
	}
	// The returned pointer addresses the request's entry.
	req.ProviderOverrideFor(Gemini).BaseURL = "https://g.example.com/v1beta"
	if got := req.ProviderOverrides[0].BaseURL; got != "https://g.example.com/v1beta" {
		t.Fatalf("BaseURL = %q, want the value written through the pointer", got)
	}
}

func TestBifrostRequestProviderOverridesNotSerialized(t *testing.T) {
	req := &BifrostRequest{RequestType: ChatCompletionRequest, ChatRequest: &BifrostChatRequest{Provider: OpenAI, Model: "gpt-4o"}}
	if err := req.UpdateProviderKey(OpenAI, Key{ID: "plugin-key", Value: SecretVar{Val: "sk-request-scoped-secret"}}); err != nil {
		t.Fatal(err)
	}
	if err := req.UpdateProviderBaseURL(OpenAI, "https://scoped.example.com"); err != nil {
		t.Fatal(err)
	}
	data, err := json.Marshal(req)
	if err != nil {
		t.Fatal(err)
	}
	for _, leaked := range []string{"ProviderOverrides", "sk-request-scoped-secret", "scoped.example.com", "plugin-key"} {
		if strings.Contains(string(data), leaked) {
			t.Errorf("marshaled request contains %q: %s", leaked, data)
		}
	}
}

// TestUpdateProviderKeyRefusesKeysThatAreNotReadyCredentials pins that a request's own key is
// a ready credential with a well-formed endpoint, and that a refused key leaves no override.
func TestUpdateProviderKeyRefusesKeysThatAreNotReadyCredentials(t *testing.T) {
	literal := func(value string) SecretVar { return SecretVar{Val: value} }
	tests := []struct {
		name    string
		key     Key
		wantErr string
	}{
		{"oauth key config", Key{OAuthKeyConfig: &OAuthKeyConfig{TokenURL: literal("http://127.0.0.1:8080/token")}}, "oauth_key_config"},
		{"anthropic workload identity", Key{AnthropicKeyConfig: &AnthropicKeyConfig{WorkloadIdentity: &AnthropicWorkloadIdentityConfig{}}}, "workload identity"},
		{"azure endpoint with a query", Key{Value: literal("k"), AzureKeyConfig: &AzureKeyConfig{Endpoint: literal("https://example.com/internal?x=1")}}, "azure_key_config.endpoint"},
		{"vllm url with a fragment", Key{VLLMKeyConfig: &VLLMKeyConfig{URL: literal("https://example.com/admin#")}}, "vllm_key_config.url"},
		{"ollama url with user info", Key{OllamaKeyConfig: &OllamaKeyConfig{URL: literal("https://user@example.com")}}, "ollama_key_config.url"},
		{"sgl url without a scheme", Key{SGLKeyConfig: &SGLKeyConfig{URL: literal("example.com")}}, "sgl_key_config.url"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := &BifrostRequest{}
			err := req.UpdateProviderKey(OpenAI, tt.key)
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("UpdateProviderKey error = %v, want one containing %q", err, tt.wantErr)
			}
			if len(req.ProviderOverrides) != 0 {
				t.Fatalf("a refused key left %d overrides on the request", len(req.ProviderOverrides))
			}
		})
	}

	// A refused key leaves an existing override as it was.
	req := &BifrostRequest{}
	if err := req.UpdateProviderKey(OpenAI, Key{ID: "kept", Value: literal("sk-kept")}); err != nil {
		t.Fatalf("UpdateProviderKey: %v", err)
	}
	if err := req.UpdateProviderKey(OpenAI, Key{OAuthKeyConfig: &OAuthKeyConfig{}}); err == nil {
		t.Fatal("an oauth key was accepted")
	}
	if got := req.ProviderOverrideFor(OpenAI); got == nil || got.Key.ID != "kept" || got.Key.Value.GetValue() != "sk-kept" {
		t.Fatalf("override after a refused key = %+v, want the earlier key", got)
	}

	// Well-formed endpoints are accepted.
	for name, key := range map[string]Key{
		"azure":  {Value: literal("k"), AzureKeyConfig: &AzureKeyConfig{Endpoint: literal("https://example.openai.azure.com")}},
		"vllm":   {VLLMKeyConfig: &VLLMKeyConfig{URL: literal("http://vllm.internal:8000/")}},
		"ollama": {OllamaKeyConfig: &OllamaKeyConfig{URL: literal("http://ollama.internal:11434")}},
	} {
		if err := (&BifrostRequest{}).UpdateProviderKey(OpenAI, key); err != nil {
			t.Errorf("%s: a well-formed endpoint was refused: %v", name, err)
		}
	}
}
