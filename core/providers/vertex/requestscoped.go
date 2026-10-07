package vertex

import (
	"errors"
	"strings"

	schemas "github.com/maximhq/bifrost/core/schemas"
)

// This file is fork-only: upstream Vertex does not implement schemas.RequestScopedProvider,
// because every credential mode it supports either reads ambient credentials (Application
// Default Credentials, AWS workload identity federation) or caches a token source per
// credential. The fork adds one mode that needs neither: a caller-supplied OAuth access token
// on VertexKeyConfig.AccessToken.

// hasCallerAccessToken reports whether key carries a caller-supplied access token, which
// getAuthTokenSource uses in place of every other Vertex credential.
func hasCallerAccessToken(key schemas.Key) bool {
	return key.VertexKeyConfig != nil && key.VertexKeyConfig.AccessToken != ""
}

// ForRequest implements schemas.RequestScopedProvider. A request-scoped Vertex key must carry a
// caller-supplied access token and nothing else that authenticates: an API key, service-account
// JSON or AWS workload identity is rejected, so an attempt can never fall back to Application
// Default Credentials or AWS federation, or populate the per-provider token-source cache. The
// region and project come from the key (or its aliases) and choose the host and path Bifrost
// calls, so they are checked against the characters Google allows; a base URL is rejected. The
// receiver is never copied, since it holds a sync.Map by value.
func (provider *VertexProvider) ForRequest(_ schemas.RequestType, key schemas.Key, baseURL string) (schemas.Provider, error) {
	if baseURL != "" {
		return nil, errors.New("vertex takes its endpoint from the key's region, not a base URL")
	}
	cfg := key.VertexKeyConfig
	switch {
	case cfg == nil:
		return nil, errors.New("vertex requires vertex_key_config")
	case cfg.AccessToken == "":
		return nil, errors.New("vertex requires a non-empty vertex_key_config access token")
	case !isBearerToken(cfg.AccessToken):
		return nil, errors.New("vertex access token must not contain whitespace or control characters")
	case key.Value.GetValue() != "":
		return nil, errors.New("vertex API keys are not supported per request, use an access token")
	case cfg.AuthCredentials.GetValue() != "" || cfg.AWSWorkloadIdentity != nil:
		return nil, errors.New("vertex service-account credentials and AWS workload identity are not supported per request, use an access token")
	case !isVertexLocation(cfg.Region.GetValue()):
		return nil, errors.New("vertex requires a valid vertex_key_config.region")
	case !isVertexProjectID(cfg.ProjectID.GetValue()):
		return nil, errors.New("vertex requires a valid vertex_key_config.project_id")
	case cfg.ProjectNumber.GetValue() != "" && !isVertexProjectNumber(cfg.ProjectNumber.GetValue()):
		return nil, errors.New("vertex_key_config.project_number must be digits")
	}
	for _, alias := range key.Aliases {
		if region := alias.Region.GetValue(); region != "" && !isVertexLocation(region) {
			return nil, errors.New("vertex alias regions must be valid Google Cloud locations")
		}
		if project := alias.ProjectID.GetValue(); project != "" && !isVertexProjectID(project) {
			return nil, errors.New("vertex alias project IDs must be valid Google Cloud project IDs")
		}
		if c := alias.VertexAliasCfg; c != nil {
			if project := c.ProjectID.GetValue(); project != "" && !isVertexProjectID(project) {
				return nil, errors.New("vertex alias project IDs must be valid Google Cloud project IDs")
			}
			if number := c.ProjectNumber.GetValue(); number != "" && !isVertexProjectNumber(number) {
				return nil, errors.New("vertex alias project numbers must be digits")
			}
		}
	}
	return provider, nil
}

// isBearerToken reports whether s can be sent as a bearer token as given: it holds no
// whitespace or control character, which could split or extend the Authorization header.
func isBearerToken(s string) bool {
	for i := 0; i < len(s); i++ {
		if c := s[i]; c <= ' ' || c == 0x7f {
			return false
		}
	}
	return true
}

// isVertexLocation reports whether s is shaped like a Google Cloud location ("us-central1",
// "europe-west4", "global", "us", "eu"): lowercase letters and digits in hyphen-separated runs.
func isVertexLocation(s string) bool {
	if s == "" || s[0] == '-' || s[len(s)-1] == '-' {
		return false
	}
	for i := 0; i < len(s); i++ {
		switch c := s[i]; {
		case c >= 'a' && c <= 'z', c >= '0' && c <= '9':
		case c == '-' && s[i-1] != '-':
		default:
			return false
		}
	}
	return true
}

// isVertexProjectID reports whether s is shaped like a Google Cloud project ID: lowercase
// letters, digits and hyphens, optionally behind a legacy "domain:" prefix whose domain may also
// hold dots. It never admits a character that ends or escapes a URL path segment.
func isVertexProjectID(s string) bool {
	domain, project, scoped := strings.Cut(s, ":")
	if !scoped {
		domain, project = "", s
	} else if domain == "" {
		return false
	}
	for i := 0; i < len(domain); i++ {
		if c := domain[i]; (c < 'a' || c > 'z') && (c < '0' || c > '9') && c != '-' && c != '.' {
			return false
		}
	}
	if project == "" || project[0] == '-' || project[len(project)-1] == '-' {
		return false
	}
	for i := 0; i < len(project); i++ {
		if c := project[i]; (c < 'a' || c > 'z') && (c < '0' || c > '9') && c != '-' {
			return false
		}
	}
	return true
}

// isVertexProjectNumber reports whether s is a non-empty run of ASCII digits.
func isVertexProjectNumber(s string) bool {
	return s != "" && schemas.IsAllDigitsASCII(s)
}
