package server

import (
	"context"
	"testing"

	"github.com/maximhq/bifrost/core/schemas"
	"github.com/maximhq/bifrost/transports/bifrost-http/handlers"
	"github.com/maximhq/bifrost/transports/bifrost-http/lib"
	"github.com/valyala/fasthttp"
)

// Without a config store there is no admin account, so /api stays behind the setup lock:
// closed to every call without the setup token, and open to one carrying it, as the local
// admin with dangerous changes still refused.
func TestStorelessAPIMiddlewaresKeepTheSetupLock(t *testing.T) {
	handlers.SetLogger(noopTestLogger{})
	type outcome struct {
		status            int
		reached, bypassed bool
	}
	call := func(ctx context.Context, setupToken, path, header string) outcome {
		var got outcome
		handler := lib.ChainMiddlewares(func(request *fasthttp.RequestCtx) {
			got.reached = true
			got.bypassed, _ = request.UserValue(schemas.BifrostContextKeyAuthBypassed).(bool)
		}, storelessAPIMiddlewares(ctx, setupToken)...)
		request := &fasthttp.RequestCtx{}
		request.Request.SetRequestURI(path)
		if header != "" {
			request.Request.Header.Set(handlers.SetupTokenHeader, header)
		}
		handler(request)
		got.status = request.Response.StatusCode()
		return got
	}
	oss := context.Background()
	for _, tt := range []struct {
		name, setupToken, path, header string
		want                           outcome
	}{
		{name: "no setup token configured", path: "/api/config", want: outcome{status: fasthttp.StatusForbidden}},
		{name: "no setup token sent", setupToken: "operator-token", path: "/api/config", want: outcome{status: fasthttp.StatusUnauthorized}},
		{name: "wrong setup token", setupToken: "operator-token", path: "/api/config", header: "guess", want: outcome{status: fasthttp.StatusForbidden}},
		{name: "setup token", setupToken: "operator-token", path: "/api/config", header: "operator-token", want: outcome{status: fasthttp.StatusOK, reached: true, bypassed: true}},
		{name: "public route", path: "/api/session/is-auth-enabled", want: outcome{status: fasthttp.StatusOK, reached: true, bypassed: true}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if got := call(oss, tt.setupToken, tt.path, tt.header); got != tt.want {
				t.Fatalf("got %+v, want %+v", got, tt.want)
			}
		})
	}
	// Enterprise builds install their own chain and never the OSS setup lock.
	enterprise := context.WithValue(oss, schemas.BifrostContextKeyIsEnterprise, true)
	if got := call(enterprise, "", "/api/config", ""); got != (outcome{status: fasthttp.StatusOK, reached: true, bypassed: true}) {
		t.Fatalf("enterprise got %+v, want the bypass marker only", got)
	}
}
