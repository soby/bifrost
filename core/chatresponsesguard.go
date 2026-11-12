package bifrost

import (
	"strings"

	"github.com/maximhq/bifrost/core/schemas"
	"github.com/valyala/fasthttp"
)

// refuseChatParamsLostToResponses refuses a chat request that is converted to the
// Responses API (BifrostContextKeyChangeRequestType, set by a plugin such as compat's
// chat-to-responses conversion) while it sets chat parameters the Responses API has no
// equivalent for: the conversion would drop them and serve a different request than
// the caller sent. A raw-body request is left alone: its body is sent as is.
func refuseChatParamsLostToResponses(ctx *schemas.BifrostContext, req *schemas.BifrostChatRequest) *schemas.BifrostError {
	if req == nil || req.Params == nil {
		return nil
	}
	if useRawBody, _ := ctx.Value(schemas.BifrostContextKeyUseRawRequestBody).(bool); useRawBody {
		return nil
	}
	unsupported := chatParamsWithoutResponsesEquivalent(req.Params)
	if len(unsupported) == 0 {
		return nil
	}
	sc := fasthttp.StatusBadRequest
	return &schemas.BifrostError{
		IsBifrostError: false,
		StatusCode:     &sc,
		Error: &schemas.ErrorField{
			Type:    new("invalid_request_error"),
			Message: "this chat request is served by the Responses API, which does not support: " + strings.Join(unsupported, ", ") + "; remove them",
		},
	}
}
