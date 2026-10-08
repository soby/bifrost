package handlers

import (
	"fmt"

	"github.com/maximhq/bifrost/core/schemas"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

var usageNullJSONSuffix = []byte(`,"usage":null}`)

// marshalChatCompletionStreamEvents converts an internal chat stream chunk
// into the OpenAI Chat Completions wire contract. Internal chunks may combine
// usage with a terminal choice. On the wire, usage is omitted unless requested
// and requested usage is sent as a separate final chunk with no choices.
//
// The common path performs one marshal and no extra allocation beyond the JSON
// output. The second marshal exists only for the single terminal chunk of an
// include_usage stream.
func marshalChatCompletionStreamEvents(response *schemas.BifrostChatResponse, includeUsage bool) (primaryJSON, usageJSON []byte, err error) {
	if response == nil {
		return nil, nil, fmt.Errorf("chat completion stream response is nil")
	}

	usage := response.Usage
	if usage == nil {
		primaryJSON, err = schemas.MarshalSorted(response)
		if err != nil {
			return nil, nil, err
		}
		if primaryJSON, err = omitAccountingChoiceNulls(response, primaryJSON); err != nil {
			return nil, nil, err
		}
		if includeUsage {
			primaryJSON = appendUsageNull(primaryJSON)
		}
		return primaryJSON, nil, nil
	}

	if len(response.Choices) > 0 {
		primary := *response
		primary.Usage = nil
		primaryJSON, err = schemas.MarshalSorted(&primary)
		if err != nil {
			return nil, nil, err
		}
		if primaryJSON, err = omitAccountingChoiceNulls(&primary, primaryJSON); err != nil {
			return nil, nil, err
		}
		if includeUsage {
			primaryJSON = appendUsageNull(primaryJSON)
		}
	}

	if !includeUsage {
		return primaryJSON, nil, nil
	}

	usageOnly := *response
	usageOnly.Choices = []schemas.BifrostResponseChoice{}
	usageJSON, err = schemas.MarshalSorted(&usageOnly)
	if err != nil {
		return nil, nil, err
	}
	return primaryJSON, usageJSON, nil
}

// appendUsageNull adds the spec-required null usage member to ordinary chunks
// when include_usage is enabled. sonic always emits a JSON object here. Keeping
// this as a direct append avoids a map conversion and its hot-path allocations.
func appendUsageNull(encoded []byte) []byte {
	if len(encoded) == 0 || encoded[len(encoded)-1] != '}' {
		return encoded
	}
	encoded = encoded[:len(encoded)-1]
	return append(encoded, usageNullJSONSuffix...)
}

// omitAccountingChoiceNulls keeps the post-finish accounting chunk on the wire
// shape released Pretxt NeMo Relay workers accept (PLATFORM-4090). After
// forwarding a content-bearing finish, the OpenAI stream producer sends one
// synthetic chunk whose only choice is content-free: index 0, an empty delta,
// and no finish reason or logprobs (providers/utils
// CreateBifrostChatCompletionChunkResponse with a nil finish reason). Upstream
// #6723 marshals finish_reason and logprobs as required nullable fields, which
// gave that chunk `"finish_reason":null,"logprobs":null`; earlier releases
// omitted both, and Relay's corrective collector accepts only the bare
// `{"delta":{},"index":0}` after a finish. This removes exactly those two nulls
// from exactly that choice. Every other chunk keeps upstream's encoding, and
// the struct pre-check keeps them on the single-marshal path.
func omitAccountingChoiceNulls(response *schemas.BifrostChatResponse, encoded []byte) ([]byte, error) {
	if len(response.Choices) != 1 {
		return encoded, nil
	}
	choice := response.Choices[0]
	if choice.Index != 0 || choice.FinishReason != nil || choice.LogProbs != nil ||
		choice.TextCompletionResponseChoice != nil || choice.ChatNonStreamResponseChoice != nil ||
		choice.ChatStreamResponseChoice == nil || !emptyStreamDelta(choice.ChatStreamResponseChoice.Delta) ||
		len(choice.ContentFilterResults) != 0 || len(choice.Error) != 0 {
		return encoded, nil
	}
	wire := gjson.GetBytes(encoded, "choices.0")
	if !wire.IsObject() || len(wire.Map()) != 4 || wire.Get("delta").Raw != "{}" ||
		wire.Get("finish_reason").Type != gjson.Null || wire.Get("logprobs").Type != gjson.Null {
		return encoded, nil
	}
	encoded, err := sjson.DeleteBytes(encoded, "choices.0.finish_reason")
	if err != nil {
		return nil, err
	}
	return sjson.DeleteBytes(encoded, "choices.0.logprobs")
}

func emptyStreamDelta(delta *schemas.ChatStreamResponseChoiceDelta) bool {
	return delta != nil && delta.Role == nil && delta.Content == nil && delta.Refusal == nil &&
		delta.Audio == nil && delta.Reasoning == nil && len(delta.ReasoningDetails) == 0 &&
		len(delta.Annotations) == 0 && len(delta.ToolCalls) == 0 && len(delta.ExtraContent) == 0
}
