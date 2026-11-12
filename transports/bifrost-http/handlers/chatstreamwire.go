package handlers

import (
	"fmt"

	"github.com/maximhq/bifrost/core/schemas"
)

// chatChunkWithoutUsage and chatChunkWithNullUsage marshal a chat stream chunk with its
// usage member left out or set to null. The outer Usage field shadows the embedded
// response's, so the response itself is never modified and accounting still sees its usage.
type chatChunkWithoutUsage struct {
	*schemas.BifrostChatResponse
	Usage *struct{} `json:"usage,omitempty"`
}

type chatChunkWithNullUsage struct {
	*schemas.BifrostChatResponse
	Usage *struct{} `json:"usage"`
}

// marshalChatCompletionStreamEvents converts an internal chat stream chunk into the OpenAI
// Chat Completions wire contract for stream_options.include_usage. Internal chunks may
// combine usage with a terminal choice. On the wire:
//
//   - without include_usage, no chunk carries a usage member, and a chunk that holds nothing
//     but usage is not sent (primaryJSON is nil);
//   - with it, every ordinary chunk carries "usage": null and the usage is sent as its own
//     chunk with an empty choices array (usageJSON), after the chunk it arrived with.
func marshalChatCompletionStreamEvents(response *schemas.BifrostChatResponse, includeUsage bool) (primaryJSON, usageJSON []byte, err error) {
	if response == nil {
		return nil, nil, fmt.Errorf("chat completion stream response is nil")
	}

	if response.Usage == nil || len(response.Choices) > 0 {
		if includeUsage {
			primaryJSON, err = schemas.MarshalSorted(chatChunkWithNullUsage{BifrostChatResponse: response})
		} else {
			primaryJSON, err = schemas.MarshalSorted(chatChunkWithoutUsage{BifrostChatResponse: response})
		}
		if err != nil {
			return nil, nil, err
		}
	}
	if response.Usage == nil || !includeUsage {
		return primaryJSON, nil, nil
	}

	usageOnly := *response
	usageOnly.Choices = []schemas.BifrostResponseChoice{}
	usageJSON, err = schemas.MarshalSorted(usageOnly.WithPreservedUsage())
	if err != nil {
		return nil, nil, err
	}
	return primaryJSON, usageJSON, nil
}
