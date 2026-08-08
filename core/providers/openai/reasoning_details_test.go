package openai

import (
	"testing"

	"github.com/bytedance/sonic"
	"github.com/maximhq/bifrost/core/schemas"
	"github.com/stretchr/testify/require"
)

// TestReasoningDetails_ParsedFromIncomingRequest reproduces GH #5274: an
// OpenAI-compatible assistant message carrying OpenRouter-style reasoning_details
// must survive unmarshal and reach the
// Bifrost schema, not be silently dropped.
func TestReasoningDetails_ParsedFromIncomingRequest(t *testing.T) {
	raw := []byte(`{
		"model": "anthropic/claude-sonnet-4-5",
		"messages": [
			{"role": "user", "content": "What is the weather in Paris?"},
			{"role": "assistant", "content": "",
			 "tool_calls": [{"id": "toolu_1", "type": "function",
				"function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}}],
			 "reasoning_details": [{"index": 0, "type": "reasoning.text",
				"text": "thinking about Paris weather", "signature": "Eu8Bsig"}]},
			{"role": "tool", "tool_call_id": "toolu_1", "content": "22C, sunny"}
		]
	}`)

	var req OpenAIChatRequest
	require.NoError(t, sonic.Unmarshal(raw, &req))

	require.Len(t, req.Messages, 3)
	assistantMsg := req.Messages[1]
	require.NotNil(t, assistantMsg.OpenAIChatAssistantMessage)
	require.Len(t, assistantMsg.OpenAIChatAssistantMessage.ReasoningDetails, 1)
	detail := assistantMsg.OpenAIChatAssistantMessage.ReasoningDetails[0]
	require.Equal(t, schemas.BifrostReasoningDetailsTypeText, detail.Type)
	require.NotNil(t, detail.Signature)
	require.Equal(t, "Eu8Bsig", *detail.Signature)

	bifrostMessages := ConvertOpenAIMessagesToBifrostMessages(req.Messages)
	require.Len(t, bifrostMessages, 3)
	require.NotNil(t, bifrostMessages[1].ChatAssistantMessage)
	require.Len(t, bifrostMessages[1].ChatAssistantMessage.ReasoningDetails, 1)
	require.Equal(t, "Eu8Bsig", *bifrostMessages[1].ChatAssistantMessage.ReasoningDetails[0].Signature)
}

// TestReasoningDetails_PreservesPlainReasoningContent covers plain reasoning
// without inventing a structured, unsigned detail.
func TestReasoningDetails_PreservesPlainReasoningContent(t *testing.T) {
	text := "plain reasoning text"
	messages := []OpenAIMessage{
		{
			Role: schemas.ChatMessageRoleAssistant,
			OpenAIChatAssistantMessage: &OpenAIChatAssistantMessage{
				Reasoning: &text,
			},
		},
	}

	bifrostMessages := ConvertOpenAIMessagesToBifrostMessages(messages)
	require.Len(t, bifrostMessages, 1)
	require.NotNil(t, bifrostMessages[0].ChatAssistantMessage)
	require.Equal(t, text, *bifrostMessages[0].ChatAssistantMessage.Reasoning)
	require.Empty(t, bifrostMessages[0].ChatAssistantMessage.ReasoningDetails)
}
