package bedrock

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/maximhq/bifrost/core/schemas"
	"github.com/stretchr/testify/require"
)

// TestToolResultStatusFromIsError verifies that a tool message carrying
// IsError converts to a Converse toolResult with status "error", and that
// non-error results keep the "success" default. Before IsError existed on
// ChatToolMessage the status was hard-coded to "success", so failed tool
// calls replayed through Bedrock looked successful to the model.
func TestToolResultStatusFromIsError(t *testing.T) {
	msgs := []schemas.ChatMessage{
		{
			Role:            schemas.ChatMessageRoleTool,
			ChatToolMessage: &schemas.ChatToolMessage{ToolCallID: schemas.Ptr("toolu_failed"), IsError: schemas.Ptr(true)},
			Content:         &schemas.ChatMessageContent{ContentStr: schemas.Ptr("command exited with code 1")},
		},
		{
			Role:            schemas.ChatMessageRoleTool,
			ChatToolMessage: &schemas.ChatToolMessage{ToolCallID: schemas.Ptr("toolu_ok")},
			Content:         &schemas.ChatMessageContent{ContentStr: schemas.Ptr("done")},
		},
	}

	converted, err := convertToolMessages(context.Background(), "anthropic.claude-sonnet-4-5-20250929-v1:0", msgs)
	if err != nil {
		t.Fatalf("convert tool messages: %v", err)
	}

	var results []*BedrockToolResult
	for _, block := range converted.Content {
		if block.ToolResult != nil {
			results = append(results, block.ToolResult)
		}
	}
	if len(results) != 2 {
		t.Fatalf("expected 2 toolResult blocks, got %d", len(results))
	}

	if results[0].ToolUseID != "toolu_failed" {
		t.Fatalf("expected first toolResult for toolu_failed, got %s", results[0].ToolUseID)
	}
	if results[0].Status == nil || *results[0].Status != "error" {
		t.Fatalf("failed tool call must map to status \"error\", got %v", results[0].Status)
	}
	if results[1].Status == nil || *results[1].Status != "success" {
		t.Fatalf("non-error tool call must keep status \"success\", got %v", results[1].Status)
	}
}

// TestToolResultEmptyKeyChatSurface verifies that a string tool result whose
// JSON carries an empty-string object key converts to a text block on the
// chat-completions surface. Converse rejects such a document in the json
// field ("The format of the value at ...toolResult.content.0.json is
// invalid"), and convertToolMessages now shares tryParseJSONIntoContentBlock
// with the Responses path, so both surfaces get the same fallback.
func TestToolResultEmptyKeyChatSurface(t *testing.T) {
	payload := `{"success":{"fullSubtreeExtensionCounts":{"":2,".md":1}}}`
	msgs := []schemas.ChatMessage{
		{
			Role:            schemas.ChatMessageRoleTool,
			ChatToolMessage: &schemas.ChatToolMessage{ToolCallID: schemas.Ptr("toolu_emptykey")},
			Content:         &schemas.ChatMessageContent{ContentStr: schemas.Ptr(payload)},
		},
	}

	converted, err := convertToolMessages(context.Background(), "anthropic.claude-sonnet-4-5-20250929-v1:0", msgs)
	if err != nil {
		t.Fatalf("convert tool messages: %v", err)
	}
	if len(converted.Content) != 1 || converted.Content[0].ToolResult == nil {
		t.Fatalf("expected a single toolResult block, got %#v", converted.Content)
	}
	content := converted.Content[0].ToolResult.Content
	if len(content) != 1 {
		t.Fatalf("expected 1 tool result content block, got %d", len(content))
	}
	if content[0].JSON != nil {
		t.Fatalf("empty-key payload must not be sent as a json block, got %s", string(content[0].JSON))
	}
	if content[0].Text == nil || *content[0].Text != payload {
		t.Fatalf("expected text fallback carrying the original payload, got %v", content[0].Text)
	}
}

// TestToolResultSingleTextBlockMatchesString verifies that a tool message whose
// content is one text block renders the same toolResult content as the
// equivalent string, keeping the block's cache_control as the trailing
// cachePoint. Clients that mark only the last message with cache_control send
// a tool result as a one-block array on that turn and as a string on later
// turns; before the fix the block form was always text while the string form
// became json, so the prompt prefix changed and the cache missed every turn.
func TestToolResultSingleTextBlockMatchesString(t *testing.T) {
	const model = "anthropic.claude-sonnet-4-5-20250929-v1:0"
	cases := []struct {
		name   string
		output string
	}{
		{"object", `{"files":["a","b"],"n":2}`},
		{"array", `[1,2,3]`},
		{"primitive", `42`},
		{"plain text", `not json at all`},
		{"empty key falls back to text", `{"":1}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			str := schemas.ChatMessage{
				Role:            schemas.ChatMessageRoleTool,
				ChatToolMessage: &schemas.ChatToolMessage{ToolCallID: schemas.Ptr("call_1")},
				Content:         &schemas.ChatMessageContent{ContentStr: schemas.Ptr(tc.output)},
			}
			blk := schemas.ChatMessage{
				Role:            schemas.ChatMessageRoleTool,
				ChatToolMessage: &schemas.ChatToolMessage{ToolCallID: schemas.Ptr("call_1")},
				Content: &schemas.ChatMessageContent{ContentBlocks: []schemas.ChatContentBlock{{
					Type:         schemas.ChatContentBlockTypeText,
					Text:         schemas.Ptr(tc.output),
					CacheControl: &schemas.CacheControl{Type: schemas.CacheControlTypeEphemeral},
				}}},
			}

			strMsg, err := convertToolMessages(context.Background(), model, []schemas.ChatMessage{str})
			if err != nil {
				t.Fatalf("convert string tool message: %v", err)
			}
			blkMsg, err := convertToolMessages(context.Background(), model, []schemas.ChatMessage{blk})
			if err != nil {
				t.Fatalf("convert one-block tool message: %v", err)
			}

			if len(strMsg.Content) != 1 || strMsg.Content[0].ToolResult == nil {
				t.Fatalf("string form: expected a single toolResult block, got %#v", strMsg.Content)
			}
			if len(blkMsg.Content) != 2 || blkMsg.Content[0].ToolResult == nil || blkMsg.Content[1].CachePoint == nil {
				t.Fatalf("one-block form: expected toolResult followed by cachePoint, got %#v", blkMsg.Content)
			}

			strBytes, err := json.Marshal(strMsg.Content[0])
			if err != nil {
				t.Fatalf("marshal string toolResult: %v", err)
			}
			blkBytes, err := json.Marshal(blkMsg.Content[0])
			if err != nil {
				t.Fatalf("marshal one-block toolResult: %v", err)
			}
			if string(strBytes) != string(blkBytes) {
				t.Fatalf("one-block toolResult must match the string form\nstring:    %s\none-block: %s", strBytes, blkBytes)
			}
		})
	}
}

// TestToolResultMultiTextBlocksStayText verifies that tool content with more
// than one text block keeps the per-block text rendering; only the single-block
// form is normalized to the string path.
func TestToolResultMultiTextBlocksStayText(t *testing.T) {
	msgs := []schemas.ChatMessage{{
		Role:            schemas.ChatMessageRoleTool,
		ChatToolMessage: &schemas.ChatToolMessage{ToolCallID: schemas.Ptr("call_multi")},
		Content: &schemas.ChatMessageContent{ContentBlocks: []schemas.ChatContentBlock{
			{Type: schemas.ChatContentBlockTypeText, Text: schemas.Ptr(`{"a":1}`)},
			{Type: schemas.ChatContentBlockTypeText, Text: schemas.Ptr(`{"b":2}`)},
		}},
	}}

	converted, err := convertToolMessages(context.Background(), "anthropic.claude-sonnet-4-5-20250929-v1:0", msgs)
	if err != nil {
		t.Fatalf("convert tool messages: %v", err)
	}
	if len(converted.Content) != 1 || converted.Content[0].ToolResult == nil {
		t.Fatalf("expected a single toolResult block, got %#v", converted.Content)
	}
	content := converted.Content[0].ToolResult.Content
	if len(content) != 2 {
		t.Fatalf("expected 2 tool result content blocks, got %d", len(content))
	}
	for i, b := range content {
		if b.JSON != nil || b.Text == nil {
			t.Fatalf("block %d: multi-block content must stay text, got %#v", i, b)
		}
	}
}

// BenchmarkConvertToolMessagesSingleTextBlock measures the one-text-block tool
// result path against the string path it now shares.
func BenchmarkConvertToolMessagesSingleTextBlock(b *testing.B) {
	const model = "anthropic.claude-sonnet-4-5-20250929-v1:0"
	out := `{"files":["a","b","c","d"],"n":4,"meta":{"truncated":false}}`
	forms := map[string]*schemas.ChatMessageContent{
		"string":    {ContentStr: schemas.Ptr(out)},
		"one-block": {ContentBlocks: []schemas.ChatContentBlock{{Type: schemas.ChatContentBlockTypeText, Text: schemas.Ptr(out), CacheControl: &schemas.CacheControl{Type: schemas.CacheControlTypeEphemeral}}}},
	}
	for name, content := range forms {
		msgs := []schemas.ChatMessage{{
			Role:            schemas.ChatMessageRoleTool,
			ChatToolMessage: &schemas.ChatToolMessage{ToolCallID: schemas.Ptr("call_1")},
			Content:         content,
		}}
		b.Run(name, func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				if _, err := convertToolMessages(context.Background(), model, msgs); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// Converse takes images in a toolResult; audio a tool returned becomes a placeholder beside them.
func TestToolResultKeepsImagesNamesOtherMedia(t *testing.T) {
	converted, err := convertToolMessages(context.Background(), "anthropic.claude-sonnet-4-5-20250929-v1:0", []schemas.ChatMessage{{
		Role:            schemas.ChatMessageRoleTool,
		ChatToolMessage: &schemas.ChatToolMessage{ToolCallID: schemas.Ptr("toolu_1")},
		Content:         &schemas.ChatMessageContent{ContentBlocks: []schemas.ChatContentBlock{
		{Type: schemas.ChatContentBlockTypeText, Text: schemas.Ptr("chart below")},
		{Type: schemas.ChatContentBlockTypeImage, ImageURLStruct: &schemas.ChatInputImage{URL: "data:image/png;base64,iVBORw0KGgo="}},
		{Type: schemas.ChatContentBlockTypeInputAudio, InputAudio: &schemas.ChatInputAudio{Data: "SUQz", Format: schemas.Ptr("mp3")}},
	}},
	}})
	if err != nil {
		t.Fatalf("convert tool messages: %v", err)
	}
	var images, placeholders int
	for _, block := range converted.Content {
		if block.ToolResult == nil {
			continue
		}
		for _, inner := range block.ToolResult.Content {
			if inner.Image != nil {
				images++
			}
			if inner.Text != nil && strings.Contains(*inner.Text, "[audio/mp3") {
				placeholders++
			}
		}
	}
	if images != 1 || placeholders != 1 {
		t.Fatalf("expected the image kept and the audio named, got %d images, %d placeholders", images, placeholders)
	}
}

// Converse requires user and assistant turns to alternate. The Chat converter folds a run of
// tool messages into one user turn of toolResult blocks; a user message that follows them used
// to become a second, adjacent user turn, which Converse rejects with a ValidationException.

func toolResultsThenUserMessages(after ...schemas.ChatMessage) []schemas.ChatMessage {
	call := func(id string) schemas.ChatAssistantMessageToolCall {
		return schemas.ChatAssistantMessageToolCall{
			ID:       schemas.Ptr(id),
			Type:     schemas.Ptr("function"),
			Function: schemas.ChatAssistantMessageToolCallFunction{Name: schemas.Ptr("issue_refund"), Arguments: `{"order":"` + id + `"}`},
		}
	}
	tool := func(id string) schemas.ChatMessage {
		return schemas.ChatMessage{
			Role:            schemas.ChatMessageRoleTool,
			Content:         &schemas.ChatMessageContent{ContentStr: schemas.Ptr("done " + id)},
			ChatToolMessage: &schemas.ChatToolMessage{ToolCallID: schemas.Ptr(id)},
		}
	}
	msgs := []schemas.ChatMessage{
		{Role: schemas.ChatMessageRoleUser, Content: &schemas.ChatMessageContent{ContentStr: schemas.Ptr("Refund orders 1 and 2")}},
		{
			Role:                 schemas.ChatMessageRoleAssistant,
			ChatAssistantMessage: &schemas.ChatAssistantMessage{ToolCalls: []schemas.ChatAssistantMessageToolCall{call("c1"), call("c2")}},
		},
		tool("c1"),
		tool("c2"),
	}
	return append(msgs, after...)
}

func requireToolResultTurn(t *testing.T, msgs []BedrockMessage) {
	t.Helper()
	require.Equal(t, BedrockMessageRoleAssistant, msgs[1].Role)
	require.Len(t, msgs[1].Content, 2)
	require.NotNil(t, msgs[1].Content[0].ToolUse)
	require.NotNil(t, msgs[1].Content[1].ToolUse)
	require.Equal(t, "c1", msgs[1].Content[0].ToolUse.ToolUseID)
	require.Equal(t, "c2", msgs[1].Content[1].ToolUse.ToolUseID)

	require.Equal(t, BedrockMessageRoleUser, msgs[2].Role)
	require.GreaterOrEqual(t, len(msgs[2].Content), 2)
	require.NotNil(t, msgs[2].Content[0].ToolResult)
	require.NotNil(t, msgs[2].Content[1].ToolResult)
	require.Equal(t, "c1", msgs[2].Content[0].ToolResult.ToolUseID)
	require.Equal(t, "c2", msgs[2].Content[1].ToolResult.ToolUseID)
}

func requireAlternatingRoles(t *testing.T, msgs []BedrockMessage) {
	t.Helper()
	for i := 1; i < len(msgs); i++ {
		require.NotEqual(t, msgs[i-1].Role, msgs[i].Role, "turns %d and %d share role %s", i-1, i, msgs[i].Role)
	}
}

func TestConvertMessages_UserMessageAfterToolResultsJoinsTheirTurn(t *testing.T) {
	msgs, _, err := convertMessages(t.Context(), "anthropic.claude-sonnet-5", toolResultsThenUserMessages(
		schemas.ChatMessage{Role: schemas.ChatMessageRoleUser, Content: &schemas.ChatMessageContent{ContentStr: schemas.Ptr("Make one refund call per response.")}},
	))
	require.NoError(t, err)
	require.Len(t, msgs, 3, "user, assistant[toolUse c1, toolUse c2], user[toolResult c1, toolResult c2, text]")
	requireAlternatingRoles(t, msgs)
	requireToolResultTurn(t, msgs)
	require.Len(t, msgs[2].Content, 3)
	require.NotNil(t, msgs[2].Content[2].Text)
	require.Equal(t, "Make one refund call per response.", *msgs[2].Content[2].Text)
}

// A mid-conversation system reminder after the tool results already folds into their turn; the
// user message that follows joins the same turn behind it, in input order.
func TestConvertMessages_UserMessageAfterToolResultsAndReminderJoinsTheirTurn(t *testing.T) {
	msgs, _, err := convertMessages(t.Context(), "anthropic.claude-sonnet-5", toolResultsThenUserMessages(
		schemas.ChatMessage{Role: schemas.ChatMessageRoleSystem, Content: &schemas.ChatMessageContent{ContentStr: schemas.Ptr("Refunds need approval.")}},
		schemas.ChatMessage{Role: schemas.ChatMessageRoleUser, Content: &schemas.ChatMessageContent{ContentStr: schemas.Ptr("Make one refund call per response.")}},
	))
	require.NoError(t, err)
	require.Len(t, msgs, 3)
	requireAlternatingRoles(t, msgs)
	requireToolResultTurn(t, msgs)
	last := msgs[2].Content[len(msgs[2].Content)-1]
	require.NotNil(t, last.Text)
	require.Equal(t, "Make one refund call per response.", *last.Text)
	reminderAt, textAt := -1, -1
	for i, block := range msgs[2].Content {
		if block.Text != nil && *block.Text != "Make one refund call per response." && reminderAt < 0 {
			reminderAt = i
		}
		if block.Text != nil && *block.Text == "Make one refund call per response." {
			textAt = i
		}
	}
	require.Greater(t, reminderAt, 1, "the reminder trails the toolResult blocks")
	require.Greater(t, textAt, reminderAt, "the user text follows the reminder")
}

// A user turn that does not follow tool results is untouched: user, user stays as it was
// (that shape is the caller's to send).
func TestConvertMessages_UserMessageAfterPlainUserTurnIsNotMerged(t *testing.T) {
	msgs, _, err := convertMessages(t.Context(), "anthropic.claude-sonnet-5", []schemas.ChatMessage{
		{Role: schemas.ChatMessageRoleUser, Content: &schemas.ChatMessageContent{ContentStr: schemas.Ptr("first")}},
		{Role: schemas.ChatMessageRoleUser, Content: &schemas.ChatMessageContent{ContentStr: schemas.Ptr("second")}},
	})
	require.NoError(t, err)
	require.Len(t, msgs, 2)
}
