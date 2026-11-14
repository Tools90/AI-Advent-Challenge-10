package main

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

const (
	StrategySliding = "sliding"
	StrategyFacts   = "facts"
	StrategySummary = "summary"

	WindowMessages = 10
	LastPairs      = 5
	SummaryEvery   = 5

	systemPrompt  = "Ты — полезный ассистент. Отвечай по-русски, кратко и по делу."
	factsPrompt   = "Ты обновляешь key-value память агента. Верни только JSON-объект с фактами: цель, ограничения, предпочтения, решения, договорённости. Сохраняй уже известные факты, добавляй новые и исправляй устаревшие. Без markdown."
	summaryPrompt = "Ты сжимаешь историю диалога. Сохрани все конкретные детали: имена, сроки, бюджеты, ограничения, принятые решения, технологии и географию. Не добавляй того, чего не было. Верни только краткое summary."
)

type Branch struct {
	ID               string         `json:"id"`
	Name             string         `json:"name"`
	ParentCheckpoint string         `json:"parentCheckpoint,omitempty"`
	Messages         []Message      `json:"messages"`
	Turns            []TurnMeta     `json:"turns,omitempty"`
	Facts            map[string]any `json:"facts,omitempty"`
	Summary          string         `json:"summary,omitempty"`
	SummarizedPairs  int            `json:"summarizedPairs,omitempty"`
	ChatUsage        Usage          `json:"chatUsage"`
	AuxUsage         Usage          `json:"auxUsage"`
	JudgeUsage       Usage          `json:"judgeUsage"`
	AuxCalls         int            `json:"auxCalls"`
	SummaryUpdates   int            `json:"summaryUpdates"`
	LastError        string         `json:"lastError,omitempty"`
}

type JudgeInfo struct {
	Correct bool   `json:"correct"`
	Reason  string `json:"reason,omitempty"`
}

type TurnMeta struct {
	Request          []Message  `json:"request,omitempty"`
	PromptTokens     int        `json:"promptTokens"`
	CompletionTokens int        `json:"completionTokens"`
	SessionTotal     int        `json:"sessionTotal"`
	Judge            *JudgeInfo `json:"judge,omitempty"`
}

type Checkpoint struct {
	ID        string  `json:"id"`
	CreatedAt string  `json:"createdAt"`
	BranchID  string  `json:"branchId"`
	Snapshot  *Branch `json:"snapshot"`
}

type StrategyState struct {
	ActiveBranch string                 `json:"activeBranch"`
	Branches     map[string]*Branch     `json:"branches"`
	Checkpoints  map[string]*Checkpoint `json:"checkpoints"`
}

type PersistedState struct {
	Version        int                       `json:"version"`
	ActiveStrategy string                    `json:"activeStrategy"`
	Strategies     map[string]*StrategyState `json:"strategies"`
}

type ChatResult struct {
	Strategy        string         `json:"strategy"`
	Branch          string         `json:"branch"`
	Reply           string         `json:"reply"`
	Request         []Message      `json:"request"`
	TurnUsage       Usage          `json:"turnUsage"`
	ChatUsage       Usage          `json:"chatUsage"`
	AuxUsage        Usage          `json:"auxUsage"`
	JudgeUsage      Usage          `json:"judgeUsage"`
	TotalUsage      Usage          `json:"totalUsage"`
	Facts           map[string]any `json:"facts,omitempty"`
	Summary         string         `json:"summary,omitempty"`
	Pairs           int            `json:"pairs"`
	SummarizedPairs int            `json:"summarizedPairs"`
	AuxCalls        int            `json:"auxCalls"`
	SummaryUpdates  int            `json:"summaryUpdates"`
}

type JudgeResult struct {
	Correct      bool   `json:"correct"`
	Reason       string `json:"reason"`
	Usage        Usage  `json:"usage"`
	SessionTotal Usage  `json:"sessionTotal"`
}

type ContextStrategy interface {
	ID() string
	BeforeUser(ctx context.Context, llm LLMClient, branch *Branch, user string) (Usage, int, error)
	Build(branch *Branch, user string) []Message
	AfterAssistant(ctx context.Context, llm LLMClient, branch *Branch) (Usage, int, error)
}

type SlidingStrategy struct{}

func (SlidingStrategy) ID() string { return StrategySliding }
func (SlidingStrategy) BeforeUser(context.Context, LLMClient, *Branch, string) (Usage, int, error) {
	return Usage{}, 0, nil
}
func (SlidingStrategy) Build(branch *Branch, user string) []Message {
	out := []Message{{Role: "system", Content: systemPrompt}}
	out = append(out, lastMessages(branch.Messages, WindowMessages)...)
	out = append(out, Message{Role: "user", Content: user})
	return out
}
func (SlidingStrategy) AfterAssistant(context.Context, LLMClient, *Branch) (Usage, int, error) {
	return Usage{}, 0, nil
}

type FactsStrategy struct{}

func (FactsStrategy) ID() string { return StrategyFacts }
func (FactsStrategy) BeforeUser(ctx context.Context, llm LLMClient, branch *Branch, user string) (Usage, int, error) {
	usage, err := updateFacts(ctx, llm, branch, user)
	return usage, 1, err
}
func (FactsStrategy) Build(branch *Branch, user string) []Message {
	system := systemPrompt
	if len(branch.Facts) > 0 {
		encoded, _ := json.MarshalIndent(branch.Facts, "", "  ")
		system += "\n\nИзвестные факты:\n" + string(encoded)
	}
	out := []Message{{Role: "system", Content: system}}
	out = append(out, lastMessages(branch.Messages, WindowMessages)...)
	out = append(out, Message{Role: "user", Content: user})
	return out
}
func (FactsStrategy) AfterAssistant(context.Context, LLMClient, *Branch) (Usage, int, error) {
	return Usage{}, 0, nil
}

type SummaryStrategy struct{}

func (SummaryStrategy) ID() string { return StrategySummary }
func (SummaryStrategy) BeforeUser(context.Context, LLMClient, *Branch, string) (Usage, int, error) {
	return Usage{}, 0, nil
}
func (SummaryStrategy) Build(branch *Branch, user string) []Message {
	system := systemPrompt
	if strings.TrimSpace(branch.Summary) != "" {
		system += "\n\nКраткое содержание предыдущей части диалога:\n" + branch.Summary
	}
	out := []Message{{Role: "system", Content: system}}
	pairs := len(branch.Messages) / 2
	lastStart := pairs - LastPairs
	if lastStart < 0 {
		lastStart = 0
	}
	for pair := lastStart; pair < pairs; pair++ {
		out = append(out, branch.Messages[pair*2], branch.Messages[pair*2+1])
	}
	out = append(out, Message{Role: "user", Content: user})
	return out
}
func (SummaryStrategy) AfterAssistant(ctx context.Context, llm LLMClient, branch *Branch) (Usage, int, error) {
	updated, usage, err := summarizeIfNeeded(ctx, llm, branch)
	if !updated {
		return usage, 0, err
	}
	return usage, 1, err
}

func strategyFor(id string) ContextStrategy {
	switch id {
	case StrategyFacts:
		return FactsStrategy{}
	case StrategySummary:
		return SummaryStrategy{}
	default:
		return SlidingStrategy{}
	}
}

func lastMessages(messages []Message, n int) []Message {
	if n <= 0 || len(messages) <= n {
		return append([]Message(nil), messages...)
	}
	return append([]Message(nil), messages[len(messages)-n:]...)
}

func updateFacts(ctx context.Context, llm LLMClient, branch *Branch, user string) (Usage, error) {
	current := branch.Facts
	if current == nil {
		current = map[string]any{}
	}
	encoded, _ := json.MarshalIndent(current, "", "  ")
	response, err := llm.Chat(ctx, []Message{
		{Role: "system", Content: factsPrompt},
		{Role: "user", Content: "Текущие facts:\n" + string(encoded) + "\n\nНовое сообщение пользователя:\n" + user},
	})
	if err != nil {
		return Usage{}, err
	}
	content := strings.TrimSpace(response.Message.Content)
	start := strings.Index(content, "{")
	end := strings.LastIndex(content, "}")
	if start < 0 || end < start {
		return response.Usage, fmt.Errorf("facts: не JSON: %s", content)
	}
	var facts map[string]any
	if err := json.Unmarshal([]byte(content[start:end+1]), &facts); err != nil {
		return response.Usage, fmt.Errorf("facts: %w", err)
	}
	if facts == nil {
		facts = map[string]any{}
	}
	branch.Facts = facts
	return response.Usage, nil
}

func summarizeIfNeeded(ctx context.Context, llm LLMClient, branch *Branch) (bool, Usage, error) {
	totalPairs := len(branch.Messages) / 2
	evicted := totalPairs - LastPairs
	if evicted < 0 {
		evicted = 0
	}
	if evicted-branch.SummarizedPairs < SummaryEvery {
		return false, Usage{}, nil
	}
	var b strings.Builder
	if strings.TrimSpace(branch.Summary) != "" {
		b.WriteString("Старое summary:\n")
		b.WriteString(branch.Summary)
		b.WriteString("\n\n")
	}
	b.WriteString("Новые вытесненные пары:\n")
	for pair := branch.SummarizedPairs; pair < evicted; pair++ {
		b.WriteString("Пользователь: ")
		b.WriteString(branch.Messages[pair*2].Content)
		b.WriteString("\nАссистент: ")
		b.WriteString(branch.Messages[pair*2+1].Content)
		b.WriteString("\n\n")
	}
	response, err := llm.Chat(ctx, []Message{
		{Role: "system", Content: summaryPrompt},
		{Role: "user", Content: b.String()},
	})
	if err != nil {
		return false, Usage{}, err
	}
	summary := strings.TrimSpace(response.Message.Content)
	if summary == "" {
		return false, response.Usage, fmt.Errorf("summary пустой")
	}
	branch.Summary = summary
	branch.SummarizedPairs = evicted
	branch.SummaryUpdates++
	return true, response.Usage, nil
}

func (b *Branch) totalUsage() Usage {
	var total Usage
	total.Add(b.ChatUsage)
	total.Add(b.AuxUsage)
	return total
}

func (b *Branch) pendingPairs() int {
	pairs := len(b.Messages) / 2
	evicted := pairs - LastPairs
	if evicted < 0 {
		evicted = 0
	}
	pending := evicted - b.SummarizedPairs
	if pending < 0 {
		return 0
	}
	return pending
}

func (b *Branch) judge(ctx context.Context, llm LLMClient, question, expected, answer string) (JudgeResult, error) {
	prompt := fmt.Sprintf(
		"Вопрос: %s\nОжидаемый факт: %s\nОтвет ассистента: %s\n\nПроверь, содержит ли ответ ожидаемый факт. Верни только JSON: {\"correct\": true|false, \"reason\": \"кратко\"}.",
		question, expected, answer,
	)
	response, err := llm.Chat(ctx, []Message{
		{Role: "system", Content: "Ты строгий судья ответов. correct=true, если в ответе присутствует ожидаемый факт, даже если ответ короче или перефразирован. Не требуй дословного совпадения. Примеры: expected «Виктор», answer «Виктор» → true; expected «15 марта», answer «дедлайн 15 марта» → true; expected «только iOS», answer «Android позже, на старте iOS» → true. Верни валидный JSON без markdown."},
		{Role: "user", Content: prompt},
	})
	if err != nil {
		return JudgeResult{}, err
	}
	b.JudgeUsage.Add(response.Usage)
	content := strings.TrimSpace(response.Message.Content)
	start := strings.Index(content, "{")
	end := strings.LastIndex(content, "}")
	if start < 0 || end < start {
		return JudgeResult{}, fmt.Errorf("судья вернул не JSON: %s", content)
	}
	var parsed struct {
		Correct bool   `json:"correct"`
		Reason  string `json:"reason"`
	}
	if err := json.Unmarshal([]byte(content[start:end+1]), &parsed); err != nil {
		return JudgeResult{}, fmt.Errorf("разбор ответа судьи: %w", err)
	}
	return JudgeResult{Correct: parsed.Correct, Reason: parsed.Reason, Usage: response.Usage, SessionTotal: b.totalUsage()}, nil
}

func sortedFacts(facts map[string]string) []string {
	keys := make([]string, 0, len(facts))
	for key := range facts {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}
