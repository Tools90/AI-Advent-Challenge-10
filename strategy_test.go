package main

import (
	"context"
	"fmt"
	"strings"
	"testing"
)

type fakeLLM struct {
	calls    int
	requests [][]Message
}

func (f *fakeLLM) Chat(ctx context.Context, messages []Message) (LLMResponse, error) {
	f.calls++
	f.requests = append(f.requests, append([]Message(nil), messages...))
	content := fmt.Sprintf("reply-%d", f.calls)
	if len(messages) > 0 && messages[0].Role == "system" {
		system := messages[0].Content
		if strings.Contains(system, "key-value память") {
			content = `{"name":"Виктор","budget":"2,5 млн"}`
		} else if strings.Contains(system, "сжимаешь историю") {
			content = "SUMMARY"
		} else if strings.Contains(system, "судья") {
			content = `{"correct":true,"reason":"ok"}`
		}
	}
	return LLMResponse{
		Message: Message{Role: "assistant", Content: content},
		Usage:   Usage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15},
	}, nil
}

func (f *fakeLLM) Model() string { return "fake" }

func addPairs(branch *Branch, count int) {
	for i := 0; i < count; i++ {
		branch.Messages = append(branch.Messages,
			Message{Role: "user", Content: fmt.Sprintf("u%d", i)},
			Message{Role: "assistant", Content: fmt.Sprintf("a%d", i)},
		)
	}
}

func TestSlidingWindowBuild(t *testing.T) {
	branch := &Branch{Facts: map[string]any{}}
	addPairs(branch, 15)
	request := (SlidingStrategy{}).Build(branch, "next")
	if len(request) != 12 {
		t.Fatalf("request messages = %d, want 12", len(request))
	}
	if request[1].Content != "u10" {
		t.Errorf("first history = %q, want u10", request[1].Content)
	}
	if request[len(request)-1].Content != "next" {
		t.Errorf("last message = %q", request[len(request)-1].Content)
	}
}

func TestFactsUpdateAndBuild(t *testing.T) {
	branch := &Branch{Facts: map[string]any{}}
	llm := &fakeLLM{}
	usage, calls, err := (FactsStrategy{}).BeforeUser(context.Background(), llm, branch, "Меня зовут Виктор")
	if err != nil {
		t.Fatal(err)
	}
	if calls != 1 || usage.TotalTokens == 0 {
		t.Errorf("calls=%d usage=%+v", calls, usage)
	}
	if branch.Facts["name"] != "Виктор" || branch.Facts["budget"] != "2,5 млн" {
		t.Errorf("facts = %+v", branch.Facts)
	}
	request := (FactsStrategy{}).Build(branch, "Какой бюджет?")
	if !strings.Contains(request[0].Content, "Виктор") {
		t.Errorf("facts not in system: %q", request[0].Content)
	}
}

func TestSummaryUpdateAndBuild(t *testing.T) {
	branch := &Branch{Facts: map[string]any{}}
	addPairs(branch, 10)
	llm := &fakeLLM{}
	usage, calls, err := (SummaryStrategy{}).AfterAssistant(context.Background(), llm, branch)
	if err != nil {
		t.Fatal(err)
	}
	if calls != 1 || usage.TotalTokens == 0 || branch.Summary != "SUMMARY" || branch.SummarizedPairs != 5 {
		t.Errorf("calls=%d usage=%+v summary=%q pairs=%d", calls, usage, branch.Summary, branch.SummarizedPairs)
	}
	request := (SummaryStrategy{}).Build(branch, "next")
	if len(request) != 12 {
		t.Fatalf("request messages = %d, want 12", len(request))
	}
	if !strings.Contains(request[0].Content, "SUMMARY") {
		t.Errorf("summary not in system: %q", request[0].Content)
	}
	if request[1].Content != "u5" {
		t.Errorf("first history = %q, want u5", request[1].Content)
	}
}

func TestBranchJudge(t *testing.T) {
	branch := &Branch{Facts: map[string]any{}}
	result, err := branch.judge(context.Background(), &fakeLLM{}, "Как зовут?", "Виктор", "Виктор")
	if err != nil {
		t.Fatal(err)
	}
	if !result.Correct || result.Usage.TotalTokens == 0 {
		t.Errorf("judge = %+v", result)
	}
}

func TestStateStorePersistence(t *testing.T) {
	dir := t.TempDir()
	store, err := NewStateStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	state := store.Get()
	branch := state.Strategies[StrategyFacts].Branches["main"]
	branch.Messages = []Message{{Role: "user", Content: "hello"}}
	branch.Facts["name"] = "Виктор"
	if err := store.Save(state); err != nil {
		t.Fatal(err)
	}
	reloaded, err := NewStateStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	got := reloaded.Get().Strategies[StrategyFacts].Branches["main"]
	if len(got.Messages) != 1 || got.Facts["name"] != "Виктор" {
		t.Errorf("reloaded branch = %+v", got)
	}
}

func TestBranchDisplayName(t *testing.T) {
	branch := &Branch{Messages: []Message{{Role: "user", Content: "Как дела у тебя сегодня?"}}}
	got := branchDisplayName(StrategySliding, branch, 1)
	want := "1. Sliding Window. Как дела у тебя..."
	if got != want {
		t.Errorf("branch name = %q, want %q", got, want)
	}
	if short := shortenPrompt("короткий вопрос"); short != "короткий вопрос" {
		t.Errorf("short prompt = %q", short)
	}
}
