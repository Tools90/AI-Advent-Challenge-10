package main

import (
	"bufio"
	"context"
	"embed"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"sync"
	"syscall"
	"time"
)

//go:embed web/index.html
var webFS embed.FS

type config struct {
	keyFile string
	apiKey  string
	baseURL string
	model   string
	addr    string
}

type server struct {
	cfg    config
	store  *StateStore
	llm    LLMClient
	chatMu sync.Mutex
}

func main() {
	if err := loadDotEnv(".env"); err != nil {
		log.Printf("config: %v", err)
	}

	cfg := loadConfig()
	if cfg.apiKey == "" {
		log.Printf("deepseek: ключ не задан в %s", cfg.keyFile)
	}

	store, err := NewStateStore("data")
	if err != nil {
		log.Fatalf("store: %v", err)
	}
	llm := NewDeepSeekClient(cfg.baseURL, cfg.apiKey, cfg.model)
	s := &server{cfg: cfg, store: store, llm: llm}

	mux := http.NewServeMux()
	mux.HandleFunc("GET /", s.index)
	mux.HandleFunc("GET /healthz", s.health)
	mux.HandleFunc("GET /api/state", s.state)
	mux.HandleFunc("POST /api/chat", s.chat)
	mux.HandleFunc("POST /api/judge", s.judge)
	mux.HandleFunc("POST /api/strategy", s.setStrategy)
	mux.HandleFunc("POST /api/branch/switch", s.switchBranch)
	mux.HandleFunc("POST /api/checkpoint", s.createCheckpoint)
	mux.HandleFunc("POST /api/reset", s.reset)
	mux.HandleFunc("POST /api/clear", s.clearData)

	srv := &http.Server{
		Addr:              cfg.addr,
		Handler:           logRequests(mux),
		ReadHeaderTimeout: 10 * time.Second,
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	go func() {
		log.Printf("сервер запущен: http://localhost%s (модель: %s)", cfg.addr, cfg.model)
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Fatalf("server: %v", err)
		}
	}()

	<-ctx.Done()
	log.Println("останавливаюсь...")
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := srv.Shutdown(shutdownCtx); err != nil {
		log.Printf("shutdown: %v", err)
	}
}

func (s *server) index(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}
	data, err := webFS.ReadFile("web/index.html")
	if err != nil {
		http.Error(w, "index.html не найден", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Header().Set("Cache-Control", "no-store")
	_, _ = w.Write(data)
}

func (s *server) health(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{
		"status":     "ok",
		"model":      s.cfg.model,
		"configured": s.cfg.apiKey != "",
	})
}

func (s *server) state(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{
		"model":      s.cfg.model,
		"configured": s.cfg.apiKey != "",
		"state":      s.store.Get(),
	})
}

func (s *server) chat(w http.ResponseWriter, r *http.Request) {
	var body struct {
		Message  string `json:"message"`
		Strategy string `json:"strategy"`
		Branch   string `json:"branch"`
	}
	if err := decodeJSON(w, r, &body); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	message := strings.TrimSpace(body.Message)
	if message == "" {
		writeError(w, http.StatusBadRequest, "сообщение пустое")
		return
	}
	if len([]rune(message)) > 4000 {
		writeError(w, http.StatusBadRequest, "сообщение слишком длинное")
		return
	}

	s.chatMu.Lock()
	defer s.chatMu.Unlock()

	state := s.store.Get()
	strategyID, _, branch, err := pickBranch(state, body.Strategy, body.Branch)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Minute)
	defer cancel()
	strategy := strategyFor(strategyID)

	auxUsage, auxCalls, err := strategy.BeforeUser(ctx, s.llm, branch, message)
	if auxUsage.TotalTokens > 0 {
		branch.AuxUsage.Add(auxUsage)
		branch.AuxCalls += auxCalls
	}
	if err != nil {
		branch.LastError = err.Error()
		_ = s.store.Save(state)
		writeError(w, http.StatusBadGateway, err.Error())
		return
	}

	request := strategy.Build(branch, message)
	response, err := s.llm.Chat(ctx, request)
	if err != nil {
		writeError(w, http.StatusBadGateway, err.Error())
		return
	}
	if strings.TrimSpace(response.Message.Content) == "" {
		writeError(w, http.StatusBadGateway, "модель вернула пустой ответ")
		return
	}

	branch.Messages = append(branch.Messages,
		Message{Role: "user", Content: message},
		Message{Role: "assistant", Content: response.Message.Content},
	)
	branch.ChatUsage.Add(response.Usage)

	afterUsage, afterCalls, afterErr := strategy.AfterAssistant(ctx, s.llm, branch)
	if afterUsage.TotalTokens > 0 {
		branch.AuxUsage.Add(afterUsage)
		branch.AuxCalls += afterCalls
	}
	if afterErr != nil {
		branch.LastError = afterErr.Error()
	} else {
		branch.LastError = ""
	}
	branch.Turns = append(branch.Turns, TurnMeta{
		Request:          request,
		PromptTokens:     response.Usage.PromptTokens,
		CompletionTokens: response.Usage.CompletionTokens,
		SessionTotal:     branch.totalUsage().TotalTokens,
	})

	if err := s.store.Save(state); err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}

	writeJSON(w, http.StatusOK, ChatResult{
		Strategy:        strategyID,
		Branch:          branch.ID,
		Reply:           response.Message.Content,
		Request:         request,
		TurnUsage:       response.Usage,
		ChatUsage:       branch.ChatUsage,
		AuxUsage:        branch.AuxUsage,
		JudgeUsage:      branch.JudgeUsage,
		TotalUsage:      branch.totalUsage(),
		Facts:           branch.Facts,
		Summary:         branch.Summary,
		Pairs:           len(branch.Messages) / 2,
		SummarizedPairs: branch.SummarizedPairs,
		AuxCalls:        branch.AuxCalls,
		SummaryUpdates:  branch.SummaryUpdates,
	})
}

func (s *server) judge(w http.ResponseWriter, r *http.Request) {
	var body struct {
		Strategy string `json:"strategy"`
		Branch   string `json:"branch"`
		Question string `json:"question"`
		Expected string `json:"expected"`
		Answer   string `json:"answer"`
	}
	if err := decodeJSON(w, r, &body); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	s.chatMu.Lock()
	defer s.chatMu.Unlock()

	state := s.store.Get()
	_, _, branch, err := pickBranch(state, body.Strategy, body.Branch)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), 2*time.Minute)
	defer cancel()
	result, err := branch.judge(ctx, s.llm, body.Question, body.Expected, body.Answer)
	if err != nil {
		writeError(w, http.StatusBadGateway, err.Error())
		return
	}
	if len(branch.Turns) > 0 {
		last := &branch.Turns[len(branch.Turns)-1]
		last.Judge = &JudgeInfo{Correct: result.Correct, Reason: result.Reason}
		last.SessionTotal = branch.totalUsage().TotalTokens
	}
	if err := s.store.Save(state); err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, result)
}

func (s *server) setStrategy(w http.ResponseWriter, r *http.Request) {
	var body struct {
		Strategy string `json:"strategy"`
	}
	if err := decodeJSON(w, r, &body); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	if strategyFor(body.Strategy).ID() != body.Strategy {
		writeError(w, http.StatusBadRequest, "неизвестная стратегия")
		return
	}
	state := s.store.Get()
	state.ActiveStrategy = body.Strategy
	if err := s.store.Save(state); err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func (s *server) switchBranch(w http.ResponseWriter, r *http.Request) {
	var body struct {
		Strategy string `json:"strategy"`
		Branch   string `json:"branch"`
	}
	if err := decodeJSON(w, r, &body); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	state := s.store.Get()
	strategyState, ok := state.Strategies[body.Strategy]
	if !ok || strategyState.Branches[body.Branch] == nil {
		writeError(w, http.StatusBadRequest, "неизвестная ветка")
		return
	}
	strategyState.ActiveBranch = body.Branch
	state.ActiveStrategy = body.Strategy
	if err := s.store.Save(state); err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func (s *server) createCheckpoint(w http.ResponseWriter, r *http.Request) {
	var body struct {
		Strategy string `json:"strategy"`
	}
	if err := decodeJSON(w, r, &body); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	state := s.store.Get()
	strategyState, ok := state.Strategies[body.Strategy]
	if !ok {
		writeError(w, http.StatusBadRequest, "неизвестная стратегия")
		return
	}
	source := strategyState.Branches[strategyState.ActiveBranch]
	if source == nil {
		writeError(w, http.StatusBadRequest, "нет активной ветки")
		return
	}

	checkpointID := fmt.Sprintf("cp-%d", time.Now().UnixNano())
	snapshot := clone(*source)
	strategyState.Checkpoints[checkpointID] = &Checkpoint{
		ID:        checkpointID,
		CreatedAt: time.Now().Format(time.RFC3339),
		BranchID:  source.ID,
		Snapshot:  &snapshot,
	}

	number := len(strategyState.Branches)
	branchID := fmt.Sprintf("branch-%d", number)
	for strategyState.Branches[branchID] != nil {
		number++
		branchID = fmt.Sprintf("branch-%d", number)
	}
	branch := clone(snapshot)
	branch.ID = branchID
	branch.Name = branchDisplayName(body.Strategy, source, number)
	branch.ParentCheckpoint = checkpointID
	strategyState.Branches[branchID] = &branch

	if err := s.store.Save(state); err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"checkpoint": checkpointID,
		"branch":     branchID,
		"name":       branch.Name,
	})
}

func strategyLabel(id string) string {
	switch id {
	case StrategyFacts:
		return "Sticky Facts"
	case StrategySummary:
		return "Summary"
	default:
		return "Sliding Window"
	}
}

func branchDisplayName(strategyID string, source *Branch, number int) string {
	name := fmt.Sprintf("%d. %s", number, strategyLabel(strategyID))
	prompt := lastUserPrompt(source.Messages)
	if prompt == "" {
		return name
	}
	return name + ". " + shortenPrompt(prompt)
}

func lastUserPrompt(messages []Message) string {
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == "user" && strings.TrimSpace(messages[i].Content) != "" {
			return strings.TrimSpace(messages[i].Content)
		}
	}
	return ""
}

func shortenPrompt(prompt string) string {
	fields := strings.Fields(prompt)
	if len(fields) > 4 {
		return strings.Join(fields[:4], " ") + "..."
	}
	runes := []rune(prompt)
	if len(runes) > 48 {
		return strings.TrimSpace(string(runes[:48])) + "..."
	}
	return prompt
}

func (s *server) clearData(w http.ResponseWriter, r *http.Request) {
	s.chatMu.Lock()
	defer s.chatMu.Unlock()
	if err := s.store.Clear(); err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func (s *server) reset(w http.ResponseWriter, r *http.Request) {
	var body struct {
		Strategy string `json:"strategy"`
		Branch   string `json:"branch"`
	}
	if err := decodeJSON(w, r, &body); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	state := s.store.Get()
	if body.Strategy == "" {
		state = NewPersistedState()
	} else {
		strategyState, ok := state.Strategies[body.Strategy]
		if !ok {
			writeError(w, http.StatusBadRequest, "неизвестная стратегия")
			return
		}
		if body.Branch == "" {
			state.Strategies[body.Strategy] = newStrategyState()
		} else if branch := strategyState.Branches[body.Branch]; branch != nil {
			branch.Messages = nil
			branch.Turns = nil
			branch.Facts = map[string]any{}
			branch.Summary = ""
			branch.SummarizedPairs = 0
			branch.ChatUsage = Usage{}
			branch.AuxUsage = Usage{}
			branch.JudgeUsage = Usage{}
			branch.AuxCalls = 0
			branch.SummaryUpdates = 0
			branch.LastError = ""
		} else {
			writeError(w, http.StatusBadRequest, "неизвестная ветка")
			return
		}
	}
	if err := s.store.Save(state); err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func pickBranch(state PersistedState, strategyID, branchID string) (string, *StrategyState, *Branch, error) {
	if strategyID == "" {
		strategyID = state.ActiveStrategy
	}
	strategyState, ok := state.Strategies[strategyID]
	if !ok {
		return "", nil, nil, fmt.Errorf("неизвестная стратегия")
	}
	if branchID == "" {
		branchID = strategyState.ActiveBranch
	}
	branch := strategyState.Branches[branchID]
	if branch == nil {
		return "", nil, nil, fmt.Errorf("неизвестная ветка")
	}
	return strategyID, strategyState, branch, nil
}

func loadConfig() config {
	keyFile := envOr("DEEPSEEK_API_KEY_FILE", "api_key.txt")
	port := envOr("PORT", "8089")
	return config{
		keyFile: keyFile,
		apiKey:  readSecret(keyFile, "sk-your-key-here"),
		baseURL: strings.TrimRight(envOr("DEEPSEEK_BASE_URL", "https://api.deepseek.com"), "/"),
		model:   envOr("DEEPSEEK_MODEL", "deepseek-flash"),
		addr:    ":" + port,
	}
}

func readSecret(path, placeholder string) string {
	data, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	value := unquote(strings.TrimSpace(strings.TrimPrefix(string(data), "\ufeff")))
	if value == placeholder || strings.Contains(value, "your-") {
		return ""
	}
	return value
}

func envOr(key, fallback string) string {
	if value := strings.TrimSpace(os.Getenv(key)); value != "" {
		return value
	}
	return fallback
}

func decodeJSON(w http.ResponseWriter, r *http.Request, dst any) error {
	r.Body = http.MaxBytesReader(w, r.Body, 1<<20)
	if err := json.NewDecoder(r.Body).Decode(dst); err != nil {
		return fmt.Errorf("некорректный JSON: %w", err)
	}
	return nil
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]string{"error": msg})
}

func logRequests(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		next.ServeHTTP(w, r)
		log.Printf("%s %s %s", r.Method, r.URL.Path, time.Since(start).Round(time.Millisecond))
	})
}

func loadDotEnv(path string) error {
	f, err := os.Open(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return fmt.Errorf("файл %s не найден, использую переменные окружения", path)
		}
		return err
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(strings.TrimPrefix(scanner.Text(), "\ufeff"))
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		key, value, ok := strings.Cut(line, "=")
		if !ok {
			continue
		}
		key = strings.TrimSpace(strings.TrimPrefix(key, "export "))
		value = unquote(strings.TrimSpace(value))
		if key == "" {
			continue
		}
		if _, exists := os.LookupEnv(key); !exists {
			_ = os.Setenv(key, value)
		}
	}
	return scanner.Err()
}

func unquote(v string) string {
	if len(v) >= 2 {
		if (v[0] == '"' && v[len(v)-1] == '"') || (v[0] == '\'' && v[len(v)-1] == '\'') {
			return v[1 : len(v)-1]
		}
	}
	return v
}
