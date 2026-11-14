package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"
)

type StateStore struct {
	mu    sync.RWMutex
	path  string
	state PersistedState
}

func NewStateStore(dir string) (*StateStore, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("создание %s: %w", dir, err)
	}
	store := &StateStore{path: filepath.Join(dir, "state.json")}
	data, err := os.ReadFile(store.path)
	if err != nil {
		if !errors.Is(err, os.ErrNotExist) {
			return nil, fmt.Errorf("чтение %s: %w", store.path, err)
		}
		store.state = NewPersistedState()
		if err := store.saveLocked(); err != nil {
			return nil, err
		}
		return store, nil
	}
	if err := json.Unmarshal(data, &store.state); err != nil {
		return nil, fmt.Errorf("разбор %s: %w", store.path, err)
	}
	store.normalize()
	return store, nil
}

func NewPersistedState() PersistedState {
	state := PersistedState{
		Version:        1,
		ActiveStrategy: StrategySliding,
		Strategies:     map[string]*StrategyState{},
	}
	for _, id := range []string{StrategySliding, StrategyFacts, StrategySummary} {
		state.Strategies[id] = newStrategyState()
	}
	return state
}

func newStrategyState() *StrategyState {
	return &StrategyState{
		ActiveBranch: "main",
		Branches: map[string]*Branch{
			"main": {
				ID:    "main",
				Name:  "Main",
				Facts: map[string]any{},
			},
		},
		Checkpoints: map[string]*Checkpoint{},
	}
}

func (s *StateStore) normalize() {
	if s.state.Version == 0 {
		s.state.Version = 1
	}
	if s.state.Strategies == nil {
		s.state.Strategies = map[string]*StrategyState{}
	}
	for _, id := range []string{StrategySliding, StrategyFacts, StrategySummary} {
		if s.state.Strategies[id] == nil {
			s.state.Strategies[id] = newStrategyState()
		}
	}
	if s.state.ActiveStrategy == "" {
		s.state.ActiveStrategy = StrategySliding
	}
	for _, strategy := range s.state.Strategies {
		if strategy.Branches == nil {
			strategy.Branches = map[string]*Branch{}
		}
		if strategy.Checkpoints == nil {
			strategy.Checkpoints = map[string]*Checkpoint{}
		}
		if len(strategy.Branches) == 0 {
			strategy.Branches["main"] = &Branch{ID: "main", Name: "Main", Facts: map[string]any{}}
		}
		if strategy.ActiveBranch == "" || strategy.Branches[strategy.ActiveBranch] == nil {
			strategy.ActiveBranch = "main"
		}
		for _, branch := range strategy.Branches {
			if branch.Facts == nil {
				branch.Facts = map[string]any{}
			}
		}
	}
}

func (s *StateStore) Get() PersistedState {
	s.mu.RLock()
	defer s.mu.RUnlock()
	state := clone(s.state)
	for _, strategy := range state.Strategies {
		for _, branch := range strategy.Branches {
			if branch.Facts == nil {
				branch.Facts = map[string]any{}
			}
		}
	}
	return state
}

func (s *StateStore) Save(state PersistedState) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.state = clone(state)
	return s.saveLocked()
}

func (s *StateStore) Clear() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.state = NewPersistedState()
	if err := os.Remove(s.path); err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	return nil
}

func (s *StateStore) saveLocked() error {
	data, err := json.MarshalIndent(s.state, "", "  ")
	if err != nil {
		return err
	}
	tmp := s.path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o600); err != nil {
		return err
	}
	_ = os.Remove(s.path)
	return os.Rename(tmp, s.path)
}

func clone[T any](v T) T {
	data, err := json.Marshal(v)
	if err != nil {
		return v
	}
	var out T
	if err := json.Unmarshal(data, &out); err != nil {
		return v
	}
	return out
}
