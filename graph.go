package main

import (
	"fmt"
	"net/http"
	"sort"
	"strings"
	"sync"
)

// Status is the lifecycle state of a task.
type Status string

const (
	StatusPending Status = "pending" // some dependency is missing or not done
	StatusReady   Status = "ready"   // all dependencies exist and are done, not yet completed
	StatusDone    Status = "done"    // completed
)

// Task is a named unit of work that may depend on other tasks by name.
type Task struct {
	Name      string   `json:"name"`
	DependsOn []string `json:"dependsOn"`
	Status    Status   `json:"status"`
	CreatedAt int64    `json:"createdAt"` // monotonic registration sequence (1-based)
}

func (t *Task) clone() *Task {
	if t == nil {
		return nil
	}
	c := *t
	if t.DependsOn != nil {
		c.DependsOn = append([]string(nil), t.DependsOn...)
	}
	return &c
}

// ---- structured errors carrying HTTP semantics ----

type statusErr struct {
	code int
	msg  string
}

func (e *statusErr) Error() string { return e.msg }

func badRequest(format string, a ...any) error {
	return &statusErr{code: http.StatusBadRequest, msg: fmt.Sprintf(format, a...)}
}
func conflict(format string, a ...any) error {
	return &statusErr{code: http.StatusConflict, msg: fmt.Sprintf(format, a...)}
}
func notFound(format string, a ...any) error {
	return &statusErr{code: http.StatusNotFound, msg: fmt.Sprintf(format, a...)}
}

// cycleErr is returned when the dependency graph contains a cycle; path is the
// sequence of task names that closes back on itself (e.g. ["A","B","A"]).
type cycleErr struct{ path []string }

func (e *cycleErr) Error() string { return "dependency cycle detected" }

// missingErr is returned when a dependency points at a task that was never
// registered, making the graph incomplete.
type missingErr struct{ names []string }

func (e *missingErr) Error() string { return "incomplete dependency graph: dangling dependencies" }

// ---- the in-memory service ----

type Service struct {
	mu    sync.Mutex
	tasks map[string]*Task
	seq   int64
}

func NewService() *Service {
	return &Service{tasks: make(map[string]*Task)}
}

func trimName(name string) string { return strings.TrimSpace(name) }

// dedupStrings trims each entry, drops blanks and duplicates while preserving
// first-seen order.
func dedupStrings(in []string) []string {
	seen := make(map[string]bool)
	out := make([]string, 0, len(in))
	for _, s := range in {
		t := strings.TrimSpace(s)
		if t == "" || seen[t] {
			continue
		}
		seen[t] = true
		out = append(out, t)
	}
	return out
}

func (s *Service) Create(name string, dependsOn []string) (*Task, error) {
	name = trimName(name)
	if name == "" {
		return nil, badRequest("task name must not be empty")
	}
	if strings.Contains(name, "/") {
		return nil, badRequest("task name must not contain '/'")
	}
	deps := dedupStrings(dependsOn)
	for _, d := range deps {
		if strings.Contains(d, "/") {
			return nil, badRequest("dependency name %q must not contain '/'", d)
		}
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if _, exists := s.tasks[name]; exists {
		return nil, conflict("task %q already exists", name)
	}
	for _, d := range deps {
		if d == name {
			return nil, badRequest("task %q cannot depend on itself", name)
		}
	}
	s.seq++
	t := &Task{Name: name, DependsOn: deps, CreatedAt: s.seq}
	s.tasks[name] = t
	s.recomputeLocked()
	return t.clone(), nil
}

func (s *Service) Get(name string) (*Task, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	t, ok := s.tasks[name]
	if !ok {
		return nil, notFound("task %q not found", name)
	}
	return t.clone(), nil
}

// List returns tasks in registration order, optionally filtered by status.
func (s *Service) List(filter Status) []*Task {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]*Task, 0, len(s.tasks))
	for _, t := range s.tasks {
		if filter != "" && t.Status != filter {
			continue
		}
		out = append(out, t.clone())
	}
	sort.Slice(out, func(i, j int) bool { return out[i].CreatedAt < out[j].CreatedAt })
	return out
}

func (s *Service) Complete(name string) (*Task, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	t, ok := s.tasks[name]
	if !ok {
		return nil, notFound("task %q not found", name)
	}
	if t.Status == StatusDone {
		return nil, conflict("task %q already completed", name)
	}
	for _, d := range t.DependsOn {
		dep, ok := s.tasks[d]
		if !ok {
			return nil, conflict("cannot complete task %q: dependency %q does not exist", name, d)
		}
		if dep.Status != StatusDone {
			return nil, conflict("cannot complete task %q: dependency %q is not completed", name, d)
		}
	}
	t.Status = StatusDone
	s.recomputeLocked()
	return t.clone(), nil
}

// Order returns a stable topological execution order: dependencies precede
// their dependents, and ties among simultaneously-ready tasks are broken by
// registration order. It returns a cycle path if a cycle exists, or the list of
// dangling dependency names if the graph is incomplete.
func (s *Service) Order() ([]string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	// 1. Incomplete graph: any dependency pointing at an unregistered task.
	missingSet := make(map[string]bool)
	for _, t := range s.tasks {
		for _, d := range t.DependsOn {
			if _, ok := s.tasks[d]; !ok {
				missingSet[d] = true
			}
		}
	}
	if len(missingSet) > 0 {
		missing := make([]string, 0, len(missingSet))
		for d := range missingSet {
			missing = append(missing, d)
		}
		sort.Strings(missing)
		return nil, &missingErr{names: missing}
	}

	// 2. Kahn's algorithm with stable (registration-order) tie-breaking.
	indeg := make(map[string]int, len(s.tasks))
	succ := make(map[string][]string) // dep -> tasks that depend on it
	for _, t := range s.tasks {
		indeg[t.Name] = len(t.DependsOn)
		for _, d := range t.DependsOn {
			succ[d] = append(succ[d], t.Name)
		}
	}

	var ready []*Task
	for _, t := range s.tasks {
		if indeg[t.Name] == 0 {
			ready = append(ready, t)
		}
	}
	popMin := func() *Task {
		minIdx := 0
		for i := 1; i < len(ready); i++ {
			if ready[i].CreatedAt < ready[minIdx].CreatedAt {
				minIdx = i
			}
		}
		t := ready[minIdx]
		ready = append(ready[:minIdx], ready[minIdx+1:]...)
		return t
	}

	order := make([]string, 0, len(s.tasks))
	emitted := 0
	for len(ready) > 0 {
		t := popMin()
		order = append(order, t.Name)
		emitted++
		for _, m := range succ[t.Name] {
			indeg[m]--
			if indeg[m] == 0 {
				ready = append(ready, s.tasks[m])
			}
		}
	}

	if emitted != len(s.tasks) {
		remaining := make(map[string]bool)
		for _, t := range s.tasks {
			if indeg[t.Name] > 0 {
				remaining[t.Name] = true
			}
		}
		return nil, &cycleErr{path: findCyclePath(remaining, succ)}
	}
	return order, nil
}

// findCyclePath performs a DFS over the remaining (cyclic) subgraph and returns
// the first cycle found as a path that closes back on its first node.
func findCyclePath(remaining map[string]bool, succ map[string][]string) []string {
	const (
		white = 0
		gray  = 1
		black = 2
	)
	color := make(map[string]int)
	var stack []string
	var path []string

	nodes := make([]string, 0, len(remaining))
	for n := range remaining {
		nodes = append(nodes, n)
	}
	sort.Strings(nodes)

	var dfs func(n string) bool
	dfs = func(n string) bool {
		color[n] = gray
		stack = append(stack, n)
		for _, m := range succ[n] {
			if !remaining[m] {
				continue
			}
			if color[m] == gray {
				for i, s := range stack {
					if s == m {
						path = append(append([]string{}, stack[i:]...), m)
						return true
					}
				}
			}
			if color[m] == white {
				if dfs(m) {
					return true
				}
			}
		}
		stack = stack[:len(stack)-1]
		color[n] = black
		return false
	}

	for _, n := range nodes {
		if color[n] == white {
			if dfs(n) {
				return path
			}
		}
	}
	return path
}

func (s *Service) recomputeLocked() {
	for _, t := range s.tasks {
		if t.Status == StatusDone {
			continue
		}
		t.Status = s.computeStatusLocked(t)
	}
}

func (s *Service) computeStatusLocked(t *Task) Status {
	for _, d := range t.DependsOn {
		dep, ok := s.tasks[d]
		if !ok || dep.Status != StatusDone {
			return StatusPending
		}
	}
	return StatusReady
}
