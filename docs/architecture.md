# Architecture

AI-DS separates planning from ML execution:

```text
CSV → profiler → problem solver → planner → validated action → experiment engine
                                        ↑                         ↓
                                  SQLite/JSONL trajectory ← evaluation result
```

`RunController` owns budgets and lifecycle. `DataProfiler` and `ProblemSolver` are
deterministic and work without an LLM. `ExperimentEngine` accepts only validated actions
and builds known sklearn pipelines from the model registry. `ExperimentStore` persists
the configuration, dataset hash, dataset profile, problem formulation, actions, results,
and lifecycle events.

The default planner is a bounded rule-based baseline. `LLMPlanner` is a callback adapter
for a provider integration that emits a single structured action. It receives a compact
state with profile summary and experiment history only. It has no execution, filesystem,
or raw-data capability.

## Safe action boundary

The schema accepts an allowlisted vocabulary; run mode exposes the executable subset:
`train_baseline`, `train_model`, and `stop`. Models are constructed exclusively through
the registry. Preprocessing choices are finite (`baseline`, `frequency`, `hash`, `target`,
`log_numeric`). No planner output is evaluated as Python, shell, SQL, or a file path.
