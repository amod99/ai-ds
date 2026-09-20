# Architecture

AI-DS separates planning from ML execution:

```text
CSV → profiler → problem solver → compact state → LLM planner
                                                   ↓
                                      strict ExperimentAction
                                                   ↓
                                      semantic schema validation
                                                   ↓
                                          experiment engine
                                                   ↓
                                  SQLite/JSONL trajectory + result
                                                   │
                                                   └──→ next compact state
```

`RunController` owns budgets and lifecycle. `DataProfiler` and `ProblemSolver` are
deterministic and work without an LLM. `ExperimentEngine` accepts only validated actions
and builds known sklearn pipelines from the model registry. `ExperimentStore` persists
the configuration, dataset hash, dataset profile, problem formulation, actions, results,
and lifecycle events.

The default `LLMPlanner` calls the OpenAI Responses API with strict Structured Outputs.
Its compact input contains feature-level summaries, problem formulation, experiment
history, current best result, and remaining experiment/runtime budget. It contains no raw
rows. The request uses no tools and `store=False`; the returned action still passes local
allowlist, task/model compatibility, fixed-metric, feature-name, and duplicate-experiment
checks before execution.

Planner calls are bounded by a timeout, output-token limit, and retry count. Each accepted
decision records its provider, configured and returned model, response ID, token usage,
attempt count, validation errors, state snapshot, and structured payload. Exhausted planner
retries fail closed and are recorded in the trajectory. The rule-based planner remains
available through `--planner rule-based` as a deterministic benchmark.

## Safe action boundary

The schema accepts an allowlisted vocabulary; run mode exposes the executable subset:
`train_baseline`, `train_model`, and `stop`. Models are constructed exclusively through
the registry. Preprocessing choices are finite (`baseline`, `frequency`, `hash`, `target`,
`log_numeric`). No planner output is evaluated as Python, shell, SQL, or a file path.

## Planner configuration

Set `OPENAI_API_KEY` for authentication. Relevant CLI options are:

- `--llm-model`
- `--llm-reasoning-effort`
- `--llm-max-output-tokens`
- `--llm-max-retries`
- `--llm-timeout-seconds`
