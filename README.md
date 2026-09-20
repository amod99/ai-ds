# AI-DS

AI-DS is a small, reproducible framework for conducting bounded machine-learning
experiments on tabular CSV datasets. It profiles a dataset, formulates a problem,
evaluates leakage-safe pipelines with cross-validation, records each decision, and
produces a final model and research report.

It is deliberately **not** an unrestricted coding agent. Planning is separated from
execution: a planner may select only validated, allowlisted experiment actions, while
the experiment engine runs deterministic Python code.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"

ai-ds profile examples/datasets/customer_churn.csv
ai-ds train examples/datasets/customer_churn.csv --target churn

# PowerShell (use `export OPENAI_API_KEY=...` on macOS/Linux)
$env:OPENAI_API_KEY = "your-api-key"
ai-ds run examples/datasets/customer_churn.csv --target churn --max-experiments 8
```

`run` uses the OpenAI Responses API planner by default. It sends only compact dataset
statistics, experiment history, the current best result, and remaining budget—never raw
rows. The response must match a strict JSON Schema and then passes an independent semantic
validator before the deterministic experiment engine can execute it.

The default model is `gpt-5.6-luna`. Override it with `--llm-model` or
`AI_DS_LLM_MODEL`; use `--planner rule-based` to run the deterministic comparison planner:

```bash
ai-ds run data.csv --target label --llm-model gpt-5.6-luna
ai-ds run data.csv --target label --planner rule-based
```

The OpenAI request uses `store=False`, has no tools, retries invalid decisions within a
small configured bound, and persists the response ID, token usage, validation errors, and
accepted structured action in the run trajectory. `CallbackPlanner` remains available for
custom provider integrations and evaluation harnesses.

## What a run writes

Each run receives a directory beneath `runs/` containing:

```text
<run-id>/
  experiments.sqlite   # configuration, profile, actions, results, trajectory
  trajectory.jsonl     # append-only, human-inspectable decision trace
  final_model.joblib   # best pipeline refit on the complete input data
  report.md            # human-readable research report
```

The primary validation protocol is fixed, shuffled cross-validation with a configured
seed. A model is selected only by validation score (ROC-AUC by default for binary
classification and RMSE for regression), never training performance.

## Safety and reproducibility

- Inputs are limited to one CSV and explicit command-line configuration.
- Dataset hashes, configuration, planner decisions, API usage, results, and stop reason are persisted.
- Transformations—including target encoding—are fitted inside each CV training fold.
- The action schema forbids arbitrary code, shell execution, and file access.
- Experiment count, elapsed time, failure count, and convergence patience are bounded.

See [docs/architecture.md](docs/architecture.md) and
[docs/evaluation.md](docs/evaluation.md) for the implementation details.
