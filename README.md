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
ai-ds run examples/datasets/customer_churn.csv --target churn --max-experiments 8
```

`run` uses the default bounded rule-based planner. Applications can supply an
`LLMPlanner` callback that returns the same validated action schema; it receives only
the compact state (profile summary, problem, experiment history, budget), never the raw
dataset or filesystem access.

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
- Dataset hashes, configuration, actions, results, and stop reason are persisted.
- Transformations—including target encoding—are fitted inside each CV training fold.
- The action schema forbids arbitrary code, shell execution, and file access.
- Experiment count, elapsed time, failure count, and convergence patience are bounded.

See [docs/architecture.md](docs/architecture.md) and
[docs/evaluation.md](docs/evaluation.md) for the implementation details.
