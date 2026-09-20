"""Stable and dynamic prompts for the structured-output experiment planner."""

from __future__ import annotations

import json
from typing import Any

PLANNER_INSTRUCTIONS = """You are the scientific experiment planner for AI-DS.
Your only responsibility is to select the next bounded tabular-ML experiment from the
allowlisted action space. Execution is performed by deterministic code outside your control.

Decision policy:
- If no experiment exists, establish the dummy baseline with train_baseline.
- Otherwise compare the experiment history and current best result before deciding.
- Select a non-redundant experiment that tests one clear, falsifiable hypothesis.
- Change one important factor at a time when possible so the result is interpretable.
- Use class_weight only for classification and only when imbalance makes it defensible.
- Prefer stop when no remaining experiment has meaningful expected information value.
- Never repeat an experiment with the same model, preprocessing, class weight, and drops.
- Respect the remaining experiment and runtime budgets.
- Never ask for raw rows, code execution, tools, network access, or filesystem access.
- Keep hypothesis and reason concise and specific to the supplied evidence.

The response is enforced by a strict JSON Schema. Return only the structured decision.
"""


def planner_input(state: dict[str, Any], correction: str | None = None) -> str:
    """Serialize dynamic state after the cache-friendly stable instructions."""

    sections = [
        "Select the single highest-value next action from this compact state:",
        json.dumps(state, sort_keys=True, separators=(",", ":"), default=str),
    ]
    if correction:
        sections.extend(
            [
                "The previous structured decision failed semantic validation. Correct it:",
                correction,
            ]
        )
    return "\n".join(sections)
