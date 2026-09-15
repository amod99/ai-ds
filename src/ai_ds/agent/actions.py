"""Action validation entry point kept separate from planner implementations."""

from __future__ import annotations

from typing import Any

from ai_ds.schemas.agent import AgentAction


def validate_action(payload: dict[str, Any], task_type: str) -> AgentAction:
    """Validate untrusted structured planner output against the V1 action schema."""

    return AgentAction.from_dict(payload, task_type)
