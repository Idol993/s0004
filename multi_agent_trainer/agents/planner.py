"""规划器智能体 - 负责任务分解和策略制定"""

from typing import Any, Dict, List, Optional
import torch

from ..utils import AgentRole, MessageType
from ..communication import MessageFactory
from .base_agent import BaseAgent, ActionRecord
from .llm_backbone import LLMConfig
from ..communication.message_bus import CentralMessageBus


class PlannerAgent(BaseAgent):
    def __init__(
        self,
        agent_id: str = "planner",
        llm_config: Optional[LLMConfig] = None,
        message_bus: Optional[CentralMessageBus] = None,
        checkpoint_dir: str = "./checkpoints/planner",
        log_dir: str = "./logs",
    ):
        if llm_config is None:
            llm_config = LLMConfig(model_name="gpt2-medium")
        super().__init__(
            agent_id=agent_id,
            role=AgentRole.PLANNER,
            llm_config=llm_config,
            message_bus=message_bus,
            checkpoint_dir=checkpoint_dir,
            log_dir=log_dir,
        )
        self._current_plan: List[Dict[str, Any]] = []
        self._plan_history: List[List[Dict[str, Any]]] = []

    def perceive(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        task = observation.get("task", "")
        context = observation.get("context", {})
        constraints = observation.get("constraints", {})
        previous_feedback = observation.get("previous_feedback", None)

        prompt_parts = [
            f"Task: {task}",
            f"Context: {context}",
            f"Constraints: {constraints}",
        ]
        if previous_feedback:
            prompt_parts.append(f"Previous feedback: {previous_feedback}")

        perceived = {
            "raw_task": task,
            "context": context,
            "constraints": constraints,
            "previous_feedback": previous_feedback,
            "prompt": "\n".join(prompt_parts),
        }
        return perceived

    def decide(self, perceived_state: Dict[str, Any]) -> Dict[str, Any]:
        prompt = (
            f"As a planning agent, decompose the following task into sub-tasks.\n\n"
            f"{perceived_state['prompt']}\n\n"
            f"Provide a structured plan with sub-tasks, their dependencies, and priorities."
        )

        plan_text = self.llm.generate(prompt, max_new_tokens=512, temperature=0.7)

        sub_tasks = self._parse_plan(plan_text)

        decision = {
            "plan_text": plan_text,
            "sub_tasks": sub_tasks,
            "plan_confidence": self._compute_plan_confidence(sub_tasks),
            "estimated_difficulty": self._estimate_difficulty(sub_tasks),
        }
        return decision

    def act(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        self._current_plan = decision["sub_tasks"]
        self._plan_history.append(list(self._current_plan))

        for i, sub_task in enumerate(decision["sub_tasks"]):
            self.send_message(
                receiver="executor",
                msg_type=MessageType.PLAN,
                content={
                    "sub_task": sub_task,
                    "task_index": i,
                    "total_tasks": len(decision["sub_tasks"]),
                    "plan_confidence": decision["plan_confidence"],
                },
            )

        return {
            "action_type": "plan_created",
            "sub_tasks": decision["sub_tasks"],
            "confidence": decision["plan_confidence"],
            "estimated_difficulty": decision["estimated_difficulty"],
        }

    def _parse_plan(self, plan_text: str) -> List[Dict[str, Any]]:
        lines = [l.strip() for l in plan_text.split("\n") if l.strip()]
        sub_tasks = []
        for i, line in enumerate(lines[:10]):
            sub_tasks.append({
                "id": f"subtask_{i}",
                "description": line,
                "priority": max(0, 10 - i),
                "dependencies": [f"subtask_{j}" for j in range(i) if i > 0 and i % 3 == 0],
                "status": "pending",
            })
        if not sub_tasks:
            sub_tasks.append({
                "id": "subtask_0",
                "description": plan_text[:200],
                "priority": 10,
                "dependencies": [],
                "status": "pending",
            })
        return sub_tasks

    def _compute_plan_confidence(self, sub_tasks: List[Dict[str, Any]]) -> float:
        if not sub_tasks:
            return 0.0
        dep_count = sum(len(t.get("dependencies", [])) for t in sub_tasks)
        complexity = len(sub_tasks) + dep_count * 0.5
        return max(0.1, min(1.0, 1.0 / (1.0 + complexity * 0.1)))

    def _estimate_difficulty(self, sub_tasks: List[Dict[str, Any]]) -> float:
        if not sub_tasks:
            return 0.5
        total_dep = sum(len(t.get("dependencies", [])) for t in sub_tasks)
        return min(1.0, (len(sub_tasks) + total_dep) / 20.0)

    @property
    def current_plan(self) -> List[Dict[str, Any]]:
        return list(self._current_plan)
