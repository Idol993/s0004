"""执行器智能体 - 负责根据计划执行具体动作"""

from typing import Any, Dict, List, Optional
import torch

from ..utils import AgentRole, MessageType
from .base_agent import BaseAgent
from .llm_backbone import LLMConfig
from ..communication.message_bus import CentralMessageBus


class ExecutorAgent(BaseAgent):
    def __init__(
        self,
        agent_id: str = "executor",
        llm_config: Optional[LLMConfig] = None,
        message_bus: Optional[CentralMessageBus] = None,
        checkpoint_dir: str = "./checkpoints/executor",
        log_dir: str = "./logs",
    ):
        if llm_config is None:
            llm_config = LLMConfig(model_name="gpt2")
        super().__init__(
            agent_id=agent_id,
            role=AgentRole.EXECUTOR,
            llm_config=llm_config,
            message_bus=message_bus,
            checkpoint_dir=checkpoint_dir,
            log_dir=log_dir,
        )
        self._execution_queue: List[Dict[str, Any]] = []
        self._execution_results: List[Dict[str, Any]] = []
        self._current_sub_task: Optional[Dict[str, Any]] = None

    def perceive(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        sub_task = observation.get("sub_task", {})
        plan_context = observation.get("plan_context", {})
        available_tools = observation.get("available_tools", [])

        prompt_parts = [
            f"Current sub-task: {sub_task.get('description', 'N/A')}",
            f"Task priority: {sub_task.get('priority', 'N/A')}",
            f"Plan context: {plan_context}",
        ]
        if available_tools:
            prompt_parts.append(f"Available tools: {available_tools}")

        return {
            "sub_task": sub_task,
            "plan_context": plan_context,
            "available_tools": available_tools,
            "prompt": "\n".join(prompt_parts),
        }

    def decide(self, perceived_state: Dict[str, Any]) -> Dict[str, Any]:
        prompt = (
            f"As an execution agent, determine the best action to complete the following sub-task.\n\n"
            f"{perceived_state['prompt']}\n\n"
            f"Provide the specific action to take, including method and parameters."
        )

        action_text = self.llm.generate(prompt, max_new_tokens=256, temperature=0.5)

        action = self._parse_action(action_text, perceived_state["sub_task"])
        action["confidence"] = self._compute_action_confidence(action)

        return action

    def act(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        self._current_sub_task = decision.get("sub_task")
        execution_result = {
            "action_type": "execute",
            "action": decision.get("action_name", "unknown"),
            "parameters": decision.get("parameters", {}),
            "confidence": decision.get("confidence", 0.5),
            "sub_task_id": decision.get("sub_task_id", ""),
            "result": self._simulate_execution(decision),
            "success": decision.get("confidence", 0.5) > 0.3,
        }
        self._execution_results.append(execution_result)

        self.send_message(
            receiver="evaluator",
            msg_type=MessageType.ACTION,
            content=execution_result,
        )

        return execution_result

    def _parse_action(self, action_text: str, sub_task: Dict[str, Any]) -> Dict[str, Any]:
        lines = [l.strip() for l in action_text.split("\n") if l.strip()]
        action_name = lines[0] if lines else "default_action"
        params = {}
        for line in lines[1:5]:
            if ":" in line:
                key, _, value = line.partition(":")
                params[key.strip()] = value.strip()

        return {
            "action_name": action_name[:100],
            "parameters": params,
            "sub_task_id": sub_task.get("id", "unknown"),
            "sub_task": sub_task,
        }

    def _compute_action_confidence(self, action: Dict[str, Any]) -> float:
        has_name = bool(action.get("action_name"))
        has_params = bool(action.get("parameters"))
        score = 0.3
        if has_name:
            score += 0.3
        if has_params:
            score += 0.2
        score += 0.2 * min(1.0, len(str(action.get("action_name", ""))) / 20.0)
        return min(1.0, score)

    def _simulate_execution(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        confidence = decision.get("confidence", 0.5)
        return {
            "output": f"Executed {decision.get('action_name', 'action')}",
            "confidence": confidence,
            "side_effects": [],
        }

    @property
    def execution_results(self) -> List[Dict[str, Any]]:
        return list(self._execution_results)
