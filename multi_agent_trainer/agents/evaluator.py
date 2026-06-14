"""评估器智能体 - 负责评估整体任务执行结果，判定成功/失败"""

from typing import Any, Dict, List, Optional
import torch

from ..utils import AgentRole, MessageType
from .base_agent import BaseAgent
from .llm_backbone import LLMConfig
from ..communication.message_bus import CentralMessageBus


class EvaluatorAgent(BaseAgent):
    def __init__(
        self,
        agent_id: str = "evaluator",
        llm_config: Optional[LLMConfig] = None,
        message_bus: Optional[CentralMessageBus] = None,
        checkpoint_dir: str = "./checkpoints/evaluator",
        log_dir: str = "./logs",
        success_threshold: float = 0.6,
    ):
        if llm_config is None:
            llm_config = LLMConfig(model_name="gpt2")
        super().__init__(
            agent_id=agent_id,
            role=AgentRole.EVALUATOR,
            llm_config=llm_config,
            message_bus=message_bus,
            checkpoint_dir=checkpoint_dir,
            log_dir=log_dir,
        )
        self.success_threshold = success_threshold
        self._evaluation_history: List[Dict[str, Any]] = []

    def perceive(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        task_result = observation.get("task_result", {})
        original_task = observation.get("original_task", "")
        execution_log = observation.get("execution_log", [])
        criteria = observation.get("criteria", {})

        prompt_parts = [
            f"Original task: {original_task}",
            f"Task result: {task_result}",
            f"Number of execution steps: {len(execution_log)}",
        ]
        if criteria:
            prompt_parts.append(f"Evaluation criteria: {criteria}")

        return {
            "task_result": task_result,
            "original_task": original_task,
            "execution_log": execution_log,
            "criteria": criteria,
            "prompt": "\n".join(prompt_parts),
        }

    def decide(self, perceived_state: Dict[str, Any]) -> Dict[str, Any]:
        prompt = (
            f"As an evaluation agent, assess the following task execution result.\n\n"
            f"{perceived_state['prompt']}\n\n"
            f"Provide a score from 0 to 1, indicate whether the task succeeded, "
            f"and list specific issues if any."
        )

        eval_text = self.llm.generate(prompt, max_new_tokens=256, temperature=0.3)

        score = self._extract_score(eval_text)
        success = score >= self.success_threshold

        decision = {
            "evaluation_text": eval_text,
            "score": score,
            "success": success,
            "issues": self._extract_issues(eval_text),
            "per_agent_scores": self._compute_per_agent_scores(perceived_state),
        }
        return decision

    def act(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        evaluation = {
            "action_type": "evaluate",
            "score": decision["score"],
            "success": decision["success"],
            "issues": decision["issues"],
            "per_agent_scores": decision["per_agent_scores"],
            "threshold": self.success_threshold,
        }

        self._evaluation_history.append(evaluation)

        self.send_message(
            receiver="reflector",
            msg_type=MessageType.EVALUATION,
            content=evaluation,
        )

        if not decision["success"]:
            self.send_message(
                receiver="*",
                msg_type=MessageType.ROLLBACK_REQUEST,
                content={
                    "reason": "task_failed",
                    "score": decision["score"],
                    "per_agent_scores": decision["per_agent_scores"],
                },
            )

        return evaluation

    def _extract_score(self, eval_text: str) -> float:
        import re
        patterns = [
            r"[Ss]core[:\s]+([0-9]*\.?[0-9]+)",
            r"[Rr]ating[:\s]+([0-9]*\.?[0-9]+)",
            r"([0-9]*\.?[0-9]+)/1\.?0?",
        ]
        for pattern in patterns:
            match = re.search(pattern, eval_text)
            if match:
                return float(match.group(1))
        return 0.4

    def _extract_issues(self, eval_text: str) -> List[Dict[str, Any]]:
        lines = [l.strip() for l in eval_text.split("\n") if l.strip()]
        issues = []
        for line in lines:
            negative_words = ["fail", "error", "wrong", "incorrect", "missing", "bad", "poor"]
            if any(w in line.lower() for w in negative_words):
                issues.append({"description": line[:200], "severity": 0.5})
        return issues[:5]

    def _compute_per_agent_scores(self, perceived_state: Dict[str, Any]) -> Dict[str, float]:
        execution_log = perceived_state.get("execution_log", [])
        scores = {}
        agent_contributions: Dict[str, List[float]] = {}

        for entry in execution_log:
            agent_id = entry.get("agent_id", "unknown")
            confidence = entry.get("confidence", 0.5)
            if agent_id not in agent_contributions:
                agent_contributions[agent_id] = []
            agent_contributions[agent_id].append(confidence)

        for agent_id, confidences in agent_contributions.items():
            if confidences:
                scores[agent_id] = sum(confidences) / len(confidences)
            else:
                scores[agent_id] = 0.5

        return scores

    @property
    def evaluation_history(self) -> List[Dict[str, Any]]:
        return list(self._evaluation_history)

    def get_latest_evaluation(self) -> Optional[Dict[str, Any]]:
        return self._evaluation_history[-1] if self._evaluation_history else None
