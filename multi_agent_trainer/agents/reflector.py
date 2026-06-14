"""反思器智能体 - 负责错误分析和策略改进建议"""

from typing import Any, Dict, List, Optional
import torch

from ..utils import AgentRole, MessageType
from .base_agent import BaseAgent
from .llm_backbone import LLMConfig
from ..communication.message_bus import CentralMessageBus


class ReflectorAgent(BaseAgent):
    def __init__(
        self,
        agent_id: str = "reflector",
        llm_config: Optional[LLMConfig] = None,
        message_bus: Optional[CentralMessageBus] = None,
        checkpoint_dir: str = "./checkpoints/reflector",
        log_dir: str = "./logs",
    ):
        if llm_config is None:
            llm_config = LLMConfig(model_name="gpt2")
        super().__init__(
            agent_id=agent_id,
            role=AgentRole.REFLECTOR,
            llm_config=llm_config,
            message_bus=message_bus,
            checkpoint_dir=checkpoint_dir,
            log_dir=log_dir,
        )
        self._reflection_history: List[Dict[str, Any]] = []
        self._improvement_suggestions: List[Dict[str, Any]] = []
        self._processed_episodes: set = set()

    def perceive(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        evaluation = observation.get("evaluation", {})
        execution_log = observation.get("execution_log", [])
        plan = observation.get("plan", [])
        failure_info = observation.get("failure_info", {})

        prompt_parts = []
        if evaluation:
            prompt_parts.append(f"Evaluation score: {evaluation.get('score', 'N/A')}")
            prompt_parts.append(f"Success: {evaluation.get('success', 'N/A')}")
            if evaluation.get("issues"):
                prompt_parts.append("Issues found:")
                for issue in evaluation["issues"]:
                    prompt_parts.append(f"  - {issue.get('description', 'Unknown')}")

        if failure_info:
            prompt_parts.append(f"Failure details: {failure_info}")

        prompt_parts.append(f"Number of plan steps: {len(plan)}")
        prompt_parts.append(f"Execution steps: {len(execution_log)}")

        return {
            "evaluation": evaluation,
            "execution_log": execution_log,
            "plan": plan,
            "failure_info": failure_info,
            "prompt": "\n".join(prompt_parts),
        }

    def decide(self, perceived_state: Dict[str, Any]) -> Dict[str, Any]:
        evaluation = perceived_state["evaluation"]
        is_failure = not evaluation.get("success", True)

        prompt = (
            f"As a reflection agent, analyze the following task execution "
            f"{'failure' if is_failure else 'success'}.\n\n"
            f"{perceived_state['prompt']}\n\n"
        )

        if is_failure:
            prompt += (
                "Identify: 1) Which agent(s) made the most impactful errors, "
                "2) What specific decisions were wrong, "
                "3) How should those agents improve? "
                "Provide agent-specific improvement suggestions."
            )
        else:
            prompt += (
                "Identify what went well and suggest minor improvements "
                "for even better performance."
            )

        reflection_text = self.llm.generate(prompt, max_new_tokens=512, temperature=0.5)

        analysis = self._parse_reflection(reflection_text, perceived_state)

        decision = {
            "reflection_text": reflection_text,
            "is_failure": is_failure,
            "responsible_agents": analysis["responsible_agents"],
            "error_analysis": analysis["error_analysis"],
            "improvement_suggestions": analysis["suggestions"],
            "root_cause_hypothesis": analysis["root_cause"],
        }
        return decision

    def act(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        episode_id = getattr(self, '_current_episode', None)
        if episode_id is not None and episode_id in self._processed_episodes:
            self.logger.info(f"Episode {episode_id} reflection already processed, skipping duplicate")
            return {
                "action_type": "reflect_skipped",
                "reason": "duplicate_episode",
                "episode_id": episode_id,
            }

        reflection = {
            "action_type": "reflect",
            "is_failure": decision["is_failure"],
            "responsible_agents": decision["responsible_agents"],
            "error_analysis": decision["error_analysis"],
            "improvement_suggestions": decision["improvement_suggestions"],
            "root_cause": decision["root_cause_hypothesis"],
            "episode_id": episode_id,
        }

        self._reflection_history.append(reflection)
        self._improvement_suggestions.extend(decision["improvement_suggestions"])
        if episode_id is not None:
            self._processed_episodes.add(episode_id)

        self.send_message(
            receiver="planner",
            msg_type=MessageType.REFLECTION,
            content={
                "improvement_suggestions": decision["improvement_suggestions"],
                "root_cause": decision["root_cause_hypothesis"],
                "is_failure": decision["is_failure"],
                "episode_id": episode_id,
            },
        )

        if decision["is_failure"]:
            self.send_message(
                receiver="memory",
                msg_type=MessageType.MEMORY_QUERY,
                content={
                    "query_type": "store",
                    "data_to_store": {
                        "type": "failure_experience",
                        "responsible_agents": decision["responsible_agents"],
                        "error_analysis": decision["error_analysis"],
                        "root_cause": decision["root_cause_hypothesis"],
                        "failed": True,
                        "episode_id": episode_id,
                    },
                },
            )

        return reflection

    def _parse_reflection(
        self, reflection_text: str, perceived_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        evaluation = perceived_state["evaluation"]
        per_agent_scores = evaluation.get("per_agent_scores", {})

        responsible_agents = []
        if not evaluation.get("success", True):
            for agent_id, score in per_agent_scores.items():
                if score < 0.5:
                    responsible_agents.append({
                        "agent_id": agent_id,
                        "contribution_score": score,
                        "likely_responsible": True,
                    })

            if not responsible_agents:
                sorted_agents = sorted(
                    per_agent_scores.items(), key=lambda x: x[1]
                )
                for agent_id, score in sorted_agents[:2]:
                    responsible_agents.append({
                        "agent_id": agent_id,
                        "contribution_score": score,
                        "likely_responsible": True,
                    })

        error_analysis = []
        lines = reflection_text.split("\n")
        for line in lines:
            error_keywords = ["error", "mistake", "wrong", "incorrect", "failure", "bad"]
            if any(kw in line.lower() for kw in error_keywords):
                error_analysis.append({"description": line.strip()[:300]})

        if not error_analysis and not evaluation.get("success", True):
            error_analysis.append({
                "description": f"Overall task failure with score {evaluation.get('score', 0)}"
            })

        suggestions = []
        for resp in responsible_agents:
            suggestions.append({
                "agent_id": resp["agent_id"],
                "suggestion": f"Agent {resp['agent_id']} should improve decision quality",
                "priority": "high" if resp["contribution_score"] < 0.3 else "medium",
            })

        root_cause = reflection_text[:500] if reflection_text else "Unknown root cause"

        return {
            "responsible_agents": responsible_agents,
            "error_analysis": error_analysis[:5],
            "suggestions": suggestions,
            "root_cause": root_cause,
        }

    @property
    def reflection_history(self) -> List[Dict[str, Any]]:
        return list(self._reflection_history)

    @property
    def improvement_suggestions(self) -> List[Dict[str, Any]]:
        return list(self._improvement_suggestions)
