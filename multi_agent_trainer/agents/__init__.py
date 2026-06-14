"""智能体模块包"""

from .base_agent import BaseAgent, ActionRecord
from .llm_backbone import LLMConfig, CheckpointInfo

AGENT_REGISTRY = {
    "planner": "multi_agent_trainer.agents.planner:PlannerAgent",
    "executor": "multi_agent_trainer.agents.executor:ExecutorAgent",
    "evaluator": "multi_agent_trainer.agents.evaluator:EvaluatorAgent",
    "memory": "multi_agent_trainer.agents.memory:MemoryAgent",
    "reflector": "multi_agent_trainer.agents.reflector:ReflectorAgent",
}


def __getattr__(name):
    if name == "PlannerAgent":
        from .planner import PlannerAgent
        return PlannerAgent
    elif name == "ExecutorAgent":
        from .executor import ExecutorAgent
        return ExecutorAgent
    elif name == "EvaluatorAgent":
        from .evaluator import EvaluatorAgent
        return EvaluatorAgent
    elif name == "MemoryAgent":
        from .memory import MemoryAgent
        return MemoryAgent
    elif name == "ReflectorAgent":
        from .reflector import ReflectorAgent
        return ReflectorAgent
    elif name == "LLMBackbone":
        from .llm_backbone import LLMBackbone
        return LLMBackbone
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BaseAgent",
    "ActionRecord",
    "LLMBackbone",
    "LLMConfig",
    "CheckpointInfo",
    "PlannerAgent",
    "ExecutorAgent",
    "EvaluatorAgent",
    "MemoryAgent",
    "ReflectorAgent",
    "AGENT_REGISTRY",
]
