"""多智能体大模型训练系统 - 主入口

系统包含五个独立的微调LLM智能体：
- 规划器 (Planner): 任务分解与策略制定
- 执行器 (Executor): 根据计划执行具体动作
- 评估器 (Evaluator): 评估整体任务执行结果
- 记忆器 (Memory): 经验存储与检索
- 反思器 (Reflector): 错误分析与策略改进

核心地狱级功能：
- 因果责任推断: 基于反事实基线计算每个智能体动作的边际贡献
- MDP时延信用分配: 基于马尔可夫决策过程处理时延奖励下的信用分配
- 选择性回滚重训练: 仅回滚和重训练责任智能体，其他保持冻结
- 开销控制: 确保训练时间复杂度和通信开销不超过总成本的15%
"""

import os
import json
from typing import Any, Dict, List, Optional

from .communication import CentralMessageBus, Message, MessageFactory
from .responsibility import (
    CausalResponsibilityInferencer,
    CounterfactualConfig,
    ResponsibilityReport,
    MDPCreditAssignmentNetwork,
    MDPCreditConfig,
)
from .training import (
    TrainingOrchestrator,
    TrainingConfig,
    SelectiveRollbackManager,
    RollbackConfig,
    OverheadBudgetManager,
)
from .utils import set_seed, get_device, TrainingMetrics, Timer


def __getattr__(name):
    _lazy = {
        "BaseAgent": ".agents.base_agent:BaseAgent",
        "PlannerAgent": ".agents.planner:PlannerAgent",
        "ExecutorAgent": ".agents.executor:ExecutorAgent",
        "EvaluatorAgent": ".agents.evaluator:EvaluatorAgent",
        "MemoryAgent": ".agents.memory:MemoryAgent",
        "ReflectorAgent": ".agents.reflector:ReflectorAgent",
        "LLMConfig": ".agents.llm_backbone:LLMConfig",
        "AGENT_REGISTRY": ".agents:AGENT_REGISTRY",
    }
    if name in _lazy:
        module_path, _, attr = _lazy[name].partition(":")
        import importlib
        mod = importlib.import_module(module_path, __name__)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def create_system(
    config_path: Optional[str] = None,
    config_dict: Optional[Dict[str, Any]] = None,
    device: str = "cuda",
    seed: int = 42,
    log_dir: str = "./logs",
    checkpoint_dir: str = "./checkpoints",
) -> TrainingOrchestrator:
    from omegaconf import OmegaConf
    from .agents.llm_backbone import LLMConfig

    if config_path and os.path.exists(config_path):
        cfg = OmegaConf.load(config_path)
    elif config_dict:
        cfg = OmegaConf.create(config_dict)
    else:
        cfg = OmegaConf.create({})

    set_seed(seed)

    training_config = TrainingConfig(
        max_episodes=cfg.get("training", {}).get("max_episodes", 1000),
        batch_size=cfg.get("training", {}).get("batch_size", 4),
        gradient_accumulation_steps=cfg.get("training", {}).get("gradient_accumulation_steps", 4),
        max_grad_norm=cfg.get("training", {}).get("max_grad_norm", 1.0),
        warmup_steps=cfg.get("training", {}).get("warmup_steps", 100),
        save_interval=cfg.get("training", {}).get("save_interval", 10),
        eval_interval=cfg.get("training", {}).get("eval_interval", 5),
        max_overhead_ratio=cfg.get("system", {}).get("max_overhead_ratio", 0.15),
        device=device,
        seed=seed,
        log_dir=log_dir,
        checkpoint_base_dir=checkpoint_dir,
    )

    llm_configs = {}
    agents_cfg = cfg.get("agents", {})
    for role in ["planner", "executor", "evaluator", "memory", "reflector"]:
        agent_cfg = agents_cfg.get(role, {})
        llm_configs[role] = LLMConfig(
            model_name=agent_cfg.get("model_name", "gpt2-medium"),
            device=device,
            learning_rate=agent_cfg.get("learning_rate", 1e-5),
        )

    cf_cfg = cfg.get("responsibility_inference", {}).get("counterfactual_baseline", {})
    counterfactual_config = CounterfactualConfig(
        num_samples=cf_cfg.get("num_samples", 5),
        baseline_type=cf_cfg.get("baseline_type", "random"),
        shapley_enabled=cf_cfg.get("shapley_enabled", True),
        shapley_num_permutations=cf_cfg.get("shapley_num_permutations", 20),
    )

    mdp_cfg = cfg.get("responsibility_inference", {}).get("mdp_credit_assignment", {})
    mdp_config = MDPCreditConfig(
        gamma=mdp_cfg.get("gamma", 0.99),
        lambda_gae=mdp_cfg.get("lambda_gae", 0.95),
        hidden_dim=mdp_cfg.get("hidden_dim", 256),
        num_layers=mdp_cfg.get("num_layers", 2),
        learning_rate=mdp_cfg.get("learning_rate", 1e-4),
    )

    rb_cfg = cfg.get("rollback", {})
    rollback_config = RollbackConfig(
        enabled=rb_cfg.get("enabled", True),
        retain_best_checkpoint=rb_cfg.get("retain_best_checkpoint", True),
        min_improvement_threshold=rb_cfg.get("min_improvement_threshold", 0.01),
        max_retries=rb_cfg.get("max_retries", 3),
        max_overhead_ratio=cfg.get("system", {}).get("max_overhead_ratio", 0.15),
    )

    orchestrator = TrainingOrchestrator(
        config=training_config,
        llm_configs=llm_configs,
        counterfactual_config=counterfactual_config,
        mdp_config=mdp_config,
        rollback_config=rollback_config,
    )

    return orchestrator


__all__ = [
    "create_system",
    "TrainingOrchestrator",
    "TrainingConfig",
    "PlannerAgent",
    "ExecutorAgent",
    "EvaluatorAgent",
    "MemoryAgent",
    "ReflectorAgent",
    "CausalResponsibilityInferencer",
    "CounterfactualConfig",
    "MDPCreditAssignmentNetwork",
    "MDPCreditConfig",
    "SelectiveRollbackManager",
    "RollbackConfig",
    "OverheadBudgetManager",
    "CentralMessageBus",
]
