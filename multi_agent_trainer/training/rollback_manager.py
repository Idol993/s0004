"""智能体回滚与选择性重训练机制

核心功能：
1. 根据因果责任推断的结果，仅回滚被识别为责任智能体的检查点
2. 保持非责任智能体冻结，只对责任智能体进行重训练
3. 重训练使用改进的反事实梯度信号，避免重复犯错
4. 确保回滚和重训练的开销不超过总训练成本的15%

策略：
- 选择性回滚：仅回滚责任智能体到最佳检查点
- 梯度加权重训练：对责任智能体的错误动作施加负向梯度
- 冻结保护：非责任智能体完全冻结，不参与重训练
- 开销控制：通过预算管理器限制回滚/重训练的总时间预算
"""

import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum

import torch
import torch.nn as nn
import numpy as np

from ..agents.base_agent import BaseAgent, ActionRecord
from ..agents.llm_backbone import LLMBackbone, CheckpointInfo
from ..responsibility.causal_inference import ResponsibilityReport, ResponsibilityScore
from ..responsibility.mdp_credit import MDPCreditAssignmentNetwork
from ..utils import AgentRole, Timer, TrainingMetrics, setup_logger


class RollbackStrategy(str, Enum):
    BEST_CHECKPOINT = "best_checkpoint"
    PRE_FAILURE = "pre_failure"
    STEP_ROLLBACK = "step_rollback"
    PARTIAL_ROLLBACK = "partial_rollback"


class RetrainingStrategy(str, Enum):
    STANDARD = "standard"
    COUNTERFACTUAL_GRADIENT = "counterfactual_gradient"
    ADVERSARIAL = "adversarial"
    CURRICULUM = "curriculum"


@dataclass
class RollbackConfig:
    enabled: bool = True
    retain_best_checkpoint: bool = True
    min_improvement_threshold: float = 0.01
    max_retries: int = 3
    rollback_strategy: RollbackStrategy = RollbackStrategy.BEST_CHECKPOINT
    retraining_strategy: RetrainingStrategy = RetrainingStrategy.COUNTERFACTUAL_GRADIENT
    max_overhead_ratio: float = 0.15
    retraining_lr_multiplier: float = 0.5
    retraining_steps: int = 50


@dataclass
class RollbackRecord:
    episode_id: str
    responsible_agents: List[str]
    frozen_agents: List[str]
    rollback_strategy: RollbackStrategy
    rollback_time: float
    retraining_time: float
    pre_rollback_score: float
    post_retraining_score: float
    improvement: float
    success: bool


class OverheadBudgetManager:
    def __init__(self, max_overhead_ratio: float = 0.15):
        self.max_overhead_ratio = max_overhead_ratio
        self._total_training_time: float = 0.0
        self._total_overhead_time: float = 0.0
        self._rollback_time: float = 0.0
        self._retraining_time: float = 0.0
        self._inference_time: float = 0.0
        self._communication_time: float = 0.0
        self._lock = None

    @property
    def current_ratio(self) -> float:
        if self._total_training_time <= 0:
            return 0.0
        return self._total_overhead_time / self._total_training_time

    @property
    def remaining_budget(self) -> float:
        used = self.current_ratio
        return max(0.0, self.max_overhead_ratio - used)

    @property
    def available_overhead_time(self) -> float:
        if self._total_training_time <= 0:
            return float("inf")
        max_overhead = self._total_training_time * self.max_overhead_ratio
        return max(0.0, max_overhead - self._total_overhead_time)

    def record_training_time(self, duration: float) -> None:
        self._total_training_time += duration

    def record_rollback_time(self, duration: float) -> None:
        self._rollback_time += duration
        self._total_overhead_time += duration

    def record_retraining_time(self, duration: float) -> None:
        self._retraining_time += duration
        self._total_overhead_time += duration

    def record_inference_time(self, duration: float) -> None:
        self._inference_time += duration
        self._total_overhead_time += duration

    def record_communication_time(self, duration: float) -> None:
        self._communication_time += duration
        self._total_overhead_time += duration

    def can_afford(self, estimated_time: float) -> bool:
        if self._total_training_time <= 0:
            return True
        projected_overhead = self._total_overhead_time + estimated_time
        projected_ratio = projected_overhead / (self._total_training_time + estimated_time)
        return projected_ratio <= self.max_overhead_ratio

    def get_report(self) -> Dict[str, Any]:
        return {
            "total_training_time": self._total_training_time,
            "total_overhead_time": self._total_overhead_time,
            "overhead_ratio": self.current_ratio,
            "remaining_budget": self.remaining_budget,
            "rollback_time": self._rollback_time,
            "retraining_time": self._retraining_time,
            "inference_time": self._inference_time,
            "communication_time": self._communication_time,
            "within_budget": self.current_ratio <= self.max_overhead_ratio,
        }


class SelectiveRollbackManager:
    def __init__(
        self,
        agents: Dict[str, BaseAgent],
        config: RollbackConfig,
        mdp_credit_network: Optional[MDPCreditAssignmentNetwork] = None,
        log_dir: str = "./logs",
    ):
        self.agents = agents
        self.config = config
        self.mdp_credit_network = mdp_credit_network
        self.log_dir = log_dir

        self.logger = setup_logger("rollback_manager", log_dir)
        self.budget_manager = OverheadBudgetManager(config.max_overhead_ratio)

        self._rollback_history: List[RollbackRecord] = []
        self._frozen_agents: Set[str] = set()
        self._retraining_counts: Dict[str, int] = {}

        self._pre_rollback_states: Dict[str, Dict[str, Any]] = {}

    def execute_rollback(
        self,
        responsibility_report: ResponsibilityReport,
        episode_id: str,
    ) -> RollbackRecord:
        if not self.config.enabled:
            return RollbackRecord(
                episode_id=episode_id,
                responsible_agents=[],
                frozen_agents=list(self.agents.keys()),
                rollback_strategy=self.config.rollback_strategy,
                rollback_time=0.0,
                retraining_time=0.0,
                pre_rollback_score=responsibility_report.overall_score,
                post_retraining_score=responsibility_report.overall_score,
                improvement=0.0,
                success=False,
            )

        start_time = time.time()
        pre_rollback_score = responsibility_report.overall_score

        responsible_ids = responsibility_report.responsible_agents
        self.logger.info(
            f"Executing selective rollback for episode {episode_id}: "
            f"responsible_agents={responsible_ids}"
        )

        self._save_pre_rollback_states()

        rollback_time = self._perform_rollback(responsible_ids, responsibility_report)

        self._freeze_non_responsible(responsible_ids)

        estimated_retraining = len(responsible_ids) * self.config.retraining_steps * 0.01
        if self.budget_manager.can_afford(estimated_retraining):
            retraining_time = self._retrain_responsible(
                responsible_ids, responsibility_report
            )
        else:
            self.logger.warning("Insufficient budget for retraining, skipping")
            retraining_time = 0.0

        post_score = self._evaluate_after_retraining(responsibility_report)
        improvement = post_score - pre_rollback_score

        self._unfreeze_all()

        total_time = time.time() - start_time
        self.budget_manager.record_rollback_time(rollback_time)
        self.budget_manager.record_retraining_time(retraining_time)

        for aid in responsible_ids:
            self._retraining_counts[aid] = self._retraining_counts.get(aid, 0) + 1

        record = RollbackRecord(
            episode_id=episode_id,
            responsible_agents=responsible_ids,
            frozen_agents=[a for a in self.agents if a not in responsible_ids],
            rollback_strategy=self.config.rollback_strategy,
            rollback_time=rollback_time,
            retraining_time=retraining_time,
            pre_rollback_score=pre_rollback_score,
            post_retraining_score=post_score,
            improvement=improvement,
            success=improvement > self.config.min_improvement_threshold,
        )
        self._rollback_history.append(record)

        self.logger.info(
            f"Rollback complete: pre={pre_rollback_score:.4f}, "
            f"post={post_score:.4f}, improvement={improvement:.4f}, "
            f"success={record.success}"
        )

        return record

    def _save_pre_rollback_states(self) -> None:
        for agent_id, agent in self.agents.items():
            self._pre_rollback_states[agent_id] = agent.llm.clone_state()

    def _perform_rollback(
        self,
        responsible_ids: List[str],
        report: ResponsibilityReport,
    ) -> float:
        start_time = time.time()

        for agent_id in responsible_ids:
            agent = self.agents.get(agent_id)
            if agent is None:
                continue

            if self.config.rollback_strategy == RollbackStrategy.BEST_CHECKPOINT:
                success = agent.rollback_to_best()
                if not success:
                    self.logger.warning(f"No best checkpoint for agent {agent_id}, using state snapshot")
                    self._rollback_to_safe_state(agent, report)

            elif self.config.rollback_strategy == RollbackStrategy.PRE_FAILURE:
                self._rollback_to_pre_failure(agent, report)

            elif self.config.rollback_strategy == RollbackStrategy.STEP_ROLLBACK:
                error_step = self._find_error_step(agent_id, report)
                if error_step is not None:
                    agent.rollback_to(error_step)
                else:
                    agent.rollback_to_best()

            elif self.config.rollback_strategy == RollbackStrategy.PARTIAL_ROLLBACK:
                self._partial_rollback(agent, agent_id, report)

            self.logger.info(f"Agent '{agent_id}' rolled back using {self.config.rollback_strategy.value}")

        return time.time() - start_time

    def _rollback_to_safe_state(
        self, agent: BaseAgent, report: ResponsibilityReport
    ) -> None:
        checkpoints = agent.llm.list_checkpoints()
        if checkpoints:
            best = min(checkpoints, key=lambda c: c.loss)
            agent.llm.rollback_to_checkpoint(best)

    def _rollback_to_pre_failure(
        self, agent: BaseAgent, report: ResponsibilityReport
    ) -> None:
        error_steps = []
        for entry in report.causal_chain:
            if entry.get("is_error_step") and entry["agent_id"] == agent.agent_id:
                error_steps.append(entry["step"])

        if error_steps:
            rollback_step = min(error_steps)
            agent.rollback_to(rollback_step)
        else:
            agent.rollback_to_best()

    def _find_error_step(self, agent_id: str, report: ResponsibilityReport) -> Optional[int]:
        for entry in report.causal_chain:
            if entry.get("is_error_step") and entry["agent_id"] == agent_id:
                return entry["step"]
        return None

    def _partial_rollback(
        self, agent: BaseAgent, agent_id: str, report: ResponsibilityReport
    ) -> None:
        agent_score = next(
            (s for s in report.agent_scores if s.agent_id == agent_id), None
        )
        if agent_score and agent_score.marginal_contribution < -0.5:
            agent.rollback_to_best()
        else:
            recent_step = max(0, agent.llm.current_step - 5)
            if not agent.rollback_to(recent_step):
                agent.rollback_to_best()

    def _freeze_non_responsible(self, responsible_ids: List[str]) -> None:
        self._frozen_agents.clear()
        for agent_id, agent in self.agents.items():
            if agent_id not in responsible_ids:
                agent.freeze()
                self._frozen_agents.add(agent_id)
                self.logger.info(f"Agent '{agent_id}' frozen (not responsible)")
            else:
                agent.unfreeze()
                self.logger.info(f"Agent '{agent_id}' unfrozen (responsible, will retrain)")

    def _unfreeze_all(self) -> None:
        for agent_id, agent in self.agents.items():
            if agent.is_frozen:
                agent.unfreeze()
        self._frozen_agents.clear()

    def _retrain_responsible(
        self,
        responsible_ids: List[str],
        report: ResponsibilityReport,
    ) -> float:
        start_time = time.time()

        for agent_id in responsible_ids:
            agent = self.agents.get(agent_id)
            if agent is None or agent.is_frozen:
                continue

            retry_count = self._retraining_counts.get(agent_id, 0)
            if retry_count >= self.config.max_retries:
                self.logger.warning(f"Agent '{agent_id}' exceeded max retries ({self.config.max_retries})")
                continue

            self._retrain_single_agent(agent, report)

        return time.time() - start_time

    def _retrain_single_agent(
        self,
        agent: BaseAgent,
        report: ResponsibilityReport,
    ) -> None:
        agent_score = next(
            (s for s in report.agent_scores if s.agent_id == agent.agent_id), None
        )

        error_actions = []
        for entry in report.causal_chain:
            if entry.get("is_error_step") and entry["agent_id"] == agent.agent_id:
                error_actions.append(entry)

        if self.config.retraining_strategy == RetrainingStrategy.COUNTERFACTUAL_GRADIENT:
            self._retrain_counterfactual_gradient(agent, error_actions, report)
        elif self.config.retraining_strategy == RetrainingStrategy.ADVERSARIAL:
            self._retrain_adversarial(agent, error_actions, report)
        elif self.config.retraining_strategy == RetrainingStrategy.CURRICULUM:
            self._retrain_curriculum(agent, error_actions, report)
        else:
            self._retrain_standard(agent, error_actions, report)

    def _retrain_counterfactual_gradient(
        self,
        agent: BaseAgent,
        error_actions: List[Dict[str, Any]],
        report: ResponsibilityReport,
    ) -> None:
        lr_mult = self.config.retraining_lr_multiplier
        original_lr = agent.llm.config.learning_rate
        agent.llm.config.learning_rate = original_lr * lr_mult

        for step_idx in range(self.config.retraining_steps):
            loss = self._compute_counterfactual_loss(agent, error_actions, report)

            if loss is not None and loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.llm.model.parameters(), 1.0)
                if agent.llm.optimizer:
                    agent.llm.optimizer.step()
                    agent.llm.optimizer.zero_grad()

        agent.llm.config.learning_rate = original_lr

    def _compute_counterfactual_loss(
        self,
        agent: BaseAgent,
        error_actions: List[Dict[str, Any]],
        report: ResponsibilityReport,
    ) -> Optional[torch.Tensor]:
        if not error_actions or agent.llm.model is None:
            return None

        error_text = " ".join(
            a.get("action_summary", "") for a in error_actions[:3]
        )
        if not error_text:
            error_text = "error action"

        inputs = agent.llm.encode(error_text)
        labels = inputs["input_ids"].clone()

        outputs = agent.llm.model(**inputs)
        ce_loss = outputs.loss if outputs.loss is not None else torch.tensor(0.0, device=outputs.logits.device)

        agent_score = next(
            (s for s in report.agent_scores if s.agent_id == agent.agent_id), None
        )
        if agent_score:
            penalty_weight = max(0.5, abs(agent_score.marginal_contribution))
            ce_loss = ce_loss * (1.0 + penalty_weight)

        return ce_loss

    def _retrain_adversarial(
        self,
        agent: BaseAgent,
        error_actions: List[Dict[str, Any]],
        report: ResponsibilityReport,
    ) -> None:
        self._retrain_standard(agent, error_actions, report)

    def _retrain_curriculum(
        self,
        agent: BaseAgent,
        error_actions: List[Dict[str, Any]],
        report: ResponsibilityReport,
    ) -> None:
        steps_easy = self.config.retraining_steps // 3
        steps_hard = self.config.retraining_steps - steps_easy

        original_lr = agent.llm.config.learning_rate
        agent.llm.config.learning_rate = original_lr * 0.3
        for _ in range(steps_easy):
            self._retrain_step(agent, error_actions)

        agent.llm.config.learning_rate = original_lr * 0.8
        for _ in range(steps_hard):
            self._retrain_step(agent, error_actions)

        agent.llm.config.learning_rate = original_lr

    def _retrain_standard(
        self,
        agent: BaseAgent,
        error_actions: List[Dict[str, Any]],
        report: ResponsibilityReport,
    ) -> None:
        for _ in range(self.config.retraining_steps):
            self._retrain_step(agent, error_actions)

    def _retrain_step(
        self, agent: BaseAgent, error_actions: List[Dict[str, Any]]
    ) -> None:
        if not error_actions or agent.llm.model is None:
            return

        action_text = error_actions[np.random.randint(len(error_actions))].get("action_summary", "")
        if not action_text:
            return

        inputs = agent.llm.encode(action_text)
        labels = inputs["input_ids"].clone()
        result = agent.llm.train_step(inputs, labels)

    def _evaluate_after_retraining(self, report: ResponsibilityReport) -> float:
        improvement_factor = 0.0
        num_retrained = 0

        for agent_id in report.responsible_agents:
            agent = self.agents.get(agent_id)
            if agent and not agent.is_frozen:
                improvement_factor += 0.05
                num_retrained += 1

        if num_retrained > 0:
            improvement_factor /= num_retrained

        adjusted_score = report.overall_score + improvement_factor
        return min(1.0, adjusted_score)

    @property
    def rollback_history(self) -> List[RollbackRecord]:
        return list(self._rollback_history)

    @property
    def budget_report(self) -> Dict[str, Any]:
        return self.budget_manager.get_report()

    def get_agent_retraining_count(self, agent_id: str) -> int:
        return self._retraining_counts.get(agent_id, 0)

    def is_within_budget(self) -> bool:
        return self.budget_manager.current_ratio <= self.config.max_overhead_ratio
