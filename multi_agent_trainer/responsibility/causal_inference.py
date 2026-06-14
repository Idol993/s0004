"""因果责任推断模块 - 基于反事实基线的边际贡献计算

核心思想：
1. 反事实基线 (Counterfactual Baseline): 假设某个智能体采取默认/随机动作，
   整体结果会变成什么样？实际结果与反事实结果的差异就是该智能体的边际贡献。

2. Shapley值近似: 利用蒙特卡洛采样近似计算每个智能体的Shapley值，
   公平分配整体回报中各智能体的贡献。

3. 责任归因: 当任务失败时，边际贡献为负（即该智能体的存在使结果变差）
   的智能体被识别为责任智能体。

时间复杂度控制:
- 使用蒙特卡洛近似而非精确Shapley值计算，将O(n!)降低到O(n * num_samples)
- 反事实基线使用随机策略替代完整策略重放，避免指数级展开
- 总开销通过采样数和近似精度参数严格控制
"""

import time
import itertools
import math
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ..agents.base_agent import BaseAgent, ActionRecord
from ..utils import AgentRole, Timer, setup_logger


@dataclass
class CounterfactualConfig:
    num_samples: int = 5
    baseline_type: str = "random"
    shapley_enabled: bool = True
    shapley_num_permutations: int = 20
    default_action_noise: float = 0.1
    counterfactual_discount: float = 0.95


@dataclass
class ResponsibilityScore:
    agent_id: str
    agent_role: str
    marginal_contribution: float
    shapley_value: float
    counterfactual_impact: float
    is_responsible: bool
    confidence: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResponsibilityReport:
    episode_id: str
    task_success: bool
    overall_score: float
    agent_scores: List[ResponsibilityScore]
    responsible_agents: List[str]
    causal_chain: List[Dict[str, Any]]
    inference_time: float
    num_samples_used: int
    overhead_ratio: float


class CounterfactualSimulator:
    def __init__(
        self,
        config: CounterfactualConfig,
        agents: Dict[str, BaseAgent],
        log_dir: str = "./logs",
    ):
        self.config = config
        self.agents = agents
        self.logger = setup_logger("counterfactual_sim", log_dir)

        self._default_action_cache: Dict[str, Dict[str, Any]] = {}
        self._rng = np.random.RandomState(42)

    def generate_default_action(self, agent_id: str, original_action: Dict[str, Any]) -> Dict[str, Any]:
        if agent_id in self._default_action_cache:
            return self._default_action_cache[agent_id]

        if self.config.baseline_type == "random":
            default = self._generate_random_action(original_action)
        elif self.config.baseline_type == "mean":
            default = self._generate_mean_action(original_action)
        else:
            default = self._generate_noisy_action(original_action)

        self._default_action_cache[agent_id] = default
        return default

    def _generate_random_action(self, original: Dict[str, Any]) -> Dict[str, Any]:
        default = {}
        for key, value in original.items():
            if isinstance(value, (int, float)):
                default[key] = float(self._rng.randn() * self.config.default_action_noise)
            elif isinstance(value, str):
                default[key] = f"default_{key}"
            elif isinstance(value, dict):
                default[key] = self._generate_random_action(value)
            elif isinstance(value, list):
                default[key] = []
            elif isinstance(value, bool):
                default[key] = False
            else:
                default[key] = value
        return default

    def _generate_mean_action(self, original: Dict[str, Any]) -> Dict[str, Any]:
        default = {}
        for key, value in original.items():
            if isinstance(value, (int, float)):
                default[key] = value * 0.5
            elif isinstance(value, dict):
                default[key] = self._generate_mean_action(value)
            else:
                default[key] = value
        return default

    def _generate_noisy_action(self, original: Dict[str, Any]) -> Dict[str, Any]:
        default = {}
        for key, value in original.items():
            if isinstance(value, (int, float)):
                noise = self._rng.randn() * self.config.default_action_noise * abs(value + 1e-8)
                default[key] = value + noise
            elif isinstance(value, dict):
                default[key] = self._generate_noisy_action(value)
            else:
                default[key] = value
        return default

    def simulate_counterfactual(
        self,
        agent_id: str,
        original_trajectory: List[ActionRecord],
        outcome_score: float,
        evaluation_fn: Optional[Callable] = None,
    ) -> float:
        self._default_action_cache.clear()

        counterfactual_actions = []
        for record in original_trajectory:
            if record.agent_id == agent_id:
                default_action = self.generate_default_action(agent_id, record.action)
                cf_record = ActionRecord(
                    action_id=record.action_id + "_cf",
                    agent_id=record.agent_id,
                    agent_role=record.agent_role,
                    step=record.step,
                    episode_id=record.episode_id,
                    observation=record.observation,
                    action=default_action,
                )
                counterfactual_actions.append(cf_record)
            else:
                counterfactual_actions.append(record)

        if evaluation_fn is not None:
            cf_score = evaluation_fn(counterfactual_actions)
        else:
            cf_score = self._estimate_counterfactual_score(
                agent_id, original_trajectory, outcome_score
            )

        return cf_score

    def _estimate_counterfactual_score(
        self,
        agent_id: str,
        original_trajectory: List[ActionRecord],
        original_score: float,
    ) -> float:
        agent_records = [r for r in original_trajectory if r.agent_id == agent_id]
        if not agent_records:
            return original_score

        num_interventions = len(agent_records)
        total_steps = len(original_trajectory)

        impact_factor = num_interventions / max(total_steps, 1)
        discount = self.config.counterfactual_discount

        estimated_cf_score = original_score * (1.0 - impact_factor * (1.0 - discount))

        for record in agent_records:
            action_confidence = record.action.get("confidence", 0.5) if isinstance(record.action, dict) else 0.5
            if action_confidence < 0.3:
                estimated_cf_score *= 1.0 + (0.3 - action_confidence) * 0.5

        return max(0.0, min(1.0, estimated_cf_score))

    def compute_marginal_contribution(
        self,
        agent_id: str,
        original_trajectory: List[ActionRecord],
        outcome_score: float,
        num_samples: Optional[int] = None,
        evaluation_fn: Optional[Callable] = None,
    ) -> Tuple[float, float]:
        if num_samples is None:
            num_samples = self.config.num_samples

        cf_scores = []
        for _ in range(num_samples):
            self._default_action_cache.clear()
            cf_score = self.simulate_counterfactual(
                agent_id, original_trajectory, outcome_score, evaluation_fn
            )
            cf_scores.append(cf_score)

        avg_cf_score = np.mean(cf_scores) if cf_scores else outcome_score
        marginal_contribution = outcome_score - avg_cf_score

        std_cf = np.std(cf_scores) if len(cf_scores) > 1 else 0.0
        confidence = 1.0 / (1.0 + std_cf)

        return float(marginal_contribution), float(confidence)


class ShapleyValueEstimator:
    def __init__(
        self,
        counterfactual_sim: CounterfactualSimulator,
        num_permutations: int = 20,
    ):
        self.counterfactual_sim = counterfactual_sim
        self.num_permutations = num_permutations
        self.logger = counterfactual_sim.logger

    def compute_shapley_values(
        self,
        agent_ids: List[str],
        original_trajectory: List[ActionRecord],
        outcome_score: float,
        evaluation_fn: Optional[Callable] = None,
    ) -> Dict[str, float]:
        n = len(agent_ids)
        if n == 0:
            return {}

        if n <= 3:
            return self._exact_shapley(agent_ids, original_trajectory, outcome_score, evaluation_fn)
        return self._monte_carlo_shapley(agent_ids, original_trajectory, outcome_score, evaluation_fn)

    def _exact_shapley(
        self,
        agent_ids: List[str],
        original_trajectory: List[ActionRecord],
        outcome_score: float,
        evaluation_fn: Optional[Callable],
    ) -> Dict[str, float]:
        n = len(agent_ids)
        shapley_values = {aid: 0.0 for aid in agent_ids}

        for perm in itertools.permutations(agent_ids):
            for i, agent_id in enumerate(perm):
                coalition_before = set(perm[:i])
                coalition_with = set(perm[:i + 1])

                v_before = self._evaluate_coalition(
                    coalition_before, agent_ids, original_trajectory, outcome_score, evaluation_fn
                )
                v_with = self._evaluate_coalition(
                    coalition_with, agent_ids, original_trajectory, outcome_score, evaluation_fn
                )

                shapley_values[agent_id] += (v_with - v_before)

        for aid in shapley_values:
            shapley_values[aid] /= math.factorial(n)

        return shapley_values

    def _monte_carlo_shapley(
        self,
        agent_ids: List[str],
        original_trajectory: List[ActionRecord],
        outcome_score: float,
        evaluation_fn: Optional[Callable],
    ) -> Dict[str, float]:
        n = len(agent_ids)
        shapley_values = {aid: 0.0 for aid in agent_ids}

        for _ in range(self.num_permutations):
            perm = list(agent_ids)
            np.random.shuffle(perm)

            for i, agent_id in enumerate(perm):
                coalition_before = set(perm[:i])
                coalition_with = set(perm[:i + 1])

                v_before = self._evaluate_coalition(
                    coalition_before, agent_ids, original_trajectory, outcome_score, evaluation_fn
                )
                v_with = self._evaluate_coalition(
                    coalition_with, agent_ids, original_trajectory, outcome_score, evaluation_fn
                )

                shapley_values[agent_id] += (v_with - v_before)

        for aid in shapley_values:
            shapley_values[aid] /= self.num_permutations

        return shapley_values

    def _evaluate_coalition(
        self,
        coalition: Set[str],
        all_agents: List[str],
        original_trajectory: List[ActionRecord],
        outcome_score: float,
        evaluation_fn: Optional[Callable],
    ) -> float:
        non_coalition_agents = set(all_agents) - coalition

        if not non_coalition_agents:
            return outcome_score

        modified_trajectory = []
        for record in original_trajectory:
            if record.agent_id in non_coalition_agents:
                default_action = self.counterfactual_sim.generate_default_action(
                    record.agent_id, record.action
                )
                cf_record = ActionRecord(
                    action_id=record.action_id + "_cf",
                    agent_id=record.agent_id,
                    agent_role=record.agent_role,
                    step=record.step,
                    episode_id=record.episode_id,
                    observation=record.observation,
                    action=default_action,
                )
                modified_trajectory.append(cf_record)
            else:
                modified_trajectory.append(record)

        if evaluation_fn is not None:
            return evaluation_fn(modified_trajectory)

        non_coalition_fraction = len(non_coalition_agents) / len(all_agents)
        estimated = outcome_score * (1.0 - non_coalition_fraction * 0.5)
        return max(0.0, min(1.0, estimated))


class CausalResponsibilityInferencer:
    def __init__(
        self,
        agents: Dict[str, BaseAgent],
        config: Optional[CounterfactualConfig] = None,
        responsibility_threshold: float = 0.3,
        max_responsible_agents: int = 3,
        log_dir: str = "./logs",
    ):
        self.agents = agents
        self.config = config or CounterfactualConfig()
        self.responsibility_threshold = responsibility_threshold
        self.max_responsible_agents = max_responsible_agents
        self.log_dir = log_dir

        self.logger = setup_logger("responsibility_inferencer", log_dir)
        self.cf_simulator = CounterfactualSimulator(self.config, agents, log_dir)
        self.shapley_estimator = ShapleyValueEstimator(
            self.cf_simulator,
            num_permutations=self.config.shapley_num_permutations,
        )

        self._inference_timer = Timer()
        self._total_inference_time: float = 0.0
        self._total_training_time: float = 0.0

    def infer_responsibility(
        self,
        episode_id: str,
        trajectory: List[ActionRecord],
        outcome_score: float,
        task_success: bool,
        evaluation_fn: Optional[Callable] = None,
    ) -> ResponsibilityReport:
        self._inference_timer.reset()
        start_time = time.time()

        with self._inference_timer:
            agent_ids = list(set(r.agent_id for r in trajectory))

            marginal_contributions: Dict[str, Tuple[float, float]] = {}
            for agent_id in agent_ids:
                mc, confidence = self.cf_simulator.compute_marginal_contribution(
                    agent_id, trajectory, outcome_score,
                    num_samples=self.config.num_samples,
                    evaluation_fn=evaluation_fn,
                )
                marginal_contributions[agent_id] = (mc, confidence)

            shapley_values: Dict[str, float] = {}
            if self.config.shapley_enabled:
                shapley_values = self.shapley_estimator.compute_shapley_values(
                    agent_ids, trajectory, outcome_score, evaluation_fn
                )

            agent_scores = []
            for agent_id in agent_ids:
                mc, confidence = marginal_contributions.get(agent_id, (0.0, 0.0))
                sv = shapley_values.get(agent_id, 0.0)
                agent = self.agents.get(agent_id)
                role = agent.role.value if agent else "unknown"

                cf_impact = mc

                is_responsible = (
                    not task_success
                    and mc < -self.responsibility_threshold
                )

                score = ResponsibilityScore(
                    agent_id=agent_id,
                    agent_role=role,
                    marginal_contribution=mc,
                    shapley_value=sv,
                    counterfactual_impact=cf_impact,
                    is_responsible=is_responsible,
                    confidence=confidence,
                    details={
                        "num_actions": sum(1 for r in trajectory if r.agent_id == agent_id),
                        "avg_confidence": self._avg_action_confidence(agent_id, trajectory),
                    },
                )
                agent_scores.append(score)

            responsible_agents = [
                s.agent_id for s in agent_scores if s.is_responsible
            ]
            responsible_agents.sort(
                key=lambda aid: next(
                    (s.marginal_contribution for s in agent_scores if s.agent_id == aid), 0.0
                )
            )
            responsible_agents = responsible_agents[:self.max_responsible_agents]

            causal_chain = self._build_causal_chain(trajectory, agent_scores, task_success)

        inference_time = time.time() - start_time
        self._total_inference_time += inference_time

        overhead_ratio = (
            self._total_inference_time / max(self._total_training_time, 1e-8)
            if self._total_training_time > 0 else 0.0
        )

        num_samples_used = len(agent_ids) * self.config.num_samples
        if self.config.shapley_enabled:
            num_samples_used += self.config.shapley_num_permutations * len(agent_ids)

        report = ResponsibilityReport(
            episode_id=episode_id,
            task_success=task_success,
            overall_score=outcome_score,
            agent_scores=agent_scores,
            responsible_agents=responsible_agents,
            causal_chain=causal_chain,
            inference_time=inference_time,
            num_samples_used=num_samples_used,
            overhead_ratio=overhead_ratio,
        )

        self.logger.info(
            f"Responsibility inference complete for episode {episode_id}: "
            f"responsible_agents={responsible_agents}, "
            f"overhead_ratio={overhead_ratio:.4f}"
        )

        return report

    def _avg_action_confidence(self, agent_id: str, trajectory: List[ActionRecord]) -> float:
        records = [r for r in trajectory if r.agent_id == agent_id]
        if not records:
            return 0.5
        confidences = []
        for r in records:
            if isinstance(r.action, dict):
                confidences.append(r.action.get("confidence", 0.5))
            else:
                confidences.append(0.5)
        return np.mean(confidences) if confidences else 0.5

    def _build_causal_chain(
        self,
        trajectory: List[ActionRecord],
        agent_scores: List[ResponsibilityScore],
        task_success: bool,
    ) -> List[Dict[str, Any]]:
        chain = []
        sorted_records = sorted(trajectory, key=lambda r: r.step)

        for record in sorted_records:
            agent_score = next(
                (s for s in agent_scores if s.agent_id == record.agent_id), None
            )
            contribution = agent_score.marginal_contribution if agent_score else 0.0

            chain.append({
                "step": record.step,
                "agent_id": record.agent_id,
                "agent_role": record.agent_role.value if isinstance(record.agent_role, AgentRole) else str(record.agent_role),
                "action_summary": str(record.action)[:200] if record.action else "",
                "marginal_contribution": contribution,
                "is_error_step": not task_success and contribution < -self.responsibility_threshold,
            })

        return chain

    def update_training_time(self, training_time: float) -> None:
        self._total_training_time += training_time

    def get_overhead_stats(self) -> Dict[str, float]:
        return {
            "total_inference_time": self._total_inference_time,
            "total_training_time": self._total_training_time,
            "overhead_ratio": (
                self._total_inference_time / max(self._total_training_time, 1e-8)
                if self._total_training_time > 0 else 0.0
            ),
        }

    def reset_stats(self) -> None:
        self._total_inference_time = 0.0
        self._total_training_time = 0.0
