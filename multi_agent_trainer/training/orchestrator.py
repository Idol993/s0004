"""训练编排器 - 多智能体协作训练的主控循环

负责：
1. 协调五个智能体的执行流程
2. 在任务失败时触发因果责任推断
3. 调用选择性回滚与重训练
4. 监控训练时间复杂度和通信开销
5. 确保总开销不超过训练总成本的15%
"""

import time
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import torch
import numpy as np

from ..agents.base_agent import BaseAgent, ActionRecord
from ..agents.planner import PlannerAgent
from ..agents.executor import ExecutorAgent
from ..agents.evaluator import EvaluatorAgent
from ..agents.memory import MemoryAgent
from ..agents.reflector import ReflectorAgent
from ..agents.llm_backbone import LLMConfig
from ..communication import CentralMessageBus
from ..responsibility.causal_inference import (
    CausalResponsibilityInferencer,
    CounterfactualConfig,
    ResponsibilityReport,
)
from ..responsibility.mdp_credit import MDPCreditAssignmentNetwork, MDPCreditConfig
from .rollback_manager import SelectiveRollbackManager, RollbackConfig, RollbackRecord
from ..utils import (
    AgentRole,
    MessageType,
    TrainingMetrics,
    Timer,
    set_seed,
    setup_logger,
    ensure_dir,
)


@dataclass
class TrainingConfig:
    max_episodes: int = 1000
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    save_interval: int = 10
    eval_interval: int = 5
    max_overhead_ratio: float = 0.15
    device: str = "cuda"
    seed: int = 42
    log_dir: str = "./logs"
    checkpoint_base_dir: str = "./checkpoints"


@dataclass
class EpisodeResult:
    episode_id: str
    success: bool
    score: float
    num_steps: int
    responsibility_report: Optional[ResponsibilityReport]
    rollback_record: Optional[RollbackRecord]
    episode_time: float
    overhead_time: float


class TrainingOrchestrator:
    def __init__(
        self,
        config: TrainingConfig,
        llm_configs: Optional[Dict[str, LLMConfig]] = None,
        counterfactual_config: Optional[CounterfactualConfig] = None,
        mdp_config: Optional[MDPCreditConfig] = None,
        rollback_config: Optional[RollbackConfig] = None,
    ):
        self.config = config
        set_seed(config.seed)
        ensure_dir(config.log_dir)
        ensure_dir(config.checkpoint_base_dir)

        self.logger = setup_logger("orchestrator", config.log_dir)
        self.metrics = TrainingMetrics()
        self._timer = Timer()

        self.message_bus = CentralMessageBus(
            enable_logging=True,
            log_dir=config.log_dir,
        )
        self.message_bus.start()

        if llm_configs is None:
            llm_configs = {}
        default_llm = LLMConfig(device=config.device)
        for role in ["planner", "executor", "evaluator", "memory", "reflector"]:
            if role not in llm_configs:
                llm_configs[role] = LLMConfig(
                    model_name=default_llm.model_name,
                    device=config.device,
                )

        self.agents: Dict[str, BaseAgent] = {}
        self._init_agents(llm_configs)

        self.responsibility_inferencer = CausalResponsibilityInferencer(
            agents=self.agents,
            config=counterfactual_config or CounterfactualConfig(),
            log_dir=config.log_dir,
        )

        self.mdp_credit_network = MDPCreditAssignmentNetwork(
            config=mdp_config or MDPCreditConfig(),
            agents=self.agents,
            log_dir=config.log_dir,
        )

        self.rollback_manager = SelectiveRollbackManager(
            agents=self.agents,
            config=rollback_config or RollbackConfig(max_overhead_ratio=config.max_overhead_ratio),
            mdp_credit_network=self.mdp_credit_network,
            log_dir=config.log_dir,
        )

        self._episode_results: List[EpisodeResult] = []
        self._current_episode: int = 0
        self._running: bool = False

        self.logger.info("Training Orchestrator initialized with 5 agents")

    def _init_agents(self, llm_configs: Dict[str, LLMConfig]) -> None:
        ckpt_base = self.config.checkpoint_base_dir

        self.agents["planner"] = PlannerAgent(
            agent_id="planner",
            llm_config=llm_configs.get("planner", LLMConfig()),
            message_bus=self.message_bus,
            checkpoint_dir=f"{ckpt_base}/planner",
            log_dir=self.config.log_dir,
        )
        self.agents["executor"] = ExecutorAgent(
            agent_id="executor",
            llm_config=llm_configs.get("executor", LLMConfig()),
            message_bus=self.message_bus,
            checkpoint_dir=f"{ckpt_base}/executor",
            log_dir=self.config.log_dir,
        )
        self.agents["evaluator"] = EvaluatorAgent(
            agent_id="evaluator",
            llm_config=llm_configs.get("evaluator", LLMConfig()),
            message_bus=self.message_bus,
            checkpoint_dir=f"{ckpt_base}/evaluator",
            log_dir=self.config.log_dir,
        )
        self.agents["memory"] = MemoryAgent(
            agent_id="memory",
            llm_config=llm_configs.get("memory", LLMConfig()),
            message_bus=self.message_bus,
            checkpoint_dir=f"{ckpt_base}/memory",
            log_dir=self.config.log_dir,
        )
        self.agents["reflector"] = ReflectorAgent(
            agent_id="reflector",
            llm_config=llm_configs.get("reflector", LLMConfig()),
            message_bus=self.message_bus,
            checkpoint_dir=f"{ckpt_base}/reflector",
            log_dir=self.config.log_dir,
        )

    def run_episode(
        self,
        task: str,
        episode_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> EpisodeResult:
        if episode_id is None:
            episode_id = f"ep_{self._current_episode}"

        episode_start = time.time()
        overhead_start = time.time()
        overhead_time = 0.0

        self.logger.info(f"=== Starting episode {episode_id} ===")

        for agent in self.agents.values():
            agent.reset_episode(episode_id)

        trajectory: List[ActionRecord] = []

        with self._timer:
            planner = self.agents["planner"]
            plan_observation = {
                "task": task,
                "context": context or {},
                "constraints": {},
            }
            plan_result = planner.step(plan_observation)
            trajectory.extend(planner.action_history)

        plan = plan_result.get("sub_tasks", [])
        self.logger.info(f"Planner created {len(plan)} sub-tasks")

        execution_results: List[Dict[str, Any]] = []

        for i, sub_task in enumerate(plan):
            with self._timer:
                executor = self.agents["executor"]
                exec_observation = {
                    "sub_task": sub_task,
                    "plan_context": {"task": task, "sub_task_index": i},
                    "available_tools": [],
                }
                exec_result = executor.step(exec_observation)
                trajectory.extend(
                    [a for a in executor.action_history if a.episode_id == episode_id]
                )

            execution_results.append(exec_result)

            mem = self.agents["memory"]
            mem_observation = {
                "query": f"execution result for sub_task {i}",
                "query_type": "store",
                "data_to_store": {
                    "sub_task": sub_task,
                    "result": exec_result,
                    "episode": episode_id,
                    "reward": exec_result.get("confidence", 0.5),
                },
            }
            mem.step(mem_observation)

        with self._timer:
            evaluator = self.agents["evaluator"]
            eval_observation = {
                "task_result": execution_results,
                "original_task": task,
                "execution_log": execution_results,
                "criteria": {},
            }
            eval_result = evaluator.step(eval_observation)

        score = eval_result.get("score", 0.0)
        success = eval_result.get("success", False)
        self.logger.info(f"Evaluation: score={score:.4f}, success={success}")

        responsibility_report = None
        rollback_record = None

        if not success:
            overhead_start_cf = time.time()

            responsibility_report = self.responsibility_inferencer.infer_responsibility(
                episode_id=episode_id,
                trajectory=self._collect_trajectory(episode_id),
                outcome_score=score,
                task_success=success,
            )

            self.logger.info(
                f"Responsible agents: {responsibility_report.responsible_agents}"
            )

            delayed_credits = self.mdp_credit_network.compute_delayed_credit_assignment(
                trajectory=self._collect_trajectory(episode_id),
                final_reward=score,
                episode_id=episode_id,
            )

            self._apply_delayed_credits(delayed_credits, episode_id)

            rollback_record = self.rollback_manager.execute_rollback(
                responsibility_report=responsibility_report,
                episode_id=episode_id,
            )

            overhead_time = time.time() - overhead_start_cf

            self.rollback_manager.budget_manager.record_inference_time(
                responsibility_report.inference_time
            )
        else:
            with self._timer:
                reflector = self.agents["reflector"]
                reflect_observation = {
                    "evaluation": eval_result,
                    "execution_log": execution_results,
                    "plan": plan,
                }
                reflector.step(reflect_observation)

        with self._timer:
            reflector = self.agents["reflector"]
            reflect_observation = {
                "evaluation": eval_result,
                "execution_log": execution_results,
                "plan": plan,
                "failure_info": {} if success else {"score": score},
            }
            reflector.step(reflect_observation)

        episode_time = time.time() - episode_start

        if success:
            self.metrics.success_count += 1
        else:
            self.metrics.failure_count += 1

        self.metrics.total_time += episode_time
        self.metrics.communication_time = self.message_bus.get_metrics().communication_time

        self.responsibility_inferencer.update_training_time(episode_time - overhead_time)
        self.rollback_manager.budget_manager.record_training_time(episode_time - overhead_time)

        self._record_transitions_to_mdp(episode_id)

        result = EpisodeResult(
            episode_id=episode_id,
            success=success,
            score=score,
            num_steps=len(execution_results),
            responsibility_report=responsibility_report,
            rollback_record=rollback_record,
            episode_time=episode_time,
            overhead_time=overhead_time,
        )
        self._episode_results.append(result)
        self._current_episode += 1

        self.logger.info(
            f"Episode {episode_id} complete: "
            f"success={success}, score={score:.4f}, "
            f"time={episode_time:.2f}s, overhead={overhead_time:.2f}s"
        )

        return result

    def _collect_trajectory(self, episode_id: str) -> List[ActionRecord]:
        all_records = []
        for agent in self.agents.values():
            all_records.extend(agent.get_actions_in_episode(episode_id))
        all_records.sort(key=lambda r: (r.step, r.agent_id))
        return all_records

    def _apply_delayed_credits(
        self, credits: Dict[str, float], episode_id: str
    ) -> None:
        for agent_id, credit in credits.items():
            agent = self.agents.get(agent_id)
            if agent:
                agent.receive_reward(credit)

    def _record_transitions_to_mdp(self, episode_id: str) -> None:
        for agent in self.agents.values():
            records = agent.get_actions_in_episode(episode_id)
            for i, record in enumerate(records):
                next_obs = records[i + 1].observation if i + 1 < len(records) else record.observation
                done = (i == len(records) - 1)
                reward = record.reward if record.reward is not None else 0.0

                self.mdp_credit_network.record_transition(
                    agent_id=record.agent_id,
                    observation=record.observation,
                    action=record.action if isinstance(record.action, dict) else {"action": record.action},
                    reward=reward,
                    next_observation=next_obs,
                    done=done,
                    step=record.step,
                    episode_id=episode_id,
                )

    def train(
        self,
        tasks: List[str],
        max_episodes: Optional[int] = None,
        callback: Optional[Callable[[EpisodeResult], None]] = None,
    ) -> Dict[str, Any]:
        if max_episodes is None:
            max_episodes = self.config.max_episodes

        self._running = True
        self.logger.info(f"Starting training for {max_episodes} episodes")

        all_results = []
        for ep in range(max_episodes):
            if not self._running:
                self.logger.info("Training stopped by user")
                break

            task = tasks[ep % len(tasks)]
            result = self.run_episode(task, episode_id=f"ep_{ep}")
            all_results.append(result)

            if callback:
                callback(result)

            if (ep + 1) % self.config.eval_interval == 0:
                self._log_progress(ep, all_results)

            if (ep + 1) % self.config.save_interval == 0:
                self._save_all_checkpoints(ep)

            if not self.rollback_manager.is_within_budget():
                self.logger.warning(
                    f"Overhead budget exceeded at episode {ep}: "
                    f"{self.rollback_manager.budget_report}"
                )

            credit_loss = self.mdp_credit_network.train_credit_network(num_steps=5)
            self.mdp_credit_network.clear_buffer()

        self._running = False
        return self._compile_training_summary(all_results)

    def _log_progress(self, episode: int, results: List[EpisodeResult]) -> None:
        recent = results[-self.config.eval_interval:]
        avg_score = np.mean([r.score for r in recent])
        success_rate = np.mean([1.0 if r.success else 0.0 for r in recent])
        avg_time = np.mean([r.episode_time for r in recent])
        overhead_pct = np.mean([r.overhead_time / max(r.episode_time, 1e-8) for r in recent])

        self.logger.info(
            f"[Episode {episode}] "
            f"avg_score={avg_score:.4f}, "
            f"success_rate={success_rate:.2%}, "
            f"avg_time={avg_time:.2f}s, "
            f"overhead={overhead_pct:.2%}"
        )

    def _save_all_checkpoints(self, episode: int) -> None:
        for agent_id, agent in self.agents.items():
            agent.save_checkpoint(episode, loss=0.0, metric_score=agent.local_reward)

    def _compile_training_summary(self, results: List[EpisodeResult]) -> Dict[str, Any]:
        if not results:
            return {}

        scores = [r.score for r in results]
        successes = [r.success for r in results]
        times = [r.episode_time for r in results]
        overheads = [r.overhead_time for r in results]

        num_rollbacks = sum(1 for r in results if r.rollback_record is not None)
        rollback_successes = sum(
            1 for r in results
            if r.rollback_record and r.rollback_record.success
        )

        total_overhead_time = sum(overheads)
        total_time = sum(times)
        overhead_ratio = total_overhead_time / max(total_time, 1e-8)

        return {
            "total_episodes": len(results),
            "final_avg_score": float(np.mean(scores[-10:])),
            "best_score": float(max(scores)),
            "overall_success_rate": float(np.mean(successes)),
            "total_training_time": total_time,
            "total_overhead_time": total_overhead_time,
            "overhead_ratio": overhead_ratio,
            "overhead_within_budget": overhead_ratio <= self.config.max_overhead_ratio,
            "num_rollbacks": num_rollbacks,
            "rollback_success_rate": (
                rollback_successes / max(num_rollbacks, 1)
            ),
            "budget_report": self.rollback_manager.budget_report,
            "responsibility_stats": self.responsibility_inferencer.get_overhead_stats(),
        }

    def stop(self) -> None:
        self._running = False
        self.message_bus.stop()

    def get_agent_statistics(self) -> Dict[str, Dict[str, Any]]:
        return {aid: agent.get_statistics() for aid, agent in self.agents.items()}

    @property
    def episode_results(self) -> List[EpisodeResult]:
        return list(self._episode_results)

    @property
    def is_running(self) -> bool:
        return self._running
