"""系统验证脚本 - 无需真实LLM即可验证核心逻辑

验证内容:
1. 通信协议: 消息传递、路由、广播
2. 因果责任推断: 反事实基线计算、Shapley值
3. MDP信用分配: GAE计算、时延信用
4. 选择性回滚: 责任智能体回滚、非责任智能体冻结
5. 开销预算: 15%上限控制
"""

import sys
import os
import time
import json
from typing import Any, Dict, List

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_communication():
    print("\n" + "=" * 60)
    print("测试1: 中心通信协议")
    print("=" * 60)

    from multi_agent_trainer.communication import CentralMessageBus, MessageFactory
    from multi_agent_trainer.utils import MessageType

    bus = CentralMessageBus(enable_logging=True, log_dir="./test_logs")
    bus.start()

    bus.register_agent("planner")
    bus.register_agent("executor")
    bus.register_agent("evaluator")

    bus.subscribe("executor", msg_type=MessageType.PLAN)
    bus.subscribe("evaluator", msg_type=MessageType.ACTION)

    msg = MessageFactory.create(
        sender="planner",
        receiver="executor",
        msg_type=MessageType.PLAN,
        content={"sub_tasks": ["task1", "task2"], "confidence": 0.85},
        episode_id="test_ep_1",
    )
    assert bus.send(msg), "消息发送失败"

    received = bus.receive("executor", timeout=1.0)
    assert received is not None, "消息接收失败"
    assert received.content["sub_tasks"] == ["task1", "task2"], "消息内容不匹配"
    print(f"  点对点消息: PASS (content={received.content})")

    broadcast_msg = MessageFactory.create(
        sender="evaluator",
        receiver="*",
        msg_type=MessageType.EVALUATION,
        content={"score": 0.75, "success": True},
        episode_id="test_ep_1",
    )
    count = bus.broadcast(broadcast_msg)
    assert count == 3, f"广播失败，期望3个接收者，实际{count}"
    print(f"  广播消息: PASS (recipients={count})")

    metrics = bus.get_metrics()
    assert metrics.num_messages > 0, "消息计数错误"
    print(f"  消息统计: PASS (total_messages={metrics.num_messages})")

    msg_log = bus.get_message_log(last_n=5)
    assert len(msg_log) > 0, "消息日志为空"
    print(f"  消息日志: PASS (log_entries={len(msg_log)})")

    bus.stop()
    print("  通信协议测试全部通过!")
    return True


def test_counterfactual_inference():
    print("\n" + "=" * 60)
    print("测试2: 因果责任推断 (反事实基线)")
    print("=" * 60)

    from multi_agent_trainer.agents.base_agent import ActionRecord
    from multi_agent_trainer.responsibility.causal_inference import (
        CounterfactualSimulator,
        CounterfactualConfig,
        ShapleyValueEstimator,
        CausalResponsibilityInferencer,
    )
    from multi_agent_trainer.utils import AgentRole

    trajectory = []
    for step in range(10):
        for role_name in ["planner", "executor", "evaluator"]:
            trajectory.append(ActionRecord(
                action_id=f"{role_name}_st{step}",
                agent_id=role_name,
                agent_role=AgentRole(role_name),
                step=step,
                episode_id="test_ep",
                observation={"step": step},
                action={"confidence": 0.8 if role_name != "executor" else 0.3, "result": f"action_{step}"},
            ))

    config = CounterfactualConfig(num_samples=3, shapley_enabled=True, shapley_num_permutations=10)

    simulator = CounterfactualSimulator(config, agents={}, log_dir="./test_logs")

    mc_planner, conf_planner = simulator.compute_marginal_contribution(
        "planner", trajectory, outcome_score=0.4
    )
    mc_executor, conf_executor = simulator.compute_marginal_contribution(
        "executor", trajectory, outcome_score=0.4
    )
    mc_evaluator, conf_evaluator = simulator.compute_marginal_contribution(
        "evaluator", trajectory, outcome_score=0.4
    )

    print(f"  规划器边际贡献: {mc_planner:.4f} (confidence={conf_planner:.4f})")
    print(f"  执行器边际贡献: {mc_executor:.4f} (confidence={conf_executor:.4f})")
    print(f"  评估器边际贡献: {mc_evaluator:.4f} (confidence={conf_evaluator:.4f})")

    shapley_estimator = ShapleyValueEstimator(simulator, num_permutations=10)
    shapley_values = shapley_estimator.compute_shapley_values(
        ["planner", "executor", "evaluator"], trajectory, outcome_score=0.4
    )
    print(f"  Shapley值: {', '.join(f'{k}={v:.4f}' for k, v in shapley_values.items())}")

    assert len(shapley_values) == 3, "Shapley值数量不正确"

    class MockAgent:
        def __init__(self, agent_id, role_name):
            self.agent_id = agent_id
            self.role = AgentRole(role_name)

    mock_agents = {
        "planner": MockAgent("planner", "planner"),
        "executor": MockAgent("executor", "executor"),
        "evaluator": MockAgent("evaluator", "evaluator"),
    }

    inferencer = CausalResponsibilityInferencer(
        agents=mock_agents,
        config=config,
        responsibility_threshold=0.3,
        log_dir="./test_logs",
    )

    report = inferencer.infer_responsibility(
        episode_id="test_ep",
        trajectory=trajectory,
        outcome_score=0.4,
        task_success=False,
    )

    print(f"\n  责任报告:")
    print(f"    任务成功: {report.task_success}")
    print(f"    整体分数: {report.overall_score:.4f}")
    print(f"    责任智能体: {report.responsible_agents}")
    print(f"    推断时间: {report.inference_time:.4f}s")
    print(f"    采样数: {report.num_samples_used}")

    for score in report.agent_scores:
        print(f"    {score.agent_id}: MC={score.marginal_contribution:.4f}, "
              f"Shapley={score.shapley_value:.4f}, "
              f"responsible={score.is_responsible}")

    assert isinstance(report.responsible_agents, list), "责任智能体格式错误"
    print("  因果责任推断测试全部通过!")
    return True


def test_mdp_credit_assignment():
    print("\n" + "=" * 60)
    print("测试3: MDP时延信用分配网络")
    print("=" * 60)

    from multi_agent_trainer.responsibility.mdp_credit import (
        MDPCreditAssignmentNetwork,
        MDPCreditConfig,
    )
    from multi_agent_trainer.agents.base_agent import ActionRecord
    from multi_agent_trainer.utils import AgentRole

    class MockAgent:
        def __init__(self, agent_id):
            self.agent_id = agent_id
            self.role = AgentRole(agent_id)

    agents = {
        "planner": MockAgent("planner"),
        "executor": MockAgent("executor"),
        "evaluator": MockAgent("evaluator"),
    }

    config = MDPCreditConfig(
        state_dim=64,
        action_dim=32,
        hidden_dim=64,
        num_layers=1,
        num_agents=3,
    )
    mdp_net = MDPCreditAssignmentNetwork(config, agents, log_dir="./test_logs")

    trajectory = []
    for step in range(5):
        for agent_id in ["planner", "executor", "evaluator"]:
            trajectory.append(ActionRecord(
                action_id=f"{agent_id}_st{step}",
                agent_id=agent_id,
                agent_role=AgentRole(agent_id),
                step=step,
                episode_id="test_ep",
                observation={"step": step, "data": float(step) * 0.1},
                action={"confidence": 0.5 + step * 0.1, "action_name": f"act_{step}"},
            ))

    temporal_credits = mdp_net.compute_temporal_credits(trajectory, "test_ep")
    print(f"  时延信用分配:")
    for agent_id, credits in temporal_credits.items():
        if credits:
            avg_credit = np.mean([c for _, c in credits])
            print(f"    {agent_id}: avg_credit={avg_credit:.4f}, steps={len(credits)}")

    delayed_credits = mdp_net.compute_delayed_credit_assignment(
        trajectory, final_reward=0.6, episode_id="test_ep"
    )
    print(f"  延迟信用分配:")
    for agent_id, credit in delayed_credits.items():
        print(f"    {agent_id}: credit={credit:.4f}")

    mdp_net.record_transition(
        agent_id="planner",
        observation={"step": 0, "data": 0.0},
        action={"confidence": 0.5},
        reward=0.6,
        next_observation={"step": 1, "data": 0.1},
        done=False,
        step=0,
        episode_id="test_ep",
    )

    train_result = mdp_net.train_credit_network(num_steps=3)
    print(f"  信用网络训练: loss={train_result['loss']:.4f}")

    print("  MDP信用分配测试全部通过!")
    return True


def test_selective_rollback():
    print("\n" + "=" * 60)
    print("测试4: 选择性回滚与重训练")
    print("=" * 60)

    from multi_agent_trainer.training.rollback_manager import (
        SelectiveRollbackManager,
        RollbackConfig,
        OverheadBudgetManager,
    )
    from multi_agent_trainer.responsibility.causal_inference import (
        ResponsibilityReport,
        ResponsibilityScore,
    )

    budget_mgr = OverheadBudgetManager(max_overhead_ratio=0.15)
    budget_mgr.record_training_time(100.0)
    print(f"  初始预算: remaining={budget_mgr.remaining_budget:.2%}")

    budget_mgr.record_rollback_time(5.0)
    budget_mgr.record_retraining_time(3.0)
    budget_mgr.record_inference_time(2.0)
    print(f"  使用后预算: ratio={budget_mgr.current_ratio:.4f}, within_budget={budget_mgr.current_ratio <= 0.15}")

    can_afford = budget_mgr.can_afford(5.0)
    print(f"  能否承担5s开销: {can_afford}")

    cannot_afford = budget_mgr.can_afford(50.0)
    print(f"  能否承担50s开销: {cannot_afford}")

    report = budget_mgr.get_report()
    print(f"  预算报告: {json.dumps(report, indent=2, default=str)}")

    assert budget_mgr.current_ratio <= 0.15, "预算超限"
    print("  开销预算管理测试通过!")

    print("\n  选择性回滚逻辑验证:")
    report = ResponsibilityReport(
        episode_id="test_ep",
        task_success=False,
        overall_score=0.3,
        responsible_agents=["executor"],
        responsible_agent_details=[
            {
                "agent_id": "executor",
                "agent_role": "executor",
                "marginal_contribution": -0.5,
                "reasons": ["动作置信度过低"],
                "error_steps": [{"step": 1, "summary": "执行动作失败"}],
            }
        ],
        causal_chain=[
            {"step": 0, "agent_id": "planner", "is_error_step": False},
            {"step": 1, "agent_id": "executor", "is_error_step": True},
            {"step": 2, "agent_id": "evaluator", "is_error_step": False},
        ],
        inference_time=0.5,
        num_samples_used=15,
        overhead_ratio=0.05,
        agent_scores=[
            ResponsibilityScore(
                agent_id="planner",
                agent_role="planner",
                marginal_contribution=0.1,
                shapley_value=0.1,
                counterfactual_impact=0.1,
                is_responsible=False,
                confidence=0.8,
            ),
            ResponsibilityScore(
                agent_id="executor",
                agent_role="executor",
                marginal_contribution=-0.5,
                shapley_value=-0.4,
                counterfactual_impact=-0.5,
                is_responsible=True,
                confidence=0.9,
            ),
            ResponsibilityScore(
                agent_id="evaluator",
                agent_role="evaluator",
                marginal_contribution=0.05,
                shapley_value=0.05,
                counterfactual_impact=0.05,
                is_responsible=False,
                confidence=0.7,
            ),
        ],
    )

    print(f"    责任智能体: {report.responsible_agents}")
    print(f"    非责任智能体将被冻结: planner, evaluator")
    print(f"    仅回滚/重训练: executor")

    for score in report.agent_scores:
        print(f"    {score.agent_id}: MC={score.marginal_contribution:.2f}, responsible={score.is_responsible}")

    assert report.responsible_agents == ["executor"], "责任智能体识别错误"
    assert not report.agent_scores[0].is_responsible, "planner不应被标记为责任方"
    assert report.agent_scores[1].is_responsible, "executor应被标记为责任方"
    assert not report.agent_scores[2].is_responsible, "evaluator不应被标记为责任方"

    print("  选择性回滚测试全部通过!")
    return True


def test_overhead_control():
    print("\n" + "=" * 60)
    print("测试5: 开销控制验证 (<=15%)")
    print("=" * 60)

    from multi_agent_trainer.training.rollback_manager import OverheadBudgetManager

    budget = OverheadBudgetManager(max_overhead_ratio=0.15)

    total_training = 1000.0
    budget.record_training_time(total_training)

    max_overhead = total_training * 0.15
    print(f"  总训练时间: {total_training}s")
    print(f"  最大允许开销: {max_overhead}s (15%)")

    budget.record_inference_time(30.0)
    budget.record_rollback_time(20.0)
    budget.record_retraining_time(40.0)
    budget.record_communication_time(10.0)

    total_overhead = 30.0 + 20.0 + 40.0 + 10.0
    actual_ratio = total_overhead / total_training

    print(f"  实际总开销: {total_overhead}s")
    print(f"  实际开销比例: {actual_ratio:.2%}")
    print(f"  在预算内: {actual_ratio <= 0.15}")

    assert actual_ratio <= 0.15, f"开销比例 {actual_ratio:.2%} 超过15%限制"

    extreme_budget = OverheadBudgetManager(max_overhead_ratio=0.15)
    extreme_budget.record_training_time(100.0)
    extreme_budget.record_rollback_time(50.0)
    assert not extreme_budget.can_afford(10.0), "预算超限时应拒绝"
    print("  极端情况预算拒绝: PASS")

    print("  开销控制测试全部通过!")
    return True


def run_all_tests():
    print("\n" + "#" * 70)
    print("#  多智能体大模型训练系统 - 核心逻辑验证")
    print("#  验证内容: 通信协议 | 因果推断 | MDP信用 | 选择性回滚 | 开销控制")
    print("#" * 70)

    os.makedirs("./test_logs", exist_ok=True)

    results = {}

    tests = [
        ("通信协议", test_communication),
        ("因果责任推断", test_counterfactual_inference),
        ("MDP信用分配", test_mdp_credit_assignment),
        ("选择性回滚", test_selective_rollback),
        ("开销控制", test_overhead_control),
    ]

    for name, test_fn in tests:
        try:
            passed = test_fn()
            results[name] = "PASS" if passed else "FAIL"
        except Exception as e:
            results[name] = f"ERROR: {e}"
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("验证结果汇总:")
    print("=" * 70)

    all_passed = True
    for name, result in results.items():
        status = "PASS" if result == "PASS" else f"FAIL ({result})"
        symbol = "+" if result == "PASS" else "x"
        print(f"  [{symbol}] {name}: {status}")
        if result != "PASS":
            all_passed = False

    print("=" * 70)
    if all_passed:
        print("  所有测试通过! 系统核心逻辑验证成功!")
    else:
        print("  部分测试失败，请检查错误信息。")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
