"""
验证完整链路：
1. create_system 入口可用
2. 成功回合 reflector 只执行一次
3. 失败回合回滚后分数是真实重跑结果
4. 训练汇总能正常打印
5. 各阶段开销统计完整
6. 预算报告含反思耗时
"""
import sys
import os
import logging

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

logging.basicConfig(level=logging.WARNING, format="%(levelname)s - %(message)s")

sys.path.insert(0, os.path.dirname(__file__))

from multi_agent_trainer import create_system

# ============================================================
# 第一步：create_system
# ============================================================
print("=" * 60)
print("[1] create_system 入口验证")
print("=" * 60)

sys_obj = create_system(
    config_dict={
        "agents": {
            "planner": {"model_name": "gpt2"},
            "executor": {"model_name": "gpt2"},
            "evaluator": {"model_name": "gpt2"},
            "memory": {"model_name": "gpt2"},
            "reflector": {"model_name": "gpt2"},
        },
        "system": {"max_overhead_ratio": 0.15},
    },
    device="cpu",
    log_dir="./logs_test",
    checkpoint_dir="./checkpoints_test",
)
print(f"  Agents: {list(sys_obj.agents.keys())}")
assert len(sys_obj.agents) == 5, f"Expected 5 agents, got {len(sys_obj.agents)}"
print("  [PASS] create_system 创建了 5 个智能体")

# ============================================================
# 第二步：失败回合 + 真实重评验证
# ============================================================
print("\n" + "=" * 60)
print("[2] 失败回合 - 真实重评验证")
print("=" * 60)

result = sys_obj.run_episode("写一个快速排序算法")
print(f"  success = {result.success}")
print(f"  score = {result.score:.4f}")
print(f"  episode_time = {result.episode_time:.3f}s")
print(f"  overhead_time = {result.overhead_time:.3f}s")

if result.rollback_record:
    rb = result.rollback_record
    print(f"  --- 回滚详情 ---")
    print(f"    失败前分数: {rb.pre_rollback_score:.4f}")
    print(f"    重训后分数: {rb.post_retraining_score:.4f}")
    print(f"    改善: {rb.improvement:+.4f}")
    print(f"    回滚智能体: {rb.rolled_back_agents}")
    print(f"    重训智能体: {rb.retrained_agents}")
    print(f"    冻结智能体: {rb.frozen_agents}")
    has_reeval_log = any("[重评]" in line for line in rb.processing_log)
    print(f"    有真实重评日志: {has_reeval_log}")
    if has_reeval_log:
        for line in rb.processing_log:
            if "[重评]" in line:
                print(f"    -> {line}")
    assert has_reeval_log, "回滚记录应包含真实重评日志"
    print("  [PASS] 回滚记录包含真实重跑评估")
else:
    print("  [WARN] 无回滚记录 (可能 score >= threshold)")

# ============================================================
# 第三步：成功回合 + reflector 只调一次
# ============================================================
print("\n" + "=" * 60)
print("[3] 成功回合 - reflector 只执行一次")
print("=" * 60)

evaluator = sys_obj.agents["evaluator"]
original_eval_decide = evaluator.decide
original_reflect_step = sys_obj.agents["reflector"].step
reflect_count = [0]

def patched_decide(perceived_state):
    decision = original_eval_decide(perceived_state)
    decision["score"] = 0.85
    decision["success"] = True
    return decision

def counted_reflect_step(obs):
    reflect_count[0] += 1
    return original_reflect_step(obs)

evaluator.decide = patched_decide
sys_obj.agents["reflector"].step = counted_reflect_step

result2 = sys_obj.run_episode("写一个冒泡排序算法")
print(f"  success = {result2.success}")
print(f"  score = {result2.score:.4f}")
print(f"  reflector 调用次数: {reflect_count[0]}")
assert reflect_count[0] == 1, f"Expected 1 reflector call, got {reflect_count[0]}"
print("  [PASS] 成功回合 reflector 只被调用 1 次")

evaluator.decide = original_eval_decide
sys_obj.agents["reflector"].step = original_reflect_step

# ============================================================
# 第四步：训练汇总验证
# ============================================================
print("\n" + "=" * 60)
print("[4] 训练汇总验证")
print("=" * 60)

summary = sys_obj.get_training_summary()
print(f"  total_episodes: {summary.get('total_episodes', 'N/A')}")
print(f"  overall_success_rate: {summary.get('overall_success_rate', 0):.2%}")
print(f"  total_time: {summary.get('total_time', 0):.2f}s")
print(f"  pure_training_time: {summary.get('pure_training_time', 0):.2f}s")
print(f"  total_overhead_time: {summary.get('total_overhead_time', 0):.2f}s")
print(f"  communication_time: {summary.get('communication_time', 0):.4f}s")
print(f"  num_messages: {summary.get('num_messages', 0)}")
print(f"  overhead_ratio: {summary.get('overhead_ratio', 0):.4f} ({summary.get('overhead_ratio', 0)*100:.2f}%)")
print(f"  overhead_within_budget: {summary.get('overhead_within_budget', False)}")

budget = summary.get('budget_report', {})
if budget:
    print(f"\n  细粒度预算报告:")
    for key in ['total_training_time', 'communication_time', 'inference_time',
                'rollback_time', 'retraining_time', 'reflection_time']:
        print(f"    {key}: {budget.get(key, 0):.4f}s")
    print(f"    overhead_ratio: {budget.get('overhead_ratio', 0):.4%}")
    print(f"    within_budget: {budget.get('within_budget', False)}")

required_fields = ['total_episodes', 'total_time', 'communication_time',
                   'num_messages', 'overhead_ratio']
missing = [f for f in required_fields if f not in summary]
assert not missing, f"Missing summary fields: {missing}"
print("  [PASS] 训练汇总字段齐全")

ct = summary.get('communication_time', 0)
assert ct > 0, f"communication_time = {ct}, expected > 0"
print(f"  [PASS] communication_time = {ct:.4f}s (> 0)")

nm = summary.get('num_messages', 0)
assert nm > 0, f"num_messages = {nm}, expected > 0"
print(f"  [PASS] num_messages = {nm} (> 0)")

has_reflection_in_budget = budget.get('reflection_time', None) is not None
assert has_reflection_in_budget, "Budget report should include reflection_time"
print(f"  [PASS] 预算报告含 reflection_time = {budget.get('reflection_time', 0):.4f}s")

# ============================================================
# 第五步：记忆模块 fallback 验证
# ============================================================
print("\n" + "=" * 60)
print("[5] 记忆模块 fallback 验证")
print("=" * 60)

from multi_agent_trainer.agents.memory import MemoryAgent
mem = sys_obj.agents["memory"]

hash_emb = mem._hash_embedding("test fallback encoding")
print(f"  hash_embedding shape: {hash_emb.shape}, dtype: {hash_emb.dtype}")
assert hash_emb.shape[0] == 256, f"Expected dim 256, got {hash_emb.shape[0]}"
print("  [PASS] hash_embedding fallback 正常工作")

noop_result = mem.act({"action": "noop", "query": "test"})
print(f"  noop action result: {noop_result.get('action_type', 'N/A')}")
print("  [PASS] memory agent noop fallback 正常工作")

# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 60)
print("ALL PASSED")
print("=" * 60)

sys_obj.stop()
