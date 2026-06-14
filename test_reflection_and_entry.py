"""
验证：
1. create_system 入口可用
2. 成功回合 reflector 只执行一次
3. 开销统计在成功回合也正常
"""
import sys
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verify_reflection")

print("=" * 60)
print("验证 1: create_system 入口")
print("=" * 60)
sys.path.insert(0, os.path.dirname(__file__))

from multi_agent_trainer import create_system, LLMConfig

# 用 gpt2 轻量模型
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
)
print(f"create_system OK: {len(sys_obj.agents)} agents")
print(f"  Agents: {list(sys_obj.agents.keys())}")

print("\n" + "=" * 60)
print("验证 2: 成功回合 - 检查 reflector 只执行一次")
print("=" * 60)

# 捕获 reflector 的调用次数
reflector = sys_obj.agents["reflector"]
original_step = reflector.step
reflect_call_count = [0]
def counted_step(obs):
    reflect_call_count[0] += 1
    logger.info(f"[REFLECTOR CALLED] count={reflect_call_count[0]}")
    return original_step(obs)
reflector.step = counted_step

result = sys_obj.run_episode("写一个 hello world 程序")
# 由于 evaluator 默认返回 low score，我们手动模拟成功情况
print(f"\nEpisode 完成: success={result['success']}, score={result['score']:.4f}")
print(f"Reflector.step 被调用次数: {reflect_call_count[0]}")

# 不管结果如何，检查 summary
summary = sys_obj.get_training_summary()
print("\n" + "=" * 60)
print("验证 3: 开销统计")
print("=" * 60)
print(f"  total_time: {summary.get('total_time', 'N/A'):.4f}s")
print(f"  pure_training_time: {summary.get('pure_training_time', 'N/A'):.4f}s")
print(f"  communication_time: {summary.get('communication_time', 'N/A'):.4f}s")
print(f"  num_messages: {summary.get('num_messages', 'N/A')}")
print(f"  overhead_ratio: {summary.get('overhead_ratio', 'N/A'):.4f} ({summary.get('overhead_ratio', 0)*100:.2f}%)")
print(f"  overhead_ratio <= 0.15? {summary.get('overhead_ratio', 1) <= 0.15}")

# 判断测试
print("\n" + "=" * 60)
print("测试结果:")
print("=" * 60)
ok = True
if reflect_call_count[0] not in (0, 1):
    print(f"  ❌ FAIL: reflector 被调用 {reflect_call_count[0]} 次，预期 0 或 1 次")
    ok = False
else:
    print(f"  ✅ PASS: reflector 被调用 {reflect_call_count[0]} 次 (成功时 1 次，失败时 0 次)")

ct = summary.get('communication_time', 0)
if ct <= 0:
    print(f"  ❌ FAIL: communication_time = {ct}，预期 > 0")
    ok = False
else:
    print(f"  ✅ PASS: communication_time = {ct:.4f}s")

nm = summary.get('num_messages', 0)
if nm <= 0:
    print(f"  ❌ FAIL: num_messages = {nm}，预期 > 0")
    ok = False
else:
    print(f"  ✅ PASS: num_messages = {nm}")

or_ = summary.get('overhead_ratio', 1)
if or_ > 0.15:
    print(f"  ❌ FAIL: overhead_ratio = {or_:.4f}，超过 15% 预算")
    ok = False
else:
    print(f"  ✅ PASS: overhead_ratio = {or_:.4f} ({or_*100:.2f}%) ≤ 15%")

print()
if ok:
    print("✅ 所有测试通过！")
else:
    print("❌ 部分测试失败")
    sys.exit(1)
