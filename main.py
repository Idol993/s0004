"""多智能体大模型训练系统 - 主入口脚本

使用方式:
    python main.py --config config.yaml --episodes 10 --device cpu
    python main.py --device cpu --episodes 5
"""

import argparse
import json
import sys
import os

from multi_agent_trainer import create_system
from multi_agent_trainer.training import TrainingOrchestrator, TrainingConfig
from multi_agent_trainer.agents.llm_backbone import LLMConfig
from multi_agent_trainer.responsibility import CounterfactualConfig, MDPCreditConfig
from multi_agent_trainer.training.rollback_manager import RollbackConfig


def parse_args():
    parser = argparse.ArgumentParser(description="多智能体大模型训练系统")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument("--episodes", type=int, default=10, help="训练轮数")
    parser.add_argument("--device", type=str, default="cpu", help="计算设备 (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--log_dir", type=str, default="./logs", help="日志目录")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="检查点目录")
    parser.add_argument("--model_name", type=str, default="gpt2", help="基础模型名称（默认gpt2，124MB轻量版）")
    parser.add_argument("--max_overhead", type=float, default=0.15, help="最大开销比例")
    return parser.parse_args()


SAMPLE_TASKS = [
    "设计一个文本分类系统，将新闻文章分为体育、科技、政治三类",
    "构建一个问答系统，能够回答关于历史事件的复杂问题",
    "开发一个代码生成工具，根据自然语言描述生成Python函数",
    "创建一个摘要生成器，将长文档压缩为关键要点",
    "实现一个对话系统，能够进行多轮上下文相关的对话",
]


def main():
    args = parse_args()

    print("=" * 70)
    print("  多智能体大模型训练系统 (Multi-Agent LLM Trainer)")
    print("  五智能体: 规划器 | 执行器 | 评估器 | 记忆器 | 反思器")
    print("  核心功能: 因果责任推断 | 选择性回滚 | MDP信用分配")
    print("=" * 70)
    print(f"\n配置:")
    print(f"  设备: {args.device}")
    print(f"  训练轮数: {args.episodes}")
    print(f"  基础模型: {args.model_name}")
    print(f"  最大开销比例: {args.max_overhead:.0%}")
    print(f"  随机种子: {args.seed}")
    print()

    training_config = TrainingConfig(
        max_episodes=args.episodes,
        device=args.device,
        seed=args.seed,
        log_dir=args.log_dir,
        checkpoint_base_dir=args.checkpoint_dir,
        max_overhead_ratio=args.max_overhead,
    )

    llm_configs = {}
    for role in ["planner", "executor", "evaluator", "memory", "reflector"]:
        llm_configs[role] = LLMConfig(
            model_name=args.model_name,
            device=args.device,
        )

    counterfactual_config = CounterfactualConfig(
        num_samples=3,
        shapley_enabled=True,
        shapley_num_permutations=10,
    )

    mdp_config = MDPCreditConfig(
        hidden_dim=128,
        num_layers=1,
    )

    rollback_config = RollbackConfig(
        max_overhead_ratio=args.max_overhead,
        retraining_steps=20,
    )

    print("正在初始化系统...")
    orchestrator = TrainingOrchestrator(
        config=training_config,
        llm_configs=llm_configs,
        counterfactual_config=counterfactual_config,
        mdp_config=mdp_config,
        rollback_config=rollback_config,
    )

    print(f"已注册智能体: {list(orchestrator.agents.keys())}")
    print()

    tasks = SAMPLE_TASKS[:args.episodes] if args.episodes <= len(SAMPLE_TASKS) else SAMPLE_TASKS

    def episode_callback(result):
        status = "SUCCESS" if result.success else "FAILED"
        print(f"  Episode {result.episode_id}: {status} (score={result.score:.4f})")
        print(f"    总耗时: {result.episode_time:.3f}s, 开销: {result.overhead_time:.3f}s "
              f"({result.overhead_time / max(result.episode_time, 1e-8):.1%})")
        if result.responsibility_report:
            report = result.responsibility_report
            print(f"    责任智能体: {report.responsible_agents}")
            for detail in report.responsible_agent_details:
                print(f"      - {detail['agent_id']} ({detail.get('agent_role', '?')}): "
                      f"MC={detail.get('marginal_contribution', 0):.4f}")
                for reason in detail.get('reasons', []):
                    print(f"          原因: {reason}")
                for err in detail.get('error_steps', [])[:2]:
                    print(f"          问题步骤 step{err.get('step','?')}: {err.get('summary','')[:80]}")
            print(f"    责任推断开销: {report.inference_time:.3f}s, 采样数: {report.num_samples_used}")
            if result.rollback_record:
                rb = result.rollback_record
                print(f"  --- 回滚与重训详情 ---")
                print(f"    失败前分数: {rb.pre_rollback_score:.4f}")
                print(f"    重训后分数: {rb.post_retraining_score:.4f}")
                print(f"    改善幅度: {rb.improvement:+.4f}")
                print(f"    回滚策略: {rb.rollback_strategy.value}, 重训策略: {rb.retraining_strategy.value}")
                print(f"    实际回滚的智能体: {rb.rolled_back_agents}")
                print(f"    实际重训的智能体: {rb.retrained_agents}")
                print(f"    被冻结的智能体 (非责任方): {rb.frozen_agents}")
                print(f"    回滚耗时: {rb.rollback_time:.3f}s, 重训耗时: {rb.retraining_time:.3f}s")
                if rb.processing_log:
                    print(f"    处理日志:")
                    for line in rb.processing_log:
                        print(f"      {line}")

    print("开始训练...")
    print("-" * 70)

    summary = orchestrator.train(
        tasks=tasks,
        max_episodes=args.episodes,
        callback=episode_callback,
    )

    print("-" * 70)
    print("\n训练完成！汇总统计:")
    print(f"  总轮数: {summary['total_episodes']}")
    print(f"  最终平均分: {summary['final_avg_score']:.4f}")
    print(f"  最佳分数: {summary['best_score']:.4f}")
    print(f"  总体成功率: {summary['overall_success_rate']:.2%}")
    print(f"  总耗时: {summary['total_time']:.2f}s")
    print(f"  纯训练时间: {summary.get('pure_training_time', 0):.2f}s")
    print(f"  总开销时间: {summary['total_overhead_time']:.2f}s")
    print(f"  通信时间: {summary.get('communication_time', 0):.4f}s (消息数={summary.get('num_messages', 0)})")
    print(f"  开销比例: {summary['overhead_ratio']:.2%} (预算上限 {summary.get('max_overhead_ratio', 0.15):.0%})")
    print(f"  开销在预算内: {'是 ✓' if summary['overhead_within_budget'] else '否 ✗'}")
    print(f"  回滚次数: {summary['num_rollbacks']}")
    if summary['num_rollbacks'] > 0:
        print(f"  回滚成功率: {summary['rollback_success_rate']:.2%}")

    budget = summary.get('budget_report', {})
    if budget:
        print(f"\n细粒度预算报告 (来自 OverheadBudgetManager):")
        print(f"  训练时间: {budget.get('total_training_time', 0):.3f}s")
        print(f"  通信时间: {budget.get('communication_time', budget.get('communication_time_from_bus', 0)):.3f}s")
        print(f"  推断时间: {budget.get('inference_time', 0):.3f}s")
        print(f"  回滚时间: {budget.get('rollback_time', 0):.3f}s")
        print(f"  重训练时间: {budget.get('retraining_time', 0):.3f}s")
        print(f"  当前开销比例: {budget.get('overhead_ratio', 0):.4%}")
        print(f"  剩余预算: {budget.get('remaining_budget', 0):.2%}")
        print(f"  预算是否充足: {'是' if budget.get('within_budget', True) else '否'}")

    resp_stats = summary.get('responsibility_stats', {})
    if resp_stats:
        print(f"\n责任推断统计:")
        print(f"  总推断时间: {resp_stats.get('total_inference_time', 0):.3f}s")
        print(f"  总训练时间参考: {resp_stats.get('total_training_time', 0):.3f}s")
        print(f"  推断开销比: {resp_stats.get('overhead_ratio', 0):.4%}")

    agent_stats = orchestrator.get_agent_statistics()
    print(f"\n智能体统计:")
    for aid, stats in agent_stats.items():
        print(f"  {aid}: 冻结={stats['is_frozen']}, 动作数={stats['num_actions']}, 累计奖励={stats['local_reward']:.4f}")

    results_path = os.path.join(args.log_dir, "training_summary.json")
    os.makedirs(args.log_dir, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n详细结果已保存至: {results_path}")

    orchestrator.stop()


if __name__ == "__main__":
    main()
