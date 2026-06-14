import sys
import traceback

try:
    print("Step 1: Importing create_system...", flush=True)
    from multi_agent_trainer import create_system
    from multi_agent_trainer.agents.llm_backbone import LLMConfig
    print("Step 1: OK", flush=True)

    print("Step 2: Creating system with gpt2 (lightweight model)...", flush=True)

    llm_configs = {
        role: LLMConfig(model_name="gpt2", device="cpu")
        for role in ["planner", "executor", "evaluator", "memory", "reflector"]
    }
    from multi_agent_trainer.training import TrainingConfig
    from multi_agent_trainer.responsibility import CounterfactualConfig, MDPCreditConfig
    from multi_agent_trainer.training.rollback_manager import RollbackConfig
    from multi_agent_trainer.training import TrainingOrchestrator

    training_config = TrainingConfig(
        max_episodes=1,
        device="cpu",
        seed=42,
        log_dir="./test_logs_run",
        checkpoint_base_dir="./test_ckpts_run",
    )

    cf_cfg = CounterfactualConfig(num_samples=2, shapley_enabled=True, shapley_num_permutations=5)
    mdp_cfg = MDPCreditConfig(hidden_dim=64, num_layers=1)
    rb_cfg = RollbackConfig(retraining_steps=5)

    sys_obj = TrainingOrchestrator(
        config=training_config,
        llm_configs=llm_configs,
        counterfactual_config=cf_cfg,
        mdp_config=mdp_cfg,
        rollback_config=rb_cfg,
    )
    print(f"Step 2: OK - Agents: {list(sys_obj.agents.keys())}", flush=True)

    print("Step 3: Running a failure-simulating episode...", flush=True)
    result = sys_obj.run_episode("写一个快速排序算法")
    print(f"Step 3: OK - success={result.success}, score={result.score:.4f}", flush=True)

    if result.responsibility_report:
        rr = result.responsibility_report
        print(f"  责任智能体: {rr.responsible_agents}", flush=True)
        for d in rr.responsible_agent_details:
            print(f"    -> {d['agent_id']}: reasons={d.get('reasons', [])}", flush=True)

    if result.rollback_record:
        rb = result.rollback_record
        print(f"  回滚记录:", flush=True)
        print(f"    失败前: {rb.pre_rollback_score:.4f}", flush=True)
        print(f"    重训后: {rb.post_retraining_score:.4f}", flush=True)
        print(f"    改善:   {rb.improvement:+.4f}", flush=True)
        print(f"    回滚:   {rb.rolled_back_agents}", flush=True)
        print(f"    重训:   {rb.retrained_agents}", flush=True)
        print(f"    冻结:   {rb.frozen_agents}", flush=True)
        for line in rb.processing_log:
            print(f"    {line}", flush=True)

    print(f"Step 4: Summary...", flush=True)
    summary = sys_obj._compile_training_summary([result])
    print(f"  总耗时: {summary.get('total_time', 0):.3f}s", flush=True)
    print(f"  纯训练: {summary.get('pure_training_time', 0):.3f}s", flush=True)
    print(f"  总开销: {summary.get('total_overhead_time', 0):.3f}s", flush=True)
    print(f"  通信时间: {summary.get('communication_time', 0):.4f}s", flush=True)
    print(f"  消息数: {summary.get('num_messages', 0)}", flush=True)
    print(f"  开销比例: {summary.get('overhead_ratio', 0):.2%}", flush=True)
    print(f"  预算内: {summary.get('overhead_within_budget', False)}", flush=True)

    print("\nAll steps passed!", flush=True)
    sys_obj.stop()

except Exception as e:
    print(f"\nERROR: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)
