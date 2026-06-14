import sys
import traceback

try:
    print("Step 1: Importing create_system...", flush=True)
    from multi_agent_trainer import create_system
    print("Step 1: OK", flush=True)

    print("Step 2: Creating system...", flush=True)
    sys_obj = create_system(
        device="cpu",
        log_dir="./test_logs3",
        checkpoint_dir="./test_ckpts3",
    )
    print(f"Step 2: OK - Agents: {list(sys_obj.agents.keys())}", flush=True)

    print("Step 3: Running episode...", flush=True)
    result = sys_obj.run_episode("测试任务：写一个问候函数")
    print(f"Step 3: OK - success={result.success}, score={result.score}", flush=True)

    print("All done!", flush=True)

except Exception as e:
    print(f"ERROR: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)
