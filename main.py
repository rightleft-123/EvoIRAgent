import json
import time
from pathlib import Path
from EXPAgent import EXPAgent

if __name__ == "__main__":
    evolution_instructions_file_path = "/data/zkl/AgenticIR/exploration_self_evolve/instructions_list/extra_tools_instruction.json"
    with open(evolution_instructions_file_path, "r") as f:
        evolution_instructions = json.load(f)

    exp_agent = EXPAgent(
        with_retrieval=False,
        with_reflection=False,
        with_rollback=False,
        silent=False
    )
    time.sleep(2)

    iter_times = 3  # 迭代次数
    # 运行多次迭代来模拟学习过程
    for i in range(iter_times):
        print(f"\n######## Starting Iteration {i+1}/{iter_times} ########")
        for _, evolution_instruction in enumerate(evolution_instructions):
            input_img_path = evolution_instruction.get("lq_image_path", "")
            exp_agent._prepare_dir(input_dir=input_img_path, output_dir=Path("/data/zkl/AgenticIR/exploration_self_evolve/stage1/output"))
            exp_agent._init_state()
            exp_agent.run_stage1(evolution_instruction)
        print(f"######## Finished Iteration {i+1}/{iter_times} ########\n")
