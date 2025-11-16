
import sys
sys.path.append("/data/zkl/AgenticIR")
import shutil
from utils.logger import get_logger
from utils.img_tree import ImgTree
from utils.misc import sorted_glob
import random
import torch
import pyiqa
import tqdm
import json
import logging
import time
from time import localtime, strftime

from pathlib import Path
from typing import Optional

from llm.depictqa import DepictQA
from llm.openrouter import OpenRouter

from vector_db import RouterExperiencePool, FundamentalExpPoolExperience

from exploration_self_evolve.instructiion_prompts.stage1_prompts import Stage1Prompts
from exploration_self_evolve.utils_img import imread, img2tensor, crop_to_smallest
from executor import executor, Tool

torch.hub.set_dir("/data/zkl/.cache/torch/hub")

class EXPAgent:
    def __init__(
        self,
        output_dir: Path = Path("/data/zkl/AgenticIR/exploration_self_evolve/stage1/output"),
        instruction_file_path: Path = Path("/data/zkl/AgenticIR/exploration_self_evolve/instructions_list/extra_tools_instruction.json"),
        llm_config_path: Path = Path("/data/zkl/AgenticIR/config.yml"),
        evaluate_degradation_by: str = "depictqa",
        with_retrieval: bool = True,
        schedule_experience_path: Optional[Path] = Path(
            "/data/zkl/AgenticIR/memory/schedule_experience.json"
        ),
        with_reflection: bool = True,
        reflect_by: str = "depictqa",
        with_rollback: bool = True,
        silent: bool = False,
    ) -> None:
        # path
        self._prepare_dir(input_dir="/data/zkl/AgenticIR/dataset/example.png", output_dir=output_dir)
        # state
        self._init_state()
        # config
        self._config(
            evaluate_degradation_by,
            with_retrieval,
            with_reflection,
            reflect_by,
            with_rollback
        )
        # components
        self._create_components(llm_config_path, schedule_experience_path, silent)
        # constants
        self._set_constants()

    def _init_state(self) -> None:
        self.plan: list[str] = []
        self.tool_execution_times: dict[str, float] = {}  # 存储每个工具的执行时间
        self.work_mem: dict = {
            "plan": {"initial": [], "adjusted": [
                # {
                #     "failed": [...] + [...],
                #     "new": [...] + [...]
                # }
            ]},
            "execution_path": {"subtasks": [], "tools": []},
            "n_invocations": 0,
            "tree": {
                "img_path": str(self.img_tree_dir / "0-img" / "input.png"),
                "best_descendant": None,
                "children": {
                    # `subtask1`: {
                    #     "best_tool": ...,
                    #     "tools": {
                    #         `tool1`: {
                    #             "degradation": ...,
                    #             "severity": ...,
                    #             "img_path": ...,
                    #             "best_descendant": ...,
                    #             "children": {...}
                    #         },
                    #         ...
                    #     }
                    # }
                },
            },
        }
        self.cur_node = self.work_mem["tree"]

    def _config(
        self,
        evaluate_degradation_by: str,
        with_retrieval: bool,
        with_reflection: bool,
        reflect_by: str,
        with_rollback: bool
    ) -> None:
        assert evaluate_degradation_by in {"gpt4v", "depictqa"}
        self.evaluate_degradation_by = evaluate_degradation_by
        self.with_retrieval = with_retrieval
        assert reflect_by in {"gpt4v", "depictqa"}
        self.with_reflection = with_reflection
        self.reflect_by = reflect_by
        self.with_rollback = with_rollback

    def _create_components(
        self,
        llm_config_path: Path,
        schedule_experience_path: Optional[Path],
        silent: bool,
    ) -> None:
        # logger
        self.qa_logger = get_logger(
            logger_name="EXPAgent QA",
            log_file=self.qa_path,
            console_log_level=logging.WARNING,
            file_format_str="%(message)s",
            silent=silent,
        )
        workflow_format_str = "%(asctime)s - %(levelname)s\n%(message)s\n"
        self.workflow_logger: logging.Logger = get_logger(
            logger_name="EXPAgent Workflow",
            log_file=self.workflow_path,
            console_format_str=workflow_format_str,
            file_format_str=workflow_format_str,
            silent=silent,
        )

        self.fundamental_experience_pool = RouterExperiencePool(capacity=25)
        # self.fundamental_experience_pool.read_experiences_from_file("/data/zkl/AgenticIR/exploration_self_evolve/output/experiences.json")
        
        # LLM
        # self.llm.query = GPT4(
        #     config_path=llm_config_path,
        #     logger=self.qa_logger,
        #     silent=silent,
        #     system_message=prompts.system_message,
        # )
        self.stage1_prompts = Stage1Prompts()

        self.llm = OpenRouter(
            config_path=llm_config_path,
            logger=self.qa_logger,
            silent=silent,
            system_message=self.stage1_prompts.get_system_message(),
            model="deepseek/deepseek-chat-v3-0324:free"
        )

        self.mllm = OpenRouter(
            config_path=llm_config_path,
            logger=self.qa_logger,
            silent=silent,
            system_message=self.stage1_prompts.get_system_message(),
            model="mistral-7b-instruct-v0.1"
        )

        self.depictqa = None
        if self.evaluate_degradation_by == "depictqa" or self.reflect_by == "depictqa":
            self.depictqa = DepictQA(logger=self.qa_logger, silent=silent)

        # experience
        if self.with_retrieval:
            assert (
                schedule_experience_path is not None
            ), "Experience should be provided."
            with open(schedule_experience_path, "r") as f:
                self.schedule_experience: str = json.load(f)["distilled"]

        # executor
        self.executor = executor
        random.seed(0)

        # IQA metrics
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device
        # No-reference metrics
        self.metric_dict = {metric: pyiqa.create_metric(metric).to(device) for metric in ['clipiqa', 'musiq', 'niqe', 'maniqa', 'brisque', 'nima']}
    
        # Full-reference metrics (初始化为空,有参考图时才创建)
        self.metric_paired_dict = {}

    def _set_constants(self) -> None:
        self.degra_subtask_dict: dict[str, str] = {
            "low resolution": "super-resolution",
            "noise": "denoising",
            "motion blur": "motion deblurring",
            "defocus blur": "defocus deblurring",
            "haze": "dehazing",
            "rain": "deraining",
            "dark": "brightening",
            "jpeg compression artifact": "jpeg compression artifact removal",
        }
        self.subtask_degra_dict: dict[str, str] = {
            v: k for k, v in self.degra_subtask_dict.items()
        }
        self.degradations = set(self.degra_subtask_dict.keys())
        self.subtasks = set(self.degra_subtask_dict.values())
        self.levels: list[str] = ["very low", "low", "medium", "high", "very high"]

    def _prepare_dir(self, input_dir: Path, output_dir: Path) -> None:
        """Sets attributes: `work_dir, img_tree_dir, log_dir, qa_path, workflow_path, summary_path`. Creates necessary directories, which will be like
        ```
        output_dir
        └── {task_id}(work_dir)
            ├── img_tree
            │   └── 0-img
            │       └── input.png
            └── logs
                ├── summary.json
                ├── workflow.log
                ├── llm_qa.md
                └── img_tree.html
        ```
        """

        task_id = f"{strftime('%y%m%d_%H%M%S', localtime())}"
        self.work_dir = output_dir / task_id
        self.work_dir.mkdir(parents=True)

        self.img_tree_dir = self.work_dir / "img_tree"
        self.img_tree_dir.mkdir()

        self.log_dir = self.work_dir / "logs"
        self.log_dir.mkdir()
        self.qa_path = self.log_dir / "llm_qa.md"
        self.workflow_path = self.log_dir / "workflow.log"
        self.work_mem_path = self.log_dir / "summary.json"

        rqd_input_dir = self.img_tree_dir / "0-img"
        rqd_input_dir.mkdir()
        rqd_input_path = rqd_input_dir / "input.png"
        self.root_input_path = rqd_input_path
        shutil.copy(input_dir, rqd_input_path)

        self._render_img_tree()

    def task_evolution(self, instruction, input_image, exp_pool1, exp_pool2):
        """
        任务演进阶段。
        判断需要使用到哪些类别的工具进行处理 (what to use)。

        Args:
            instruction (str): 用户指令.
            input_image (str): 输入图片路径.
            exp_pool1 (dict): Ps 工具池的经验.
            exp_pool2 (dict): 外部工具池的经验.

        Returns:
            list: 需要使用的工具类别列表, e.g., ['dehazing', 'jpeg_artifact_removal']
        """
        self.workflow_logger.info("--- 2. Task Evolution Stage ---")
        # 在这里构建 Prompt 并调用 LLM
        # prompt = f"""..."""
        # llm_response = self.call_llm(prompt)

        # 模拟返回
        mock_tool_categories = ['dehazing', 'jpeg_artifact_removal']
        
        return mock_tool_categories

    def solution_evolution(self, instruction):
        self.workflow_logger.info("---- Solution Evolution Stage ----")
        # 读取 instruction 中的信息并构建用于检索的 prompt
        retrieval_prompt = self.stage1_prompts.get_experience_retrieval_prompt(instruction=instruction)
        # 从经验池中检索局部经验：至多3条成功经验 + 1条失败经验
        retrieval_experiences = self.fundamental_experience_pool.get_relevant_experiences(task=retrieval_prompt, 
                                                                             instruction=instruction,
                                                                             top_k=3,
                                                                             threshold=0.6,
                                                                             success_only=False,
                                                                             return_str=True)
        # 从经验池中获取统计信息：工具的使用次数 + 工具的综合排名
        retrieval_tools_nums, retrieval_tools_basic_info, retrieval_tools_index_mapping = self.executor.get_tools_info(instruction=instruction, mapping_dict=self.degra_subtask_dict)
        retrieval_tools_execution_info = self.executor.get_tools_statistics(instruction=instruction, mapping_dict=self.degra_subtask_dict)
        self.tools_nums, self.tools_index_mapping = retrieval_tools_nums, retrieval_tools_index_mapping

        # 构建 Prompt 并调用 LLM 进行方案演进
        solution_evolution_prompt, json_schema = self.stage1_prompts.get_format_solution_evolution_prompt(
            instruction=instruction,
            tools_basic_info=retrieval_tools_basic_info,
            tools_execution_info=retrieval_tools_execution_info,
            experience_info=retrieval_experiences,
            num_plans=retrieval_tools_nums
        )

        self.workflow_logger.info("Start querying Agent for solution evolution...")
        # _, llm_response = self.llm.query(
        #     prompt=solution_evolution_prompt,
        #     format_check="JSON"
        # )
        llm_response = '{"plan1": {"tool": "xrestormer", "metrics": {"Full-Ref Rank": 2, "Non-Ref Rank": 2, "VLM-Ref Rank": 1, "Aesthetic Rank": 2}, "rationale": "XRestormer handles complex haze scenes well and ranks first in VLM-Ref due to superior edge detail recovery. It is slightly behind Dehazeformer in Full-Ref, Non-Ref, and Aesthetic Scores."}, "plan2": {"tool": "ridcp", "metrics": {"Full-Ref Rank": 4, "Non-Ref Rank": 4, "VLM-Ref Rank": 4, "Aesthetic Rank": 4}, "rationale": "RIDCP is reliable for medium haze regions but ranks lowest across all metrics compared to the other tools, due to less advanced haze removal and contrast enhancement."}, "plan3": {"tool": "dehazeformer", "metrics": {"Full-Ref Rank": 1, "Non-Ref Rank": 1, "VLM-Ref Rank": 2, "Aesthetic Rank": 1}, "rationale": "Dehazeformer is a state-of-the-art dehazing tool. It is ranked highest in Full-Ref, Non-Ref, and Aesthetic Scores due to its excellent haze removal and contrast restoration, while slightly behind XRestormer in VLM-Ref because of minor edge artifacts."}, "plan4": {"tool": "maxim", "metrics": {"Full-Ref Rank": 3, "Non-Ref Rank": 3, "VLM-Ref Rank": 3, "Aesthetic Rank": 3}, "rationale": "Maxim is effective for medium to high haze levels, providing moderate improvements across all metrics. Its performance is solid but lower than Dehazeformer and XRestormer."}}'
        self.workflow_logger.info("Finished querying Agent for solution evolution...")

        solutions = json.loads(llm_response)
        self.plan = [{f"{self.degra_subtask_dict[instruction['degradation_info'][0].get('degradation_global')]}":
            [solutions[plan]["tool"] for _, plan in enumerate(solutions, start=1)]
        }]
        self.plan_dict = solutions

        for i in range(len(solutions)):
            self.workflow_logger.info(f"Plan {i+1}: Tool = {solutions[f'plan{i+1}']['tool']}, Predict FR ranking = {solutions[f'plan{i+1}']['metrics']['Full-Ref Rank']}, Predict NR ranking = {solutions[f'plan{i+1}']['metrics']['Non-Ref Rank']}, Predict VLM ranking = {solutions[f'plan{i+1}']['metrics']['VLM-Ref Rank']}, Predict Aesthetic ranking = {solutions[f'plan{i+1}']['metrics']['Aesthetic Rank']}, Rationale = {solutions[f'plan{i+1}']['rationale']}")
        self.workflow_logger.info('---- Finished Solution Evolving Stage----')

    def execute_subtask(self, cache: Optional[Path]) -> bool:
        """Invokes tools to try to execute the top subtask in `self.plan` on `self.cur_node["img_path"]`, the directory of which is "0-img". Returns success or not. Updates `self.plan` and `self.cur_node`. Generates a directory parallel to "0-img", containing multiple directories, each of which contains outputs of a tool.\n
        Before:
        ```
        .
        ├── 0-img
        │   └── {input_path}
        └── ...
        ```
        After:
        ```
        .
        ├── 0-img
        │   └── {input_path}
        ├── {subtask_dir}
        |   ├── {tool_dir} 1
        |   │   └── 0-img
        |   │       └── output.png
        |   ├── ...
        |   └── {tool_dir} n
        |       └── 0-img
        |           └── output.png
        └── ...
        ```
        """
        self.workflow_logger.info('---- Start Execution Stage-----')
        while self.plan:
            plan_info = self.plan.pop(0)
            subtask = list(plan_info.keys())[0]
            tools_info = plan_info[subtask]

            subtask_dir, degradation, toolbox = self._prepare_for_subtask(subtask)
            res_degra_level_dict: dict[str, list[Path]] = {}
            success = True

            for i in range(len(tools_info)):
                tool_info = tools_info[i]
                for tool in toolbox:
                    if tool.tool_name == tool_info:
                        break
                
                self.workflow_logger.info(f"Plan {i+1}: Invoking tool {tool.tool_name} for subtask {subtask}...")

                self.work_mem["n_invocations"] += 1
                # prepare directory
                tool_dir = subtask_dir / f"tool-{tool.tool_name}"
                output_dir = tool_dir / "0-img"
                output_dir.mkdir(parents=True)

                # invoke tool with timing
                start_time = time.time()
                if cache is None:
                    tool(
                        input_dir=Path(self.cur_node["img_path"]).parent,
                        output_dir=output_dir,
                        silent=True,
                    )
                else:
                    dst_path = output_dir / "output.png"
                    rel_path = dst_path.relative_to(self.img_tree_dir)
                    src_path = cache / rel_path
                    dst_path.symlink_to(src_path)
                end_time = time.time()
                execution_time = end_time - start_time
                
                # 存储执行时间
                self.tool_execution_times[tool.tool_name] = execution_time
                self.workflow_logger.info(f"Tool {tool.tool_name} execution time: {execution_time:.2f} seconds")
                
                output_path = sorted_glob(output_dir)[0]

                if self.with_reflection:
                    degra_level = self.evaluate_tool_result(output_path, degradation)
                    self._record_tool_res(output_path, degra_level)
                    res_degra_level_dict.setdefault(degra_level, []).append(output_path)
                    if degra_level == "very low":
                        res_degra_level = "very low"
                        best_tool_name = tool.tool_name
                        # best_img_path = output_path
                        break
                else:
                    best_tool_name = tool.tool_name
                    # best_img_path = output_path
                    res_degra_level = "none"
                    self._record_tool_res(output_path, "none")
                    # self.executor.record_executed_subtask(subtask=subtask, tool_name=tool.tool_name)

            else:  # no result with "very low" degradation level
                for res_level in self.levels[1:]:
                    if res_level in res_degra_level_dict:
                        candidates = res_degra_level_dict[res_level]
                        self.workflow_logger.info("Searching for the best tool...")
                        best_img_path = self.search_best_by_comp(candidates)
                        best_tool_name = self._get_name_stem(best_img_path.parents[1].name)
                        if res_level != "low":  # fail
                            success = False
                        res_degra_level = res_level
                        break

            self.cur_node["children"][subtask]["best_tool"] = best_tool_name
            self.cur_node = self.cur_node["children"][subtask]["tools"][best_tool_name]
            if self.with_rollback and not success:
                self.cur_node["best_descendant"] = str(best_img_path)
                done_subtasks, _ = self._get_execution_path(Path(self.cur_node['img_path']))
                self.work_mem["plan"]["adjusted"].append({
                    "failed": f"{done_subtasks} + {self.plan}", "new": None
                })

            self._dump_summary()
            self._render_img_tree()
            self.workflow_logger.info(
                f"{subtask.capitalize()} result: "
                f"{self._img_nickname(self.cur_node['img_path'])} "
                f"with {res_degra_level} severity.")
                
            return success

    def _get_execution_path(self, img_path: Path) -> tuple[list[str], list[str]]:
        """Returns the execution path of the restored image (list of subtask and tools)."""
        exe_path = self._img_tree.get_execution_path(img_path)
        if not exe_path:
            return [], []
        subtasks, tools = zip(*exe_path)
        return list(subtasks), list(tools)
    
    def _img_nickname(self, img_path: str | Path) -> str:
        """Image name to display in log, showing the execution path."""        
        if isinstance(img_path, str):
            img_path = Path(img_path)
        subtasks, tools = self._get_execution_path(img_path)
        if not subtasks:
            return "input"
        return "-".join([f"{subtask}@{tool}" 
                         for subtask, tool in zip(subtasks, tools)])

    def _prepare_for_subtask(
            self, subtask: str
        ) -> tuple[Path, str, list[Tool]]:
            self.workflow_logger.info(
                f"Executing {subtask} on {self._img_nickname(self.cur_node['img_path'])}..."
            )

            subtask_dir = Path(self.cur_node["img_path"]).parents[1] / f"subtask-{subtask}"
            subtask_dir.mkdir()

            degradation = self.subtask_degra_dict[subtask]
            toolbox = self.executor.toolbox_router[subtask]
            random.shuffle(toolbox)

            return subtask_dir, degradation, toolbox
    
    def _record_tool_res(self, img_path: Path, degra_level: str) -> None:
        tool_name = self._get_name_stem(img_path.parents[1].name)
        subtask = self._get_name_stem(img_path.parents[2].name)
        degradation = self.subtask_degra_dict[subtask]

        # log
        self.workflow_logger.info(
            f"Severity of {degradation} of {self._img_nickname(img_path)} "
            f"is {degra_level}."
        )

        # update working memory
        cur_children = self.cur_node["children"]
        if subtask not in cur_children:
            cur_children[subtask] = {"best_tool": None, "tools": {}}
        assert tool_name not in cur_children[subtask]["tools"]
        cur_children[subtask]["tools"][tool_name] = {
            "degradation": degradation,
            "severity": degra_level,
            "img_path": str(img_path),
            "best_descendant": None,
            "children": {},
        }

    def evaluate_solutions(self, instruction):
        """
        评估每个解决方案的效果,并返回奖励。
        对每张图片都进行 score-iqa 和 vlm-iqa 评估。
        VLM-IQA 阶段将每个工具的输出与其他工具的输出进行两两比较。

        Args:
            instruction (dict): 包含指令信息,包括 gt_image_path 等.

        Returns:
            list: 包含每个工具评估结果的列表.
        """
        self.workflow_logger.info("---- Start Feedback Stage ----")
        ground_truth_image = instruction.get('gt_image_path', "")

        plan = [{f"{self.degra_subtask_dict[instruction['degradation_info'][0].get('degradation_global')]}":
            [self.plan_dict[plan]["tool"] for _, plan in enumerate(self.plan_dict, start=1)]
        }]

        while plan:
            plan_info = plan.pop(0)
            subtask = list(plan_info.keys())[0]
            tools_info = plan_info[subtask]

            evaluation_results = []
            processed_images = {}  # 存储所有工具的输出图像路径
            # 第一轮: Score-IQA 评估 和 降质程度评估
            for i, tool_name in enumerate(tools_info):
                processed_image_path = (
                self.img_tree_dir  # 直接使用类属性
                / f"subtask-{subtask}" 
                / f"tool-{tool_name}" 
                / "0-img" 
                / "output.png"
                )
                
                processed_images[tool_name] = processed_image_path
                
                self.workflow_logger.info(
                f"Evaluating solution {i+1} with tool {tool_name}..."
                )
                
                # 1. Score-IQA 评估
                score_iqa_result = self.evaluate_by_iqa(
                in_path=processed_image_path, 
                ref_path=ground_truth_image if ground_truth_image else None, 
                ntest=1
                )
                
                # 区分有参考和无参考指标
                non_ref_metrics = {}
                full_ref_metrics = {}
                aesthetic_ref_metrics = {}
                
                for metric_name, score in score_iqa_result.items():
                    if metric_name in ['psnr', 'lpips', 'dists', 'ssim']:
                        full_ref_metrics[metric_name] = score
                    elif metric_name in ['nima']:
                        aesthetic_ref_metrics[metric_name] = score
                    else:
                        non_ref_metrics[metric_name] = score

                # 暂存结果
                tool_evaluation = {
                "tool_name": tool_name,
                "processed_image_path": str(processed_image_path),
                "execution_time": self.tool_execution_times.get(tool_name, 0.0),  # 添加执行时间
                "score_iqa": {
                    "non_reference": non_ref_metrics,
                    "full_reference": full_ref_metrics if ground_truth_image else {},
                    "aesthetic_reference": aesthetic_ref_metrics if ground_truth_image else {},
                },
                "vlm_iqa": {}
                }
                
                evaluation_results.append(tool_evaluation)

            # 第二轮: 两两比较质量 (VLM-IQA + Score-IQA)
            if len(processed_images) > 1:
                self.workflow_logger.info("=== Starting pairwise quality comparison ===")
                
                tool_names = list(processed_images.keys())
                for i, result in enumerate(evaluation_results):
                    current_tool = result["tool_name"]
                    current_img = processed_images[current_tool]
                    
                    # VLM-IQA 比较
                    vlm_comparisons = []
                    # Score-IQA 比较 (无参考)
                    non_ref_comparisons = []
                    aesthetic_ref_comparisons = []
                    # Score-IQA 比较 (有参考)
                    full_ref_comparisons = []
                    
                    for other_tool in tool_names:
                        if other_tool == current_tool:
                            continue
                        
                        other_img = processed_images[other_tool]
                        other_result = next(r for r in evaluation_results if r["tool_name"] == other_tool)
                        
                        # 1. VLM-IQA 比较
                        if self.depictqa is not None:
                            quality_prompt, quality_comparison = self.depictqa.query(
                                img_path_lst=[current_img, other_img],
                                task="comp_quality",
                                degradation=None
                            )
                            
                            vlm_comparison = {
                                "compared_with": other_tool,
                                "result": quality_comparison  # "former" 或 "latter"
                            }
                            vlm_comparisons.append(vlm_comparison)
                            
                            self.workflow_logger.info(
                                f"VLM quality comparison: {current_tool} vs {other_tool} -> "
                                f"{quality_comparison} is better"
                            )
                        
                        # 2. Score-IQA 无参考指标比较
                        current_non_ref = result["score_iqa"]["non_reference"]
                        other_non_ref = other_result["score_iqa"]["non_reference"]
                        
                        non_ref_win_count = 0
                        non_ref_total = 0
                        for metric_name in current_non_ref.keys():
                            current_score = current_non_ref[metric_name]
                            other_score = other_non_ref[metric_name]
                            # 对于无参考指标,分数越高越好
                            if metric_name in ['clipiqa', 'musiq', 'maniqa']:
                                if current_score > other_score:
                                    non_ref_win_count += 1
                            elif metric_name == 'niqe':
                                if current_score < other_score:
                                    non_ref_win_count += 1
                            non_ref_total += 1
                        
                        non_ref_comparison = {
                        "compared_with": other_tool,
                        "win_count": non_ref_win_count,
                        "total_metrics": non_ref_total,
                        "win_rate": non_ref_win_count / non_ref_total if non_ref_total > 0 else 0
                        }
                        non_ref_comparisons.append(non_ref_comparison)

                        self.workflow_logger.info(
                        f"Score-IQA (non-ref) comparison: {current_tool} vs {other_tool} -> "
                        f"{non_ref_win_count}/{non_ref_total} metrics better"
                        )

                        cur_aesthetic_ref = result["score_iqa"]["aesthetic_reference"]
                        other_aesthetic_ref = other_result["score_iqa"]["aesthetic_reference"]
                        aesthetic_win_count = 0
                        aesthetic_total = 0
                        for metric_name in cur_aesthetic_ref.keys():
                            current_score = cur_aesthetic_ref[metric_name]
                            other_score = other_aesthetic_ref[metric_name]
                            # NIMA 分数越高越好
                            if metric_name in ['nima']:
                                if current_score > other_score:
                                    aesthetic_win_count += 1
                            aesthetic_total += 1

                        aesthetic_ref_comparison = {
                            "compared_with": other_tool,
                            "win_count": aesthetic_win_count,
                            "total_metrics": aesthetic_total,
                            "win_rate": aesthetic_win_count / aesthetic_total if aesthetic_total > 0 else 0
                        }
                        aesthetic_ref_comparisons.append(aesthetic_ref_comparison)

                        self.workflow_logger.info(
                            f"Score-IQA (aesthetic-ref) comparison: {current_tool} vs {other_tool} -> "
                            f"{aesthetic_win_count}/{aesthetic_total} metrics better"
                        )

                        # 3. Score-IQA 有参考指标比较
                        if ground_truth_image and result["score_iqa"]["full_reference"]:
                            current_full_ref = result["score_iqa"]["full_reference"]
                            other_full_ref = other_result["score_iqa"]["full_reference"]
                            
                            full_ref_win_count = 0
                            full_ref_total = 0
                            for metric_name in current_full_ref.keys():
                                current_score = current_full_ref[metric_name]
                                other_score = other_full_ref[metric_name]
                                # PSNR/SSIM 越高越好, LPIPS/DISTS 越低越好
                                if metric_name in ['psnr', 'ssim']:
                                    if current_score > other_score:
                                        full_ref_win_count += 1
                                elif metric_name in ['lpips', 'dists']:
                                    if current_score < other_score:
                                        full_ref_win_count += 1
                                full_ref_total += 1
                        
                        full_ref_comparison = {
                            "compared_with": other_tool,
                            "win_count": full_ref_win_count,
                            "total_metrics": full_ref_total,
                            "win_rate": full_ref_win_count / full_ref_total if full_ref_total > 0 else 0
                        }
                        full_ref_comparisons.append(full_ref_comparison)

                        
                        self.workflow_logger.info(
                            f"Score-IQA (full-ref) comparison: {current_tool} vs {other_tool} -> "
                            f"{full_ref_win_count}/{full_ref_total} metrics better"
                        )
                
                    # 将所有比较结果添加到该工具的评估结果中
                    result["vlm_iqa"]["quality_comparisons"] = vlm_comparisons
                    result["vlm_iqa"]["win_rate"] = sum(1 for c in vlm_comparisons if c["result"] == "former") / len(vlm_comparisons) if vlm_comparisons else 0
                    
                    result["score_iqa"]["non_ref_comparisons"] = non_ref_comparisons
                    result["score_iqa"]["non_ref_overall_win_rate"] = sum(c["win_rate"] for c in non_ref_comparisons) / len(non_ref_comparisons) if non_ref_comparisons else 0

                    result["score_iqa"]["aesthetic_ref_comparisons"] = aesthetic_ref_comparisons
                    result["score_iqa"]["aesthetic_ref_overall_win_rate"] = sum(c["win_rate"] for c in aesthetic_ref_comparisons) / len(aesthetic_ref_comparisons) if aesthetic_ref_comparisons else 0

                    if full_ref_comparisons:
                        result["score_iqa"]["full_ref_comparisons"] = full_ref_comparisons
                        result["score_iqa"]["full_ref_overall_win_rate"] = sum(c["win_rate"] for c in full_ref_comparisons) / len(full_ref_comparisons) if full_ref_comparisons else 0
                    
                    self.workflow_logger.info(
                        f"{current_tool} comparison summary:\n"
                        f"  VLM-IQA win rate: {result['vlm_iqa']['win_rate']:.2%}\n"
                        f"  Non-ref score win rate: {result['score_iqa']['non_ref_overall_win_rate']:.2%}\n"
                        f"  Aesthetic-ref score win rate: {result['score_iqa']['aesthetic_ref_overall_win_rate']:.2%}\n"
                        f"  Full-ref score win rate: {result['score_iqa'].get('full_ref_overall_win_rate', 0):.2%}"
                    )
            
            # 输出完整评估结果
            for result in evaluation_results:
                self.workflow_logger.info(
                    f"=== Evaluation Summary for {result['tool_name']} ===\n"
                    f"Execution Time: {result['execution_time']:.2f} seconds\n"
                    f"Score-IQA: {result['score_iqa']}\n"
                    f"VLM-IQA: {result['vlm_iqa']}"
                )

            self.executor.record_executed_subtask(subtask=subtask, evaluation_results=evaluation_results)
            self._calculate_rankings(evaluation_results, has_ground_truth=bool(ground_truth_image))

        self.evaluation_results = evaluation_results
        self.workflow_logger.info("---- Finished Feedback Stage ----")

    def evaluate_by_iqa(self, in_path, ref_path, ntest):
        in_path = Path(in_path) if not isinstance(in_path, Path) else in_path
        
        # 如果需要使用 full-reference metrics 且尚未初始化,则初始化
        if ref_path is not None and not self.metric_paired_dict:
            self.metric_paired_dict["psnr"] = pyiqa.create_metric(
                'psnr', test_y_channel=True, color_space='ycbcr'
            ).to(self.device)
            self.metric_paired_dict["lpips"] = pyiqa.create_metric('lpips').to(self.device)
            self.metric_paired_dict["dists"] = pyiqa.create_metric('dists').to(self.device)
            self.metric_paired_dict["ssim"] = pyiqa.create_metric(
                'ssim', test_y_channel=True, color_space='ycbcr'
            ).to(self.device)

        self.workflow_logger.info(f'Processing image: {in_path}')
        
        result_per_image = {}
        
        im_in = imread(in_path, chn='rgb', dtype='float32')
        im_in_tensor = img2tensor(im_in).to(self.device)
        
        # 使用预加载的 non-reference metrics
        for key, metric in self.metric_dict.items():
            with torch.amp.autocast("cuda"):
                score = metric(im_in_tensor).item()
                result_per_image[key] = score

        # 如果有参考图,使用 full-reference metrics
        if ref_path is not None:
            ref_path = Path(ref_path) if not isinstance(ref_path, Path) else ref_path
            self.workflow_logger.info(f'With reference: {ref_path}')
            im_ref = imread(ref_path, chn='rgb', dtype='float32')
            im_ref_tensor = img2tensor(im_ref).to(self.device)
            im_in_tensor, im_ref_tensor = crop_to_smallest(im_in_tensor, im_ref_tensor)
            
            for key, metric in self.metric_paired_dict.items():
                score = metric(im_in_tensor, im_ref_tensor).item()
                result_per_image[key] = score
        
        return result_per_image

    def _calculate_rankings(self, evaluation_results: list, has_ground_truth: bool) -> None:
        """
        计算各个维度的排名，并按照 tools_index_mapping 的顺序组织结果。
        每个排名列表中的元素是元组 (预测排名_dict, 实际排名)。
        """

        # 从 JSON 中读取预测排名
        # 假设 self.plan_dict 是你之前的 llm_response 转成的字典
        # 每个 plan 包含: tool, metrics, rationale
        tool_name_to_predicted_rank = {}
        for plan in self.plan_dict.values():
            tool_name = plan['tool']
            tool_name_to_predicted_rank[tool_name] = plan['metrics']  # metrics 是字典，四个指标的预测排名

        # 通用函数：根据某个评分字段计算排名（高分优先）
        def _get_actual_ranks(score_fn):
            sorted_tools = sorted(
                [(r['tool_name'], score_fn(r)) for r in evaluation_results],
                key=lambda x: x[1],
                reverse=True,
            )
            return {tool_name: rank for rank, (tool_name, _) in enumerate(sorted_tools, start=1)}

        # 各维度实际排名
        vlm_rank = _get_actual_ranks(lambda r: r['vlm_iqa']['win_rate'])
        non_ref_rank = _get_actual_ranks(lambda r: r['score_iqa']['non_ref_overall_win_rate'])
        aesthetic_ref_rank = _get_actual_ranks(lambda r: r['score_iqa']['aesthetic_ref_overall_win_rate'])
        full_ref_rank = (
            _get_actual_ranks(lambda r: r['score_iqa'].get('full_ref_overall_win_rate', 0))
            if has_ground_truth else {}
        )
        
        # 执行时间排名（时间越短排名越高）
        def _get_time_ranks():
            sorted_tools = sorted(
                [(r['tool_name'], r['execution_time']) for r in evaluation_results],
                key=lambda x: x[1],  # 按时间升序排序
                reverse=False,  # 时间短的排名靠前
            )
            return {tool_name: rank for rank, (tool_name, _) in enumerate(sorted_tools, start=1)}
        
        time_rank = _get_time_ranks()

        # 综合排名 (Borda Count) - 不包含执行时间排名
        n_tools = self.tools_nums
        tool_borda_scores = {}
        for r in evaluation_results:
            tool = r['tool_name']
            score = (
                (n_tools - vlm_rank[tool] + 1)
                + (n_tools - non_ref_rank[tool] + 1)
                + (n_tools - aesthetic_ref_rank[tool] + 1)
            )
            if has_ground_truth:
                score += (n_tools - full_ref_rank[tool] + 1)
            tool_borda_scores[tool] = score

        combined_rank = {
            tool: rank
            for rank, (tool, _) in enumerate(
                sorted(tool_borda_scores.items(), key=lambda x: x[1], reverse=True),
                start=1,
            )
        }

        # 通用函数：返回按照 tools_index_mapping 顺序排列的 (predicted_rank_dict, actual_rank)
        def _map_rank(rank_dict, metric=None, include_execution_time=False):
            results = []
            for tool_name in sorted(self.tools_index_mapping, key=lambda x: self.tools_index_mapping[x]):
                if tool_name in rank_dict:
                    if include_execution_time:
                        # 对于执行时间排名，返回 (执行时间, 排名)
                        execution_time = next((r['execution_time'] for r in evaluation_results if r['tool_name'] == tool_name), 0.0)
                        results.append((execution_time, rank_dict[tool_name]))
                    elif metric is not None:
                        # 如果指定 metric，就返回该指标的预测排名
                        predicted_rank_dict = tool_name_to_predicted_rank.get(tool_name, {})
                        predicted_rank = predicted_rank_dict.get(metric)
                        results.append((predicted_rank, rank_dict[tool_name]))
                    else:
                        # 返回整个 metrics 字典
                        predicted_rank_dict = tool_name_to_predicted_rank.get(tool_name, {})
                        results.append((predicted_rank_dict, rank_dict[tool_name]))
            return results

        # 存储最终结果
        self.vlm_ranking = _map_rank(vlm_rank, metric="VLM-Ref Rank")
        self.non_ref_ranking = _map_rank(non_ref_rank, metric="Non-Ref Rank")
        self.aesthetic_ref_ranking = _map_rank(aesthetic_ref_rank, metric="Aesthetic Rank")
        self.full_ref_ranking = _map_rank(full_ref_rank, metric="Full-Ref Rank") if has_ground_truth else []
        self.time_ranking = _map_rank(time_rank, include_execution_time=True)  # 新增执行时间排名，返回 (执行时间, 排名)
        self.combined_ranking = _map_rank(combined_rank)  # 返回整个 metrics 字典 + 综合实际排名



    def feedback_analysis(self, instruction):
        self.workflow_logger.info("---- Feedback Analysis Stage ----")
        ground_truth_image = instruction.get('gt_image_path', "")

        plan = [{f"{self.degra_subtask_dict[instruction['degradation_info'][0].get('degradation_global')]}":
            [self.plan_dict[plan]["tool"] for _, plan in enumerate(self.plan_dict, start=1)]
        }]

        while plan:
            plan_info = plan.pop(0)
            subtask = list(plan_info.keys())[0]
            tools_info = plan_info[subtask]

            feedback_results = []
            for i, tool_name in enumerate(tools_info):
                feedback_analysis_prompt, json_schema = self.stage1_prompts.get_format_feedback_analysis_prompt(
                    instruction=instruction,
                    solution=self.plan_dict[f"plan{i+1}"],
                    evaluation_result=self.evaluation_results[i],
                    non_ref_ranking=self.non_ref_ranking[i],
                    aesthetic_ref_ranking=self.aesthetic_ref_ranking[i],
                    full_ref_ranking=self.full_ref_ranking[i] if ground_truth_image else None,
                    vlm_based_ranking=self.vlm_ranking[i],
                    time_ranking=self.time_ranking[i],
                    tools_nums=self.tools_nums,
                )

                processed_image_path = (
                    self.img_tree_dir  # 直接使用类属性
                    / f"subtask-{subtask}" 
                    / f"tool-{tool_name}" 
                    / "0-img" 
                    / "output.png"
                    )

                # _, mllm_response = self.mllm.query(
                #     img_path_lst=[ground_truth_image, processed_image_path] if ground_truth_image else [processed_image_path],
                #     prompt=feedback_analysis_prompt,
                # )

                mllm_response = f"### Detailed Analysis of Image Restoration Task\n\n#### 1. Strengths\n- **Effective Dehazing in Specific Regions**: The tool xrestormer demonstrated strong performance in regions where haze severity was very high. The no-reference IQA scores (clipiqa: 0.47, musiq: 71.50, niqe: 3.02, maniqa: 0.45) indicate that the tool was able to significantly reduce the haze, making the image appear clearer in those regions.\n- **Superior Performance Against Certain Baselines**: In no-reference comparisons, xrestormer outperformed dehazeformer and ridcp in 3 out of 4 and 4 out of 4 regions respectively, achieving an overall no-ref win rate of 0.88. This suggests that xrestormer is highly effective in scenarios where reference images are not available.\n- **VLM-based IQA Performance**: The VLM-based IQA score showed that xrestormer won against both dehazeformer and ridcp, indicating that the tool's output is visually more pleasing and coherent according to visual-language models.\n\n#### 2. Weaknesses\n- **Poor Full-Reference Performance**: The full-reference IQA scores (psnr: 16.59, lpips: 0.13, dists: 0.12, ssim: 0.90) and the corresponding win rates (0.00 against both dehazeformer and ridcp) suggest that xrestormer struggles when compared to a clean reference image. This indicates potential issues with preserving fine details and structural similarity.\n- **Limited Adaptability**: The tool's performance dropped significantly in full-reference evaluations, implying that it may not adapt well to scenarios where the ground truth is known. This could be a limitation in practical applications where reference images are available.\n- **Regional Limitations**: Although the tool performed well in regions with very high haze severity, it did not address other types of degradations effectively, as indicated by the uniform degradation features across all regions.\n\n#### 3. Evaluation Domains\n- **No-Reference Metrics**: xrestormer excels in no-reference metrics, making it suitable for scenarios where a reference image is not available. The tool's strength lies in enhancing visual quality perceptually.\n- **Full-Reference Metrics**: The tool's weakness is evident in full-reference metrics, indicating that it may introduce artifacts or fail to preserve structural details when compared to a clean reference image.\n- **VLM-based Evaluation**: The tool's output is visually coherent and pleasing, as indicated by the VLM-based IQA score, suggesting its effectiveness in real-world applications where visual appeal is crucial.\n\n#### 4. Improvement Suggestions\n- **Hybrid Approach**: Combine xrestormer with other tools that excel in full-reference evaluations to leverage the strengths of both. For example, using xrestormer for initial dehazing followed by a refinement step with a tool that preserves structural details could improve overall performance.\n- **Region-Specific Processing**: Develop a region-specific processing pipeline that applies xrestormer only to regions with high haze severity and uses other specialized tools for regions with different types of degradations.\n- **Adaptive Algorithms**: Incorporate adaptive algorithms that can switch between different dehazing techniques based on the severity and type of haze present in the image. This could improve the tool's adaptability and performance across various scenarios.\n- **Training on Diverse Datasets**: Train the tool on a more diverse dataset that includes images with varying types and severities of degradations. This could help the tool generalize better and perform consistently across different evaluation domains."

                feedback_results.append(mllm_response)

        self.workflow_logger.info("---- Finished Feedback Analysis Stage ----")
        self.feedback_analysis_result = feedback_results
        # return feedback_results

    def experience_compression(self, instruction: dict) -> None:
        """
        将新的经验压缩并存入经验池。

        Args:
            new_experience (dict): 新的经验数据。
        """
        self.workflow_logger.info("---- Experience Compression Stage ----")
        experience_compression_prompt, json_schema = self.stage1_prompts.get_format_experience_compression_prompt(
            solution=self.plan_dict,
            feedback_analysis_result=self.feedback_analysis_result
        )

        # _, llm_response = self.llm.query(
        #     prompt=experience_compression_prompt,
        # )
        llm_response = ""

        new_experience = FundamentalExpPoolExperience(
            # <MetaInfo>
            task_id="task" + str(time.strftime("%Y%m%d-%H%M%S")),
            degradation_type=instruction['degradation_info'][0].get('degradation_global'),
            severity_level=instruction['degradation_info'][0].get('severity_level', 'medium'),
            
            # <SolutionInfo>
            task_solution_info=self.plan_dict,
            
            # <FeedbackInfo> - 从 evaluation_results 中提取排名信息
            tools_ref_score_ranking=self.full_ref_ranking,
            tools_non_ref_score_ranking=self.non_ref_ranking,
            tools_aesthetic_ref_score_ranking=self.aesthetic_ref_ranking,
            tools_vlm_score_ranking=self.vlm_ranking,
            tools_avg_score_ranking=self.combined_ranking,

            # <ExperienceInfo> - 从 LLM 的反馈分析结果中提取
            positive_experience="",
            negative_experience="",
            user_preference_experience="",
            image_degradation_content=self.stage1_prompts.get_image_degradation_prompt(instruction),  # 或从 instruction 中获取
            image_degradation_content_embedding=None,  # 将在 add_experience 时自动计算
        )

        self.fundamental_experience_pool.add_experience(
            experience=new_experience
        )
        # 可选：将经验池保存到文件
        # self.fundamental_experience_pool.save_experiences_to_file("/data/zkl/AgenticIR/exploration_self_evolve/output/experiences.json")
        self.workflow_logger.info("---- Finished Experience Compression Stage ----")

    def _get_tools_ranking_by_metric(self, metric_type: str) -> list[int]:
        """根据指标类型获取工具排名"""
        if metric_type not in ['full_reference', 'non_reference']:
            return []
        
        # 计算每个工具的平均分数
        tool_scores = []
        for result in self.evaluation_results:
            metrics = result['score_iqa'].get(metric_type, {})
            if not metrics:
                tool_scores.append((result['tool_name'], 0))
                continue
            avg_score = sum(metrics.values()) / len(metrics)
            tool_scores.append((result['tool_name'], avg_score))
        
        # 排序并返回排名
        sorted_tools = sorted(tool_scores, key=lambda x: x[1], reverse=True)
        return [i + 1 for i in range(len(sorted_tools))]

    def _get_tools_vlm_ranking(self) -> list[int]:
        """根据 VLM-IQA 结果获取工具排名"""
        tool_scores = []
        for result in self.evaluation_results:
            win_rate = result['vlm_iqa'].get('win_rate', 0)
            tool_scores.append((result['tool_name'], win_rate))
        
        sorted_tools = sorted(tool_scores, key=lambda x: x[1], reverse=True)
        return [i + 1 for i in range(len(sorted_tools))]

    def _get_tools_avg_ranking(self) -> list[int]:
        """使用 Borda Count 计算平均排名"""
        # 获取三种排名
        ref_ranking = self._get_tools_ranking_by_metric('full_reference')
        non_ref_ranking = self._get_tools_ranking_by_metric('non_reference')
        vlm_ranking = self._get_tools_vlm_ranking()
        
        # Borda Count: 排名越高,分数越低
        n_tools = len(self.evaluation_results)
        tool_borda_scores = [0] * n_tools
        
        for i in range(n_tools):
            if ref_ranking:
                tool_borda_scores[i] += ref_ranking[i]
            tool_borda_scores[i] += non_ref_ranking[i]
            tool_borda_scores[i] += vlm_ranking[i]
        
        # 根据 Borda 分数排序(分数越小越好)
        ranked_indices = sorted(range(n_tools), key=lambda i: tool_borda_scores[i])
        final_ranking = [0] * n_tools
        for rank, idx in enumerate(ranked_indices, start=1):
            final_ranking[idx] = rank
        
        return final_ranking

    def _extract_brief_reflection(self, detailed_reflection: str) -> str:
        """从详细反思中提取简要反思"""
        # 可以使用 LLM 提取或简单截取前几句
        lines = detailed_reflection.split('\n')
        brief_lines = [line for line in lines[:3] if line.strip() and not line.startswith('#')]
        return ' '.join(brief_lines[:2]) if brief_lines else detailed_reflection[:200]
    
    def run_stage1(self, evolution_instruction: list[dict]) -> None:
        # 1. 获取 instruction 的信息
        
        # 2. 跳过 task evolution 阶段

        # 3. 进入 solution evolution 阶段
        self.solution_evolution(evolution_instruction)

        # 4. 进入 execute subtask 阶段
        self.execute_subtask(None)

        # 5. 进入 feedback 阶段
        self.evaluate_solutions(evolution_instruction)

        # 6. 进入 feedback 分析阶段
        self.feedback_analysis(evolution_instruction)

        # 7. 进入 experience compression 阶段
        self.experience_compression(evolution_instruction)

    def _get_name_stem(self, name: str) -> str:
        return name[name.find("-") + 1 :]
    
    @property
    def _img_tree(self) -> ImgTree:
        return ImgTree(self.img_tree_dir, html_dir=self.log_dir)
    
    def _render_img_tree(self) -> None:
        self._img_tree.to_html()

    def _dump_summary(self) -> None:
        with open(self.work_mem_path, "w") as f:
            json.dump(self.work_mem, f, indent=2)
  