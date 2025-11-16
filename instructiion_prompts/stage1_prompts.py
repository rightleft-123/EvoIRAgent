class Stage1Prompts:
    def __init__(self) -> None:
        self.system_message = "You are an expert in image restoration. Given an image of low quality, your task is guiding the user to utilize various tools to enhance its quality. The input image may suffer from various kinds of degradations, including low resolution, noise, motion blur, defocus blur, haze, rain, dark, and jpeg compression artifact. The available tools each specialize in addressing one of the above eight degradations, i.e., super-resolution, denoising, motion deblurring, defocus deblurring, dehazing, deraining, brightening, and jpeg compression artifact removal. The following will be a continuation of an interaction between you and a user to restore an image. Note that if the user specifies the output format, you must strictly follow it without any other words."
        
        self.solution_evolution_prompt = """
        You are an expert in image restoration with comprehensive knowledge of various types of image degradation and their severity levels, as well as extensive proficiency in employing multiple specialized tools to improve image quality. Your goal is to accurately identify the degradation types present in an image based on both textual and visual information, and to effectively utilize the available tools to address these degradations.\n\n
        The current image's degradation information includes a specified degradation type denoted as: {degradation_type}, with a defined severity level: {severity_level}. The image is segmented into four distinct regions, each containing detailed degradation characteristics described as: {image_degradation_content}.\n\n
        You have access to a set of restoration tools summarized as: {tools_basic_info}; please pay close attention to their index numbers. When generating your restoration plans, organize them in ascending order according to these tool indices. Additionally, you have historical data on each tool's past performance and execution details, provided as: {tools_execution_info}, which should inform your planning.\n\n
        Furthermore, you possess prior experience and insights from handling similar degradation cases, encapsulated as: {experience_info}, which should guide and optimize your tool selection strategy. If there are any specific current requirements, indicated as: {specific_requirement}, these must be prioritized above all else when formulating your plans.\n\n
        Using all this information, your task is to develop {num_plans} distinct execution plans for image restoration. Each plan must specify the chosen tool, predict and rank its performance across four metrics—Full-Ref Score, Non-Ref Score, VLM-Ref Score, and Aesthetic Score—relative to the other plans, and include a concise rationale explaining the reasoning behind the tool selection and the ranking of each score. The output should present the plans sequentially, ordered by the tool indices from smallest to largest.
        """

        self.feedback_analysis_prompt = """
        You are an expert in the field of image restoration, specializing in critically reflecting on and refining the image restoration information you have processed, including both textual and visual data. 
        Your primary objective is to deeply understand how various types of image degradation and their severity levels manifest in low-quality images.
        The specific task at hand is: remove {degradation_type} with a severity level of {severity_level} from a given image. This image contains detailed degradation information segmented across four distinct regions: {detailed_info}.
        In the current image processing workflow, your implemented restoration plan is: {current_plan}, which includes the predicted rankings across four evaluation dimensions prior to execution from other expert model. Following a comprehensive image quality assessment procedure, your plan’s performance was ranked among {tools_nums} different approaches, with scores and rankings detailed as {ranking_info}.
        Your responsibility now is to conduct a thorough analysis of the strengths and weaknesses of your current plan by integrating five key elements: the task objective, task-specific information, image data, the proposed restoration plan, and the final evaluation rankings.
        Begin your analysis anchored in the shared task context, focusing on two main dimensions:
        1.	Task Completion and Image Understanding: Assess how well the task was completed, the scenarios where the solution is applicable, and the understanding of the paired ground truth and restored images.
        2.	Dual Analytical Perspectives:
        - Data-centric Analysis: Interpret the ranking and score results objectively, strictly based on your own plan’s performance metrics without speculating about other methods, highlighting the advantages and disadvantages of your execution strategy.
        - User-centric Analysis: Consider the evaluation criteria from the user’s perspective, emphasizing factors such as fidelity, perceptual quality, aesthetics, and computational efficiency, reflecting potential user priorities.
        Ensure your feedback is balanced, concise, and precise, as it will serve as verification material for another expert’s review. Any discrepancies or inaccurate analyses may lead to serious consequences.
        """

        self.experience_compression_prompt = """
        You are an expert skilled in synthesizing experiential knowledge and organizing any input information into well-founded, coherent, and in-depth textual summaries. Your specialty lies in accurately and succinctly comparing multiple proposed solutions by clearly outlining their respective advantages and disadvantages.
        Currently, there are {tools_nums} execution plans related to the present task scenario. Each plan includes the plan name, predicted ranking with reasoning, actual verified ranking, and a detailed list of its strengths and weaknesses. The execution plans are as follows: {execution_plans}.
        Your task is to produce a comprehensive summary evaluating all these plans collectively. Horizontally, focus on identifying both the differences and commonalities in strengths and weaknesses across the various solutions. Vertically, concentrate on extracting insights about their performance across shared dimensions, particularly emphasizing two sub-dimensions: data-centric and user-centric aspects.
        In your final output, provide a concise synthesis/insights covering three key areas: positive information, negative information, and user preference insights. This summary should effectively guide future predictions and decision-making for similar task scenarios, enabling execution outcomes that closely align with actual results.
        """

    def get_image_degradation_prompt(self, instruction: dict) -> str:
        degradation_prompt = ""
        for i in range(len(instruction["degradation_info"])):
            degradation_type, severity_level = instruction["degradation_info"][i].get("degradation_global"), instruction["degradation_info"][i].get("severity_global")
            detailed_info = ""
            detailed_info += f"On the top left: the {degradation_type} severity is {instruction['degradation_info'][i]['degradation_local'].get('severity_top_left')}, the degradation features are {instruction['degradation_info'][i]['degradation_local'].get('detail_desc_top_left')}\n"
            detailed_info += f"On the top right: the {degradation_type} severity is {instruction['degradation_info'][i]['degradation_local'].get('severity_top_right')}, the degradation features are {instruction['degradation_info'][i]['degradation_local'].get('detail_desc_top_right')}\n"
            detailed_info += f"On the bottom left: the {degradation_type} severity is {instruction['degradation_info'][i]['degradation_local'].get('severity_bottom_left')}, the degradation features are {instruction['degradation_info'][i]['degradation_local'].get('detail_desc_bottom_left')}\n"
            detailed_info += f"On the bottom right: the {degradation_type} severity is {instruction['degradation_info'][i]['degradation_local'].get('severity_bottom_right')}, the degradation features are {instruction['degradation_info'][i]['degradation_local'].get('detail_desc_bottom_right')}\n"
            degradation_prompt += f"{i+1}. The image suffers from {degradation_type} with severity {severity_level}. The detailed degradation related information in four regions are as follows: \n{detailed_info}"
        
        return degradation_prompt


    def get_experience_retrieval_prompt(self, instruction: dict) -> str:
        retrieval_prompt = ""
        for i in range(len(instruction["degradation_info"])):
            degradation_type, severity_level = instruction["degradation_info"][i].get("degradation_global"), instruction["degradation_info"][i].get("severity_global")
            detailed_info = ""
            detailed_info += f"On the top left: the {degradation_type} severity is {instruction['degradation_info'][i]['degradation_local'].get('severity_top_left')}, the degradation features are {instruction['degradation_info'][i]['degradation_local'].get('detail_desc_top_left')}\n"
            detailed_info += f"On the top right: the {degradation_type} severity is {instruction['degradation_info'][i]['degradation_local'].get('severity_top_right')}, the degradation features are {instruction['degradation_info'][i]['degradation_local'].get('detail_desc_top_right')}\n"
            detailed_info += f"On the bottom left: the {degradation_type} severity is {instruction['degradation_info'][i]['degradation_local'].get('severity_bottom_left')}, the degradation features are {instruction['degradation_info'][i]['degradation_local'].get('detail_desc_bottom_left')}\n"
            detailed_info += f"On the bottom right: the {degradation_type} severity is {instruction['degradation_info'][i]['degradation_local'].get('severity_bottom_right')}, the degradation features are {instruction['degradation_info'][i]['degradation_local'].get('detail_desc_bottom_right')}"
            retrieval_prompt += f"{i+1}. Remove {degradation_type} with severity {severity_level} from an image. The image also contains detailed degradation related information in four regions: \n{detailed_info}"
        
        return retrieval_prompt

    def get_system_message(self) -> str:
        return self.system_message

    def get_format_solution_evolution_prompt(
        self,
        instruction: dict,
        tools_basic_info: str,
        tools_execution_info: dict,
        experience_info: str,
        num_plans: int,
        specific_requirement: str="No specific requirement"
    ) -> str:
        
        image_degradation_content = ""
        for i in range(len(instruction["degradation_info"])):
            degradation_type, severity_level = instruction["degradation_info"][i].get("degradation_global"), instruction["degradation_info"][i].get("severity_global")
            detailed_info = ""
            detailed_info += f"On the top left: the {degradation_type} severity is {instruction['degradation_info'][i]['degradation_local'].get('severity_top_left')}, the degradation features are {instruction['degradation_info'][i]['degradation_local'].get('detail_desc_top_left')}\n"
            detailed_info += f"On the top right: the {degradation_type} severity is {instruction['degradation_info'][i]['degradation_local'].get('severity_top_right')}, the degradation features are {instruction['degradation_info'][i]['degradation_local'].get('detail_desc_top_right')}\n"
            detailed_info += f"On the bottom left: the {degradation_type} severity is {instruction['degradation_info'][i]['degradation_local'].get('severity_bottom_left')}, the degradation features are {instruction['degradation_info'][i]['degradation_local'].get('detail_desc_bottom_left')}\n"
            detailed_info += f"On the bottom right: the {degradation_type} severity is {instruction['degradation_info'][i]['degradation_local'].get('severity_bottom_right')}, the degradation features are {instruction['degradation_info'][i]['degradation_local'].get('detail_desc_bottom_right')}\n"
            image_degradation_content += f"{i+1}. The image suffers from {degradation_type} with severity {severity_level}. The detailed degradation related information in four regions are as follows: {detailed_info}.\n"

        tools_execution_info_str = f"This information is gathered by {tools_execution_info['Total execution cnt']} executions.\n There are 4 aspects for each tool, including Ref Score, Non-Ref Score, VLM Score, and Aesthetic Score (it's statistical meaning is generating from Borda Count which means the higher the score, more preferred).\n"
        for tool_name, exec_info in tools_execution_info.items():
            if tool_name != "Total execution cnt":
                tools_execution_info_str += f"Tool: {tool_name}, Past Executions: Full-Ref Score: {exec_info['Full-Ref Score']}, Non-Ref Score: {exec_info['Non-Ref Score']}, VLM-Ref Score: {exec_info['VLM-Ref Score']}, Aesthetic Score: {exec_info['Aesthetic-Ref Score']}\n"
        
        prompt = self.solution_evolution_prompt.format(
            degradation_type=degradation_type,
            severity_level=severity_level,
            image_degradation_content=image_degradation_content,
            tools_basic_info=tools_basic_info,
            tools_execution_info=tools_execution_info_str,
            experience_info=experience_info,
            specific_requirement=specific_requirement,
            num_plans=num_plans
        )

        json_schema = {
            "type": "object",
            "properties": {},
            "required": []
        }

        metrics_rank = ["Full-Ref Rank", "Non-Ref Rank", "VLM-Ref Rank", "Aesthetic Rank"]

        for i in range(num_plans):
            json_schema["properties"][f"plan{i+1}"] = {
                "type": "object",
                "properties": {
                    "tool": {
                        "type": "string",
                        "description": "The tool used for the plan"
                    },
                    "metrics_rank": {
                        "type": "object",
                        "properties": {
                            metric_rank: {
                                "type": "integer",
                                "description": f"Rank of this plan relative to other plans for {metric_rank}"
                            } for metric_rank in metrics_rank
                        },
                        "required": metrics_rank
                    },
                    "rationale": {
                        "type": "string",
                        "description": "The rationale behind the tool selection and ranking of each score"
                    }
                },
                "required": ["tool", "metrics_rank", "rationale"]
            }
            json_schema["required"].append(f"plan{i+1}")

        return prompt, json_schema

    def get_format_feedback_analysis_prompt(
        self,
        instruction: dict,
        solution: str,
        evaluation_result: dict,
        non_ref_ranking: tuple,
        aesthetic_ref_ranking: tuple,
        full_ref_ranking: tuple,
        vlm_based_ranking: tuple,
        time_ranking: tuple,
        tools_nums: int
    ) -> str:
        degradation, severity_level = instruction["degradation_info"][0].get("degradation_global"), instruction["degradation_info"][0].get("severity_global")
        detailed_info = ""
        detailed_info += f"On the top left: the {degradation} severity is {instruction['degradation_info'][0]['degradation_local'].get('severity_top_left')}, the degradation features are {instruction['degradation_info'][0]['degradation_local'].get('detail_desc_top_left')}\n"
        detailed_info += f"On the top right: the {degradation} severity is {instruction['degradation_info'][0]['degradation_local'].get('severity_top_right')}, the degradation features are {instruction['degradation_info'][0]['degradation_local'].get('detail_desc_top_right')}\n"
        detailed_info += f"On the bottom left: the {degradation} severity is {instruction['degradation_info'][0]['degradation_local'].get('severity_bottom_left')}, the degradation features are {instruction['degradation_info'][0]['degradation_local'].get('detail_desc_bottom_left')}\n"
        detailed_info += f"On the bottom right: the {degradation} severity is {instruction['degradation_info'][0]['degradation_local'].get('severity_bottom_right')}, the degradation features are {instruction['degradation_info'][0]['degradation_local'].get('detail_desc_bottom_right')}\n"

        current_plan = f"Plan: Tool = {solution['tool']}, Predict FR ranking = {solution['metrics']['Full-Ref Rank']}, Predict NR ranking = {solution['metrics']['Non-Ref Rank']}, Predict VLM ranking = {solution['metrics']['VLM-Ref Rank']}, Predict Aesthetic ranking = {solution['metrics']['Aesthetic Rank']}, Rationale = {solution['rationale']}"

        ranking_info = ""
        ranking_info += f"Full-reference score indicators that focus on the numerical differences between images and reference images, emphasizing the quantification of pixel, structure, or perceptual similarity, emphasizing \"reconstruction accuracy\" and minimizing errors. They are suitable for scenes with reference images, but may differ from human subjective perception.\n"
        ranking_info += f"Full-reference score: {', '.join([f'{k}: {v:.2f}' for k, v in evaluation_result['score_iqa']['full_reference'].items()])}.\n"
        ranking_info += f"Full-reference rankings compared to another {tools_nums} model: {full_ref_ranking[1]} / {tools_nums}\n"
        
        ranking_info += f"Non-reference score index evaluates quality without a reference image, predicts visual quality through internal statistical features or learning models of the image, focuses on universality and robustness, and is closer to practical applications, but the prediction may still differ from subjective perception.\n"
        ranking_info += f"Non-reference score: {', '.join([f'{k}: {v:.2f}' for k, v in evaluation_result['score_iqa']['non_reference'].items()])}.\n"
        ranking_info += f"Non-reference rankings compared to another {tools_nums} model: {non_ref_ranking[1]} / {tools_nums}\n"

        ranking_info += f"Aesthetic score index evaluates the visual quality of images based on aesthetic principles, focusing on factors such as composition, color harmony, and overall visual appeal. It is particularly useful for applications where subjective perception of beauty is important.\n"
        ranking_info += f"Aesthetic-reference score: {', '.join([f'{k}: {v:.2f}' for k, v in evaluation_result['score_iqa']['aesthetic_reference'].items()])}.\n"
        ranking_info += f"Aesthetic-reference rankings compared to another {tools_nums} model: {aesthetic_ref_ranking[1]} / {tools_nums}\n"

        ranking_info += f"Text feedback based on VLM generates text descriptions through visual language models, pointing out issues such as image blurring, color cast, or structural defects. It emphasizes interpretability and multidimensional feedback, making the evaluation closer to human subjective feelings and suitable for scenarios that require feedback explanation rather than just scores.\n"
        ranking_info += f"VLM-based win rate compared to another {tools_nums} model is: {evaluation_result['vlm_iqa']['win_rate']}.\n"        
        ranking_info += f"VLM-based rankings compared to another {tools_nums} model: {vlm_based_ranking[1]} / {tools_nums}\n"

        ranking_info += f"Execution times evaluates the efficiency of image restoration, focusing on processing speed and resource consumption. It is particularly important in real-time applications or scenarios with limited computational resources, balancing quality and efficiency.\n"
        ranking_info += f"Execution time: {time_ranking[0]:.2f}.\n"
        ranking_info += f"Execution time rankings compared to another {tools_nums} model: {time_ranking[1]} / {tools_nums}\n"

        prompt = self.feedback_analysis_prompt.format(
            degradation_type=degradation,
            severity_level=severity_level,
            detailed_info=detailed_info,
            current_plan=current_plan,
            ranking_info=ranking_info,
            tools_nums=tools_nums
        )

        json_schema = {
            "type": "object",
            "properties": {
                "analysis": {
                "type": "object",
                "properties": {
                    "task_completion_and_image_understanding": {
                        "type": "string",
                        "description": "Analysis of task completion and image understanding"
                    },
                    "data_centric_analysis": {
                        "type": "string",
                        "description": "Analysis of data-centric aspects"
                    },
                    "user_centric_analysis": {
                        "type": "string",
                        "description": "Analysis of user-centric aspects"
                    }
                },
                "required": [
                    "task_completion_and_image_understanding",
                    "data_centric_analysis",
                    "user_centric_analysis"
                ]
                }
            },
            "required": ["analysis"]
        }

        return prompt, json_schema

    def get_format_experience_compression_prompt(
        self,
        solution: dict,
        feedback_analysis_result: list,
    ) -> str:
        
        execution_plans = ""
        for i in range(len(solution)):
            execution_plans += f"\n--- Execution Plan {i+1} ---\n"
            execution_plans += f"Plan: Tool = {solution[f'plan{i+1}'] ['tool']}, Predict FR ranking = {solution[f'plan{i+1}'] ['metrics']['Full-Ref Rank']}, Predict NR ranking = {solution[f'plan{i+1}'] ['metrics']['Non-Ref Rank']}, Predict VLM ranking = {solution[f'plan{i+1}'] ['metrics']['VLM-Ref Rank']}, Predict Aesthetic ranking = {solution[f'plan{i+1}'] ['metrics']['Aesthetic Rank']}, Rationale = {solution[f'plan{i+1}'] ['rationale']}"
            execution_plans += f"For plan{i+1} the feedback is: {feedback_analysis_result[i]}\n\n"
        
        prompt = self.experience_compression_prompt.format(
            tools_nums=len(solution),
            execution_plans=execution_plans
        )

        json_schema = {
            "type": "object",
            "properties": {
                "positive_information": { 
                    "type": "string", 
                    "description": "Key strengths, successful outcomes, or effective strategies observed." 
                },
                "negative_information": { 
                    "type": "string", 
                    "description": "Weaknesses, challenges, or underperforming aspects identified." 
                },
                "user_preference_insights": { 
                    "type": "string", 
                    "description": "Observed or inferred user priorities, tendencies, and decision-making cues." 
                }
            },
            "required": ["positive_information", "negative_information", "user_preference_insights"]
        }

        return prompt, json_schema

