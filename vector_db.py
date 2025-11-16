import os
import json
import sys
import numpy as np
import time
import random
from collections import defaultdict
from typing import List, Dict, Optional, Union
from dataclasses import dataclass, field
from FlagEmbedding import FlagModel
import logging
logger = logging.getLogger(__name__)

encode_model = FlagModel("BAAI/bge-large-en-v1.5", 
                query_instruction_for_retrieval="Represent this sentence for searching relevant passages:") 


def calculate_cos_similarity_A_and_Batch_B(A, B):
    dot_product = np.dot(A, B.T)
    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B, axis=1)  # (4,)
    cosine_sim = dot_product / (norm_B * norm_A)
    return cosine_sim


def calculate_cos_similarity_A_and_B(A, B):
    dot_product = np.dot(B, A)
    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B)
    cosine_sim = dot_product / (norm_B * norm_A)
    return cosine_sim


@dataclass
# Experiences are related to Photoshop tool usage
class ExpPool1Experience:
    """Router experience record"""
    decision: str               # 'forward', 'split', 'execute'


@dataclass
# Experiences are related to external tool usage
class FundamentalExpPoolExperience:
    """Router experience record"""
    "<MetaInfo>"
    task_id: int
    degradation_type: str      # Degradation type (e.g., blur, noise)
    severity_level: str        # Severity level (e.g., very low, low, medium, high, very high)

    "<SolutionInfo>"
    task_solutions_info: List[int]       # Ranking reasons of the solutions among all generated solutions

    "<FeedbackInfo>"
    tools_ref_score_ranking: List[int]          # Ranked list of tools in reference socre IQA for this degradation (predict ranking, feedback ranking)
    tools_non_ref_score_ranking: List[int]      # Ranked list of tools in non-reference score IQA for this degradation (predict ranking, feedback ranking)
    tools_aesthetic_score_ranking: List[int]    # Ranked list of tools in aesthetic score IQA for this degradation (predict ranking, feedback ranking)
    tools_vlm_score_ranking: List[int]          # Ranked list of tools in VLM score IQA for this degradation (predict ranking, feedback ranking)
    tools_avg_score_ranking: List[int]          # Ranked list of tools in average score IQA for this degradation (Borda Count)

    "<ExperienceInfo>"
    positive_experience: str          # Description of positive experiences with the tool
    negative_experience: str          # Description of negative experiences with the tool
    user_preference_experience: str   # Description of user preference experiences with the tool
    image_degradation_content: str    # Description about the degradation information on the image
    image_degradation_content_embedding: np.ndarray = None  # Vector representation of the image degradation content

    timestamp: float = field(default_factory=time.time)  # Time when this experience was recorded

@dataclass
# Experiences are related to combined tool usage (combination & permutation of tools)
class ExpPool3Experience:
    """Router experience record"""


class BaseExperiencePool:
    """Base class for experience pool"""
    def __init__(self, capacity: int = 50):
        self.capacity = capacity
        self.embedding_cache = {}
        self.experiences = defaultdict(list)

    def __len__(self):
        return len(self.experiences)
    
    def compute_embedding(self, text: str) -> np.ndarray:
        """Compute vector representation of text"""
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        global encode_model
        embedding = encode_model.encode(text)

        self.embedding_cache[text] = embedding
        return embedding

    def spearman_rho(self, true_rank, pred_rank) -> float:
        """
        计算 Spearman 等级相关系数 (rho)
        true_rank: 真实排名列表，如 [1, 2, 3]
        pred_rank: 实际排名列表，如 [3, 1, 2]
        返回值: rho ∈ [-1, 1]
        """
        true_rank = np.array(true_rank)
        pred_rank = np.array(pred_rank)
        n = len(true_rank)
        diff = true_rank - pred_rank
        diff_sq_sum = np.sum(diff ** 2)
        rho = 1 - (6 * diff_sq_sum) / (n * (n**2 - 1))
        return rho


class RouterExperiencePool(BaseExperiencePool):
    def add_experience(self, experience: FundamentalExpPoolExperience) -> None:
        degradation_type = experience.degradation_type
        # ensure existence of experience embedding(task_major_problem + task_progress_text + task_description)
        if not experience.image_degradation_content_embedding:
            experience.image_degradation_content_embedding = self.compute_embedding(experience.image_degradation_content)

        self.experiences[degradation_type].append(experience)
        success_experience_num = len(self.experiences[degradation_type])
        if success_experience_num > self.capacity and self.capacity >= 0:
            self._smart_eviction(degradation_type, experience)

    def update_experience(self, experience: FundamentalExpPoolExperience) -> None:
        """Update an existing experience in the pool"""
        degradation_type = experience.degradation_type
        for idx, exp in enumerate(self.experiences[degradation_type]):
            if exp.task_id == experience.task_id:
                self.experiences[degradation_type][idx] = experience
                break


    def _smart_eviction(self, degradation_type: str, new_experience: FundamentalExpPoolExperience) -> None:
        """Smart eviction strategy, let the LLM Agent decide which trajectory to remove"""
        current_time = time.time()

        retrieval_info = self.get_relevant_experiences(
            task=new_experience.brief_reflection,
            degradation_type=new_experience.degradation_type,
            top_k=1,
            threshold=0.5,
            success_only=True
        )

        trajectory_info = []
        # Generate input for the agent: the new experience and the existing experiences in the pool
        for exp in retrieval_info:
            if exp.severity_level == new_experience.severity_level:
                trajectory_info.append({
                    'task_id': exp.task_id,
                    'degradation_type': exp.degradation_type,
                    'severity_level': exp.severity_level,
                    'tools_name': exp.tools_name,
                    'tools_ranking': exp.tools_ranking,
                    'task_solution': exp.task_solution_tool_name,
                    'task_solution_ranking': exp.task_solution_ranking,
                    'detail_reflection': exp.detail_reflection,
                    'brief_reflection': exp.brief_reflection,
                })
        if len(trajectory_info) == 0:
            random_remove_id = random.choice(retrieval_info).task_id
            self.remove_selected_trajectory(degradation_type, random_remove_id, new_experience)
            return 0

        # Current task information (new experience)
        new_task_info = {
            'task_id': new_experience.task_id,
            'degradation_type': new_experience.degradation_type,
            'severity_level': new_experience.severity_level,
            'tools_name': new_experience.tools_name,
            'tools_ranking': new_experience.tools_ranking,
            'task_solution': new_experience.task_solution_tool_name,
            'task_solution_ranking': new_experience.task_solution_ranking,
            'detail_reflection': new_experience.detail_reflection,
            'brief_reflection': new_experience.brief_reflection,
        }

        # Prepare the prompt for LLM Agent decision-making
        agent_input = self.prepare_agent_input(trajectory_info, new_task_info)

        # Generate the decision through LLM: return the task_id of the trajectory to remove
        trajectory_to_remove = self.make_agent_decision(agent_input)

        # Evict the selected trajectory based on agent's decision
        self.remove_selected_trajectory(degradation_type, trajectory_to_remove, new_experience)

    def prepare_agent_input(self, trajectory_info: List[Dict], new_task_info: Dict) -> str:
        """Prepare the input prompt for the LLM agent"""
        input_str = "Here is the list of historical router trajectories (task experiences):\n"
        for traj in trajectory_info:
            input_str += f"Task ID: {traj['task_id']}, Task Type: {traj['degradation_type']}, Severity Level: {traj['severity_level']}, Tools Information: {traj['tools_name']}, Real Ranking: {traj['tools_ranking']}, Predict Ranking: {traj['task_solution_ranking']}, Detr: {traj['detail_reflection']}, Brief Reflection: {traj['brief_reflection']}\n"

        input_str += "\nCurrent Task Information:\n"
        input_str += f"Task ID: {new_task_info['task_id']}, Task Type: {new_task_info['degradation_type']}, Severity Level: {new_task_info['severity_level']}, Tools Information: {new_task_info['tools_name']}, Real Ranking: {new_task_info['tools_ranking']}, Predict Ranking: {new_task_info['task_solution_ranking']}, Detr: {new_task_info['detail_reflection']}, Brief Reflection: {new_task_info['brief_reflection']}\n"

        input_str += "\nPlease decide which trajectory would be most beneficial for improving your abilities. Choose the least valuable trajectory (either the new one or an existing one) to remove, and return the Task ID of the trajectory to remove.\n"

        input_str += "\nReturn the Task ID of the trajectory to evict.\n"

        return input_str

    def make_agent_decision(self, agent_input: str) -> str:
        """Generate decision based on the agent's evaluation using LLM"""
        # Feed input to LLM model (like encode_model) for decision-making
        decision_embedding = encode_model.encode(agent_input)
        
        # Interpret the output decision from LLM: which task ID to evict
        trajectory_to_remove = self.interpret_decision(decision_embedding)
        
        return trajectory_to_remove

    def interpret_decision(self, decision_embedding: np.ndarray) -> str:
        """Interpret the decision from the LLM model output"""
        # Assuming the model outputs an embedding that corresponds to the task ID of the trajectory to remove
        # For example, the model might output a high score for the most likely task ID to evict
        task_id_to_remove = np.argmax(decision_embedding)  # Assuming the model generates a vector for task IDs
        
        return str(task_id_to_remove)

    def remove_selected_trajectory(self, degradation_type: str, task_id_to_remove: str, new_experience: FundamentalExpPoolExperience) -> None:
        """Remove the selected trajectory from the experience pool based on Task ID"""
        # Check if the trajectory to remove is the new experience
        if task_id_to_remove == new_experience.task_id:
            # If the new experience is chosen to be removed, don't add it
            print(f"Evicting new experience with Task ID: {task_id_to_remove}")
        else:
            # Remove the specified experience by Task ID
            for exp in self.experiences[degradation_type]:
                if exp.task_id == task_id_to_remove:
                    self.experiences[degradation_type].remove(exp)
                    print(f"Evicting existing experience with Task ID: {task_id_to_remove}")
                    break
    
    def calculate_similarity(self, exp1: FundamentalExpPoolExperience, exp2: FundamentalExpPoolExperience) -> float:
        """Calculate similarity between two routing experiences"""
        # Calculate vector similarity
        vector_similarity = np.dot(exp1.embedding, exp2.embedding) / (np.linalg.norm(exp1.embedding) * np.linalg.norm(exp2.embedding))
        
        # Consider routing decision similarity
        decision_similarity = 1.0 if exp1.decision == exp2.decision else 0.5
        
        # Consider target agent similarity
        target_similarity = 1.0 if exp1.target_agent_id == exp2.target_agent_id else 0.7
        
        # Weighted average
        return 1 * vector_similarity + 0 * decision_similarity + 0 * target_similarity

    def get_relevant_experiences(self, task, instruction, top_k=3, threshold=0.7, return_str=False):
        """Get relevant routing experiences, add threshold check"""
        degradation_type = instruction['degradation_info'][0].get('degradation_global')
        severity_level = instruction['degradation_info'][0].get('severity_global')
        query_embedding = self.compute_embedding(task)
        all_scores = []

        for exp in self.experiences[degradation_type]:
            # Check if the experience matches the severity level
            if exp.severity_level != severity_level:
                continue
            # Basic similarity, mainly similarity between task context and task description
            similarity = np.dot(query_embedding, exp.image_degradation_content_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(exp.image_degradation_content_embedding))
            all_scores.append((-similarity, exp))

        # Sort by similarity
        sorted_scores = sorted(all_scores, key=lambda x: x[0])
        score_list = []
        for score, exp in sorted_scores:
            score_list.append(score)
        # print(f"score_list: {score_list}")
        logger.info(f"score_list: {score_list}")
        logger.info(f"top_k: {top_k}")
        if len(sorted_scores) == 0:
            return "<no valid experiences>"
        
        # If the number of valid experiences is less than top_k, return all valid experiences
        valid_success_experiences = [exp for _, exp in sorted_scores if -1 *_ >= threshold]
        experience = []
        if len(valid_success_experiences) < top_k:
            experience.extend(valid_success_experiences)
        else:
            experience.extend(valid_success_experiences[:top_k])

        if return_str:
            return ""
        else:
            return experience


    def get_newest_experience(self,degradation_type, k=1):
        experiences = []
        experiences.extend(self.experiences[degradation_type])
        return experiences[:-k]

    def get_all_experiences(self):

        success_return_list = []

        for degradation_type, experiences in self.experiences.items():
            for experience in experiences:
                temp_dict = {
                    'task_id': experience.task_id,
                    'degradation_type': experience.degradation_type,
                    'severity_level': experience.severity_level,
                    'task_solution_info': experience.task_solution_info,
                    'tools_ref_score_ranking': experience.tools_ref_score_ranking,
                    'tools_non_ref_score_ranking': experience.tools_non_ref_score_ranking,
                    'tools_aesthetic_score_ranking': experience.tools_aesthetic_score_ranking,
                    'tools_vlm_score_ranking': experience.tools_vlm_score_ranking,
                    'tools_avg_score_ranking': experience.tools_avg_score_ranking,
                    'positive_experience': experience.positive_experience,
                    'negative_experience': experience.negative_experience,
                    'user_preference_experience': experience.user_preference_experience,
                    'image_degradation_content': experience.image_degradation_content,
                    'timestamp': experience.timestamp,
                }
                success_return_list.append(temp_dict)

        return [*success_return_list]
    
    def write_experiences_to_file(self, filepath: str) -> None:
        """Write all experiences to a JSON file"""
        all_experiences = self.get_all_experiences()
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(all_experiences, f, ensure_ascii=False, indent=4)
        logger.info(f"Experiences written to {filepath}")

    def read_experiences_from_file(self, filepath: str) -> None:
        """Read experiences from a JSON file and load them into the pool"""
        if not os.path.exists(filepath):
            logger.warning(f"File {filepath} does not exist.")
            return
        
        with open(filepath, 'r', encoding='utf-8') as f:
            all_experiences = json.load(f)
        
        for exp_data in all_experiences:
            experience = FundamentalExpPoolExperience(
                task_id=exp_data['task_id'],
                degradation_type=exp_data['degradation_type'],
                severity_level=exp_data['severity_level'],
                task_solutions_info=exp_data['task_solutions_info'],
                tools_ref_score_ranking=exp_data['tools_ref_score_ranking'],
                tools_non_ref_score_ranking=exp_data['tools_non_ref_score_ranking'],
                tools_aesthetic_score_ranking=exp_data['tools_aesthetic_score_ranking'],
                tools_vlm_score_ranking=exp_data['tools_vlm_score_ranking'],
                tools_avg_score_ranking=exp_data['tools_avg_score_ranking'],
                positive_experience=exp_data['positive_experience'],
                negative_experience=exp_data['negative_experience'],
                user_preference_experience=exp_data['user_preference_experience'],
                image_degradation_content=exp_data['image_degradation_content'],
                timestamp=exp_data['timestamp'],
            )
            self.add_experience(experience)
        
        logger.info(f"Experiences loaded from {filepath}")

"""
class ExecutorExperiencePool(BaseExperiencePool):
    def add_experience(self, abilities, experience: ExecutorExperience) -> None:

        task_type = experience.task_type
        # ensure existence of experience embedding(task_major_problem + task_progress_text + task_description)
        if not experience.embedding:
            experience.embedding = self.compute_embedding(experience.task_major_problem + experience.task_progress_text + experience.task_description)
        
        if experience.success == False:
            self.failure_experiences[task_type].append(experience)
            return 0
        elif experience.success == True:
            self.experiences[task_type].append(experience)

            retrieval_experience_num = 0
            for key, value in self.retrieval_experiences.items():
                for exp in value:
                    retrieval_experience_num += 1
            
            if retrieval_experience_num >= self.capacity and self.capacity >= 0:
                self._smart_eviction(abilities, task_type, experience)
            else:
                self.retrieval_experiences[task_type].append(experience)


    def _smart_eviction(self, abilities, task_type: str, new_experience: ExecutorExperience) -> None:
        "Smart eviction strategy, let the LLM Agent decide which trajectory to remove"
        current_time = time.time()

        # Generate input for the agent: the new experience and the existing experiences in the pool
        trajectory_info = []
        for existed_task_type, experiences in self.retrieval_experiences.items():
            for exp in experiences:
                # Add relevant information, including success rate and ability score
                temp_ability_dict = {}
                for ability_name in task_to_ability_map[exp.task_type]:
                    temp_ability_dict[ability_name] = abilities[ability_name]
                trajectory_info.append({
                    'task_id': exp.task_id,
                    'task_type': exp.task_type,
                    'success': exp.success,
                    'major_problem': exp.task_major_problem,    
                    'progress_text': exp.task_progress_text,   
                    'task_description': exp.task_description, 
                    'progress_text': exp.task_progress_text,        
                    'task_thought': exp.task_thought,
                    'result': exp.result,
                    'agent_ability': temp_ability_dict,  # Agent's ability in this task type 
                    'timestamp': exp.timestamp,
                })


        # Current task information (new experience)
        temp_ability_dict = {}
        for ability_name in task_to_ability_map[exp.task_type]:
            temp_ability_dict[ability_name] = abilities[ability_name]
        new_task_info = {
            'task_id': new_experience.task_id,
            'task_type': new_experience.task_type,
            'success': new_experience.success,
            'major_problem': new_experience.task_major_problem,     
            'progress_text': new_experience.task_progress_text,  
            'task_description': new_experience.task_description,  
            'progress_text': new_experience.task_progress_text,            
            'task_thought': new_experience.task_thought,
            'result': new_experience.result,
            'agent_ability': temp_ability_dict,  # Agent's ability in this task type 4
            'timestamp': new_experience.timestamp,
        }

        # Prepare the prompt for LLM Agent decision-making
        agent_input = self.prepare_agent_input(trajectory_info, new_task_info, abilities)

        # Generate the decision through LLM: return the task_id of the trajectory to remove
        trajectory_to_remove = self.make_agent_decision(agent_input)

        # Evict the selected trajectory based on agent's decision
        self.remove_selected_trajectory(task_type, trajectory_to_remove, new_experience)

    def prepare_agent_input(self, trajectory_info: List[Dict], new_task_info: Dict, abilities) -> str:
        "Prepare the input prompt for the LLM agent"
        input_str = "Here is the list of historical trajectories (task experiences):\n"
        for traj in trajectory_info:
            input_str += f"Task ID: {traj['task_id']}, Task Type: {traj['task_type']}, Agent's Ability in this task: {traj['agent_ability']}, Major Problem: {traj['major_problem']}, Progress Text: {traj['progress_text']}\n"

        input_str += "\nCurrent Task Information:\n"
        input_str += f"Task ID: {new_task_info['task_id']}, Task Type: {new_task_info['task_type']}, Major Problem: {new_task_info['major_problem']}, Progress Text: {new_task_info['progress_text']}\n"
        # abilities = str(abilities)
        # input_str += f"Your overall abilities are {abilities}"
        input_str += "\nPlease decide which trajectory would be most beneficial for improving your abilities. Choose the least valuable trajectory (either the new one or an existing one) to remove, and return the Task ID of the trajectory to remove.\n"

        input_str += "\nReturn the Task ID of the trajectory to evict.\n"

        return input_str

    def make_agent_decision(self, agent_input: str) -> str:
        "Generate decision based on the agent's evaluation using LLM"
        # Feed input to LLM model (like encode_model) for decision-making
        decision_embedding = encode_model.encode(agent_input)
        
        # Interpret the output decision from LLM: which task ID to evict
        trajectory_to_remove = self.interpret_decision(decision_embedding)
        
        return trajectory_to_remove

    def interpret_decision(self, decision_embedding: np.ndarray) -> str:
        "Interpret the decision from the LLM model output"
        # Assuming the model outputs an embedding that corresponds to the task ID of the trajectory to remove
        # For example, the model might output a high score for the most likely task ID to evict
        task_id_to_remove = np.argmax(decision_embedding)  # Assuming the model generates a vector for task IDs
        
        return str(task_id_to_remove)

    def remove_selected_trajectory(self, task_type: str, task_id_to_remove: str, new_experience: ExpPool2Experience) -> None:
        "Remove the selected trajectory from the experience pool based on Task ID"
        # Check if the trajectory to remove is the new experience
        if task_id_to_remove == new_experience.task_id:
            # If the new experience is chosen to be removed, don't add it
            print(f"Evicting new experience with Task ID: {task_id_to_remove}")
        else:
            # Remove the specified experience by Task ID
            for exp in self.retrieval_experiences[task_type]:
                if exp.task_id == task_id_to_remove:
                    self.retrieval_experiences[task_type].remove(exp)
                    print(f"Evicting existing experience with Task ID: {task_id_to_remove}")
                    break
    

    def calculate_similarity(self, exp1: ExecutorExperience, exp2: ExecutorExperience) -> float:
        "Calculate similarity between two execution experiences"
        # Vector similarity
        A_embedding = encode_model.encode(exp1.task_major_problem + exp1.task_progress_text + exp1.task_description)
        B_embedding = encode_model.encode(exp2.task_major_problem + exp2.task_progress_text + exp2.task_description)

        cos_similarity = calculate_cos_similarity_A_and_B(A_embedding, B_embedding)

        # Execution time similarity
        time_diff = abs(exp1.execution_time - exp2.execution_time)
        time_similarity = 1.0 / (1 + time_diff / 60)  # Normalized to minutes
        return 0.7 * cos_similarity + 0.3 * time_similarity


    def get_relevant_experiences(self, task, success_only=True, top_k=0, threshold=0.7):
        "Get relevant execution experiences, based on task context text, add threshold check"
        query_embedding = encode_model.encode(task.major_problem + task.progress_text + task.description).reshape(1, -1)

        all_texts = []
        all_efficiency_bonus = []
        all_experiences = []

        for experiences in self.retrieval_experiences.values():
            for exp in experiences:
                if success_only and not exp.success:
                    continue
                all_texts.append(exp.task_major_problem + exp.task_progress_text + exp.task_description)
                all_efficiency_bonus.append(1.0 / (1 + exp.execution_time / 60))
                all_experiences.append(exp)

        if len(all_experiences) == 0:
            return []
        logger.info(f"len(all_experiences) {task.task_id}: {len(all_experiences)}")
        # 获取所有文本的嵌入
        all_texts_embeddings = encode_model.encode(all_texts, batch_size=512).reshape(-1, 1024)
        cos_similarity = calculate_cos_similarity_A_and_Batch_B(query_embedding, all_texts_embeddings)
        all_efficiency_bonus = np.array(all_efficiency_bonus)
        logger.info(f"top_k: {top_k}")
        logger.info(f"executor cos_similarity: {cos_similarity}")
        logger.info(f"executor all_efficiency_bonus: {all_efficiency_bonus}")

        # Calculate weighted scores
        scores = cos_similarity * (1 + 0.2 * all_efficiency_bonus)
        logger.info(f"executor scores: {scores}")
        # If score is an array, convert to scalar for comparison
        valid_experiences = []
        for score, exp in zip(scores[0], all_experiences):
            if score >= threshold:
                valid_experiences.append((score, exp))
        # If the number of valid experiences is less than top_k, return all valid experiences
        logger.info(f"executor len(valid_experiences): {len(valid_experiences)}")
        if len(valid_experiences) < top_k:
            return [exp for _, exp in valid_experiences]

        # Sort by score
        sorted_scores = sorted(valid_experiences, key=lambda x: x[0], reverse=True)

        # Return top_k experiences
        return [exp for _, exp in sorted_scores[:top_k]]

  
    def get_relevant_experiences_by_thought(self, task, success_only=True, top_k=0, threshold=0.7):
        "Get relevant execution experiences based on the Thought of the experience, add threshold check"
        query_embedding = encode_model.encode(task.thought).reshape(1,-1)

        all_thoughts = []
        all_efficiency_bonus = []
        all_experiences = []

        for experiences in self.retrieval_experiences.values():
            for exp in experiences:
                if success_only and not exp.success:
                    continue
                all_thoughts.append(exp.task_thought)
                all_efficiency_bonus.append(1.0 / (1 + exp.execution_time / 60))
                all_experiences.append(exp)

        if len(all_experiences) == 0:
            return []

        all_thoughts_embeddings = encode_model.encode(all_thoughts, batch_size=512).reshape(-1, 1024)
        cos_similarity = calculate_cos_similarity_A_and_Batch_B(query_embedding, all_thoughts_embeddings)
        all_efficiency_bonus = np.array(all_efficiency_bonus)

        scores = cos_similarity * (1 + 0.2 * all_efficiency_bonus)
        valid_experiences = []
        for score, exp in zip(scores[0], all_experiences):
            if score >= threshold:
                valid_experiences.append((score, exp))
        
        logger.info(f"top_k: {top_k}")
        logger.info(f"executor get_relevant_experiences_by_thought len(valid_experiences): {len(valid_experiences)}")

        # If the number of valid experiences is less than top_k, return all valid experiences
        if len(valid_experiences) < top_k:
            experiences = [exp for _, exp in valid_experiences]
            logger.info(f"executor get_relevant_experiences_by_thought len(experiences): {len(experiences)}")
            return experiences

        # Sort by score
        sorted_scores = sorted(valid_experiences, key=lambda x: x[0], reverse=True)

        # Return top_k experiences
        experiences = [exp for _, exp in sorted_scores[:top_k]]
        logger.info(f"executor get_relevant_experiences_by_thought len(experiences): {len(experiences)}")
        return experiences


    def set_experience_success_state(self, task_id, task_type, success):
        for index in range(len(self.retrieval_experiences[task_type])):
            if self.retrieval_experiences[task_type][index].task_id == task_id:
                self.retrieval_experiences[task_type][index].success = success
                
                

    def get_all_experiences(self):

        return_list = []

        for task_type, experiences in self.retrieval_experiences.items():            
            for experience in experiences:
                temp_dict = {
                    "task_id" : experience.task_id,
                    "task_type" : experience.task_type,
                    "task_major_problem" : experience.task_major_problem,
                    "task_description" : experience.task_description,
                    "task_progress_text" : experience.task_progress_text,
                    "task_thought" : experience.task_thought,
                    "result" : experience.result,
                    "execution_time" : experience.execution_time,
                    "success" : experience.success,
                    "timestamp" : experience.timestamp,
                }
                return_list.append(temp_dict)

        return return_list
"""


if __name__ == "__main__":
    exp_pool = RouterExperiencePool(capacity=2)

    # new_exp0 = ExpPool2Experience(
    #     task_id=1,
    #     degradation_type="blur",
    #     severity_level="medium",
    #     tools_name=["toolA", "toolB", "toolC", "toolD"],
    #     tools_ranking=[1, 2, 3, 4],
    #     task_solution=["Use toolA", "Use toolB.", "Use toolC.", "Use toolD."],
    #     task_solution_confidence=[0.9, 0.8, 0.7, 0.6],
    #     task_solution_ranking=[1, 2, 3, 4],
    #     detail_reflection="This task was moderately difficult due to the blur.",
    #     brief_reflection="Handled medium blur effectively."
    # )
    # exp_pool.add_experience(experience=new_exp0)

    # new_exp1 = ExpPool2Experience(
    #     task_id=2,
    #     degradation_type="blur",
    #     severity_level="medium",
    #     tools_name=["toolB", "toolA", "toolC", "toolD"],
    #     tools_ranking=[2, 1, 3, 4],
    #     task_solution=["Use toolB", "Use toolA.", "Use toolC.", "Use toolD."],
    #     task_solution_confidence=[0.9, 0.8, 0.7, 0.6],
    #     task_solution_ranking=[2, 1, 3, 4],
    #     detail_reflection="This task was moderately difficult due to the blur.",
    #     brief_reflection="Handled medium blur effectively."
    # )
    # exp_pool.add_experience(experience=new_exp1)

    # new_exp2 = ExpPool2Experience(
    #     task_id=3,
    #     degradation_type="blur",
    #     severity_level="medium",
    #     tools_name=["toolC", "toolA", "toolB", "toolD"],
    #     tools_ranking=[3, 1, 2, 4],
    #     task_solution=["Use toolA", "Use toolC.", "Use toolB.", "Use toolD."],
    #     task_solution_confidence=[0.9, 0.8, 0.7, 0.6],
    #     task_solution_ranking=[1, 3, 2, 4],
    #     detail_reflection="This task was moderately difficult due to the blur.",
    #     brief_reflection="Handled medium blur effectively."
    # )
    # exp_pool.add_experience(experience=new_exp2)

    # new_exp3 = ExpPool2Experience(
    #     task_id=4,
    #     degradation_type="blur",
    #     severity_level="medium",
    #     tools_name=["toolD", "toolA", "toolB", "toolC"],
    #     tools_ranking=[4, 1, 2, 3],
    #     task_solution=["Use toolA", "Use toolB.", "Use toolC.", "Use toolD."],
    #     task_solution_confidence=[0.9, 0.8, 0.7, 0.6],
    #     task_solution_ranking=[1, 2, 3, 4],
    #     detail_reflection="This task was moderately difficult due to the blur.",
    #     brief_reflection="Handled medium blur effectively."
    # )
    # exp_pool.add_experience(experience=new_exp3)

    # exp_pool.write_experiences_to_file("/data/zkl/AgenticIR/exploration_self_evolve/output/experiences.json")
    exp_pool.read_experiences_from_file("/data/zkl/AgenticIR/exploration_self_evolve/experience_pool/experiences.json")
    exp = exp_pool.get_relevant_experiences(task="The image is a bit blurry, please help me to choose the best tool to enhance the image quality.", degradation_type="blur", severity_level="low", top_k=1, threshold=0.5, success_only=False)
    print(exp)