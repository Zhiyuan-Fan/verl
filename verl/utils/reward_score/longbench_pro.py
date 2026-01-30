# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Dict, Set, Optional
import pytrec_eval
from itertools import combinations


class LongBenchProEvaluator:
    """LongBench-Pro 评估器"""

    def __init__(self):
        # 任务映射：从带前缀的任务名到原始任务名
        self.task_mapping = {
            "longbench_pro_t1.1_global_cohesive_retrieval": "T1.1 Global Cohesive Retrieval",
            "longbench_pro_t1.2_key_snippet_retrieval": "T1.2 Key-Snippet Retrieval",
            "longbench_pro_t2.1_global_timeline_reconstruction": "T2.1 Global Timeline Reconstruction",
            "longbench_pro_t2.2_local_causal_chain_sorting": "T2.2 Local Causal Chain Sorting",
            "longbench_pro_t3.1_multi_doc_integration_qa": "T3.1 Multi-Doc Integration QA",
            "longbench_pro_t3.2_single_hop_fact_qa": "T3.2 Single-Hop Fact QA",
            "longbench_pro_t4.1_global_coverage_constrained_summary": "T4.1 Global-Coverage Constrained Summary",
            "longbench_pro_t4.2_query_focused_summary": "T4.2 Query-Focused Summary",
            "longbench_pro_t5.1_full_sentence_citation_alignment": "T5.1 Full-Sentence Citation Alignment",
            "longbench_pro_t5.2_key_statement_citation_alignment": "T5.2 Key-Statement Citation Alignment",
            "longbench_pro_t6.1_large_scale_document_clustering": "T6.1 Large-Scale Document Clustering",
            "longbench_pro_t6.2_targeted_subset_cluster_identification": "T6.2 Targeted Subset Cluster Identification",
            "longbench_pro_t6.3_global_frequency_analysis": "T6.3 Global Frequency Analysis",
            "longbench_pro_t7.1_global_conflict_inconsistency_localization": "T7.1 Global Conflict & Inconsistency Localization",
            "longbench_pro_t7.2_targeted_rule_condition_violation_detection": "T7.2 Targeted Rule or Condition Violation Detection",
            "longbench_pro_t7.3_comprehensive_error_anomaly_sweep": "T7.3 Comprehensive Error & Anomaly Sweep",
            "longbench_pro_t8.1_structured_multi_source_consistency_verification": "T8.1 Structured Multi-Source Consistency Verification",
            "longbench_pro_t8.2_single_source_targeted_aggregation": "T8.2 Single-Source Targeted Aggregation",
            "longbench_pro_t8.3_long_context_procedural_state_tracking": "T8.3 Long-Context Procedural State Tracking",
            "longbench_pro_t9.1_dependency_aware_multi_version_impact_analysis": "T9.1 Dependency-Aware Multi-Version Impact Analysis",
            "longbench_pro_t9.2_localized_interface_change_detection": "T9.2 Localized Interface Change Detection",
            "longbench_pro_t10.1_large_scale_in_context_rule_induction": "T10.1 Large-Scale In-Context Rule Induction",
            "longbench_pro_t10.2_targeted_example_based_rule_induction": "T10.2 Targeted Example-Based Rule Induction",
            "longbench_pro_t11.1_long_range_entity_commitment_tracking": "T11.1 Long-Range Entity & Commitment Tracking",
            "longbench_pro_t11.2_short_range_reference_resolution_state_query": "T11.2 Short-Range Reference Resolution & State Query"
        }

        # 任务到指标的映射
        self.task_metric_config = {
            "T1.1 Global Cohesive Retrieval": "NDCG",
            "T1.2 Key-Snippet Retrieval": "NDCG",
            "T2.1 Global Timeline Reconstruction": "Pairwise_Accuracy",
            "T2.2 Local Causal Chain Sorting": "Pairwise_Accuracy",
            "T3.1 Multi-Doc Integration QA": "Accuracy",
            "T3.2 Single-Hop Fact QA": "Accuracy",
            "T4.1 Global-Coverage Constrained Summary": "Summary",
            "T4.2 Query-Focused Summary": "Summary",
            "T5.1 Full-Sentence Citation Alignment": "F1_Score",
            "T5.2 Key-Statement Citation Alignment": "F1_Score",
            "T6.1 Large-Scale Document Clustering": "SubEM",
            "T6.2 Targeted Subset Cluster Identification": "F1_Score",
            "T6.3 Global Frequency Analysis": "Pairwise_Accuracy",
            "T7.1 Global Conflict & Inconsistency Localization": "F1_Score",
            "T7.2 Targeted Rule or Condition Violation Detection": "F1_Score",
            "T7.3 Comprehensive Error & Anomaly Sweep": "F1_Score",
            "T8.1 Structured Multi-Source Consistency Verification": "SubEM",
            "T8.2 Single-Source Targeted Aggregation": "SubEM",
            "T8.3 Long-Context Procedural State Tracking": "SubEM",
            "T9.1 Dependency-Aware Multi-Version Impact Analysis": "F1_Score",
            "T9.2 Localized Interface Change Detection": "F1_Score",
            "T10.1 Large-Scale In-Context Rule Induction": "SubEM",
            "T10.2 Targeted Example-Based Rule Induction": "SubEM",
            "T11.1 Long-Range Entity & Commitment Tracking": "Accuracy",
            "T11.2 Short-Range Reference Resolution & State Query": "Accuracy"
        }

    def get_answer_area(self, text: str) -> str:
        """提取答案区域"""
        if "[Answer]" in text or "[答案]" in text:
            if "[Answer]" in text:
                last_answer_start = text.rfind('[Answer]')
                if last_answer_start != -1:
                    text = text[last_answer_start + 8:]
            else:
                last_answer_start = text.rfind('[答案]')
                if last_answer_start != -1:
                    text = text[last_answer_start + 4:]
        return text.strip()

    def fix_space(self, text: str) -> str:
        return ' '.join(text.split())

    def normalize_answers(self, answers: List[str]) -> List[str]:
        return [self.fix_space(a.lower().strip()) for a in answers]

    def normalize_prediction(self, prediction: str) -> List[str]:
        prediction_area = self.get_answer_area(prediction)
        return [self.fix_space(p.strip()) for p in prediction_area.lower().split("\n")]

    def calculate_accuracy(self, answers: List[str], prediction: str) -> float:
        """准确率计算"""
        answers = self.normalize_answers(answers)
        predictions = self.normalize_prediction(prediction)

        if len(answers) == 0 or len(predictions) == 0:
            return 0.0

        return 1.0 if answers[0] == predictions[0] else 0.0

    def calculate_f1_score(self, answers: List[str], prediction: str) -> float:
        """F1分数计算"""
        answers = self.normalize_answers(answers)
        predictions = self.normalize_prediction(prediction)

        answer_set = set(answers)
        prediction_set = set(predictions)

        common = answer_set & prediction_set
        if len(common) == 0 or len(prediction_set) == 0 or len(answer_set) == 0:
            return 0.0

        precision = len(common) / len(prediction_set)
        recall = len(common) / len(answer_set)

        if precision + recall == 0:
            return 0.0

        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def calculate_subem(self, answers: List[str], prediction: str) -> float:
        """子集精确匹配"""
        answers = self.normalize_answers(answers)
        predictions = self.normalize_prediction(prediction)

        if len(answers) == 0 or len(predictions) == 0:
            return 0.0

        score = 0.0
        for a in answers:
            if a in predictions:
                score += 1.0
        return score / len(answers)

    def calculate_ndcg(self, answers: List[str], prediction: str) -> float:
        """NDCG计算"""
        answers = self.normalize_answers(answers)
        predictions = self.normalize_prediction(prediction)

        if len(answers) == 0 or len(predictions) == 0:
            return 0.0

        k_value = len(answers)

        answers_dict = {
            'query': {a: len(answers) - i for i, a in enumerate(answers)}
        }
        predictions_dict = {
            'query': {p: len(predictions) - i for i, p in enumerate(predictions)}
        }

        ndcg_string = f"ndcg_cut.{k_value}"
        evaluator = pytrec_eval.RelevanceEvaluator(answers_dict, {ndcg_string})
        scores = evaluator.evaluate(predictions_dict)

        ndcg = 0.0
        for query_id in scores.keys():
            ndcg += scores[query_id][f"ndcg_cut_{k_value}"]

        return ndcg / len(scores)

    def calculate_pairwise_accuracy(self, answers: List[str], prediction: str) -> float:
        """配对准确率计算"""
        answers = self.normalize_answers(answers)
        predictions = self.normalize_prediction(prediction)

        if len(answers) == 0 or len(answers) == 1 or len(predictions) == 0 or len(predictions) == 1:
            return 0.0

        n_total = len(predictions) * (len(predictions) - 1) // 2
        prediction_indices = {p: i for i, p in enumerate(predictions)}
        n_correct = 0

        for a, b in combinations(answers, 2):
            if a in prediction_indices and b in prediction_indices:
                if prediction_indices[a] < prediction_indices[b]:
                    n_correct += 1

        return n_correct / n_total


# 全局评估器实例
_evaluator = None


def get_evaluator():
    """获取全局评估器实例"""
    global _evaluator
    if _evaluator is None:
        _evaluator = LongBenchProEvaluator()
    return _evaluator


def compute_score(solution_str: str, ground_truth: List[str], task_name: str = None, extra_info: Dict = None) -> float:
    """
    计算LongBench-Pro任务的分数

    Args:
        solution_str (str): 模型预测结果
        ground_truth (List[str]): 标准答案列表
        task_name (str): 任务名称 (格式: longbench_pro_{task_name})
        extra_info (Dict): 额外信息，可能包含task_name

    Returns:
        float: 0.0到1.0之间的分数

    Raises:
        ValueError: 如果任务名称无效或Summary任务缺少embedding模型
    """
    print("Here: inside longbench_pro.compute_score()...")
    evaluator = get_evaluator()

    # 从extra_info或参数获取任务名称
    # if task_name is None and extra_info is not None:
    #     task_name = extra_info.get('task_name')
    
    ### "extra_info": {
                # "task_name": 
    
    # 这里从extra_info里面获取task name来计算不同的reward score；
    # 外层的data source需要在validation的时候，保证所有相同的data source都是被处理成一致的了；
    task_name = None
    # 只支持从 extra_info 里面获取 task_name
    if extra_info is not None:
        task_name = extra_info.get('task_name')

    if task_name is None:
        raise ValueError("Task name must be provided either as parameter or in extra_info")

    # 映射到原始任务名
    if task_name not in evaluator.task_mapping:
        raise ValueError(f"Unknown task: {task_name}. Available tasks: {list(evaluator.task_mapping.keys())}")

    original_task_name = evaluator.task_mapping[task_name]
    metric_name = evaluator.task_metric_config[original_task_name]

    # Summary任务需要特殊处理，这里返回警告
    if metric_name == "Summary":
        print(f"Warning: Summary metric requires embedding model for task {task_name}. Returning 0.0")
        return {
            "score": 0,
            "acc": 0,
            "format": 0
        }

    try:
        if metric_name == "NDCG":
            print("Here: calculating NDCG...")
            score = evaluator.calculate_ndcg(ground_truth, solution_str)
            print("NDCG calculation Done.")
        elif metric_name == "Pairwise_Accuracy":
            print("Here: calculating Pairwise Accuracy...")
            score = evaluator.calculate_pairwise_accuracy(ground_truth, solution_str)
            print("Pairwise_Accuracy calculation Done.")
        elif metric_name == "Accuracy":
            print("Here: calculating Accuracy...")
            score = evaluator.calculate_accuracy(ground_truth, solution_str)
            print("Accuracy calculation Done.")
        elif metric_name == "F1_Score":
            print("Here: calculating F1 Score...")
            score = evaluator.calculate_f1_score(ground_truth, solution_str)
            print("F1 calculation Done.")
        elif metric_name == "SubEM":
            print("Here: calculating SubEM...")
            score = evaluator.calculate_subem(ground_truth, solution_str)
            print("SubEM calculation Done.")
        else:
            raise ValueError(f"Unknown metric: {metric_name}")
        
        print(f"calculated reward is {score}!!! DONE !!!")

        score = max(0.0, min(1.0, score))
        
        return {
            "score": score,
            "acc": score,
            "format": 0
        }

    except Exception as e:
        print(f"Error calculating metric for {task_name}: {e}")
        return {
            "score": 0,
            "acc": 0,
            "format": 0
        }
