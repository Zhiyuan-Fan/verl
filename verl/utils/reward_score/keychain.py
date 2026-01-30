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

import re
from typing import Dict, Optional


class KeyChainEvaluator:
    """KeyChain 数据集评估器 - 实现 Two-way Substring Exact Match"""

    def __init__(self):
        # 用于提取 \boxed{} 中答案的正则表达式
        self.boxed_pattern = re.compile(r'\\boxed\{([^}]*)\}', re.IGNORECASE)

    def extract_boxed_answer(self, text: str) -> Optional[str]:
        """
        从文本中提取 \\boxed{} 内的答案

        Args:
            text: 模型输出文本

        Returns:
            提取的答案，如果没找到返回None
        """
        matches = self.boxed_pattern.findall(text)
        if matches:
            # 返回最后一个 \boxed{} 中的内容
            return matches[-1].strip()
        return None

    def normalize_text(self, text: str) -> str:
        """
        标准化文本 - 转小写，去除多余空格

        Args:
            text: 原始文本

        Returns:
            标准化后的文本
        """
        if text is None:
            return ""
        return ' '.join(text.lower().strip().split())

    def two_way_substring_match(self, predicted: str, ground_truth: str) -> bool:
        """
        Two-way Substring Exact Match
        根据LoongRL论文: ri = 1 if (a ⊆ yans) or (yans ⊆ a), else 0

        Args:
            predicted: 预测答案
            ground_truth: 标准答案

        Returns:
            是否匹配
        """
        predicted_norm = self.normalize_text(predicted)
        ground_truth_norm = self.normalize_text(ground_truth)

        if not predicted_norm or not ground_truth_norm:
            return False

        # Two-way substring match:
        # 1. ground_truth 是 predicted 的子串
        # 2. predicted 是 ground_truth 的子串
        return (ground_truth_norm in predicted_norm) or (predicted_norm in ground_truth_norm)

    def calculate_reward(self, solution_str: str, ground_truth: str) -> float:
        """
        计算KeyChain任务的奖励

        Args:
            solution_str: 模型输出
            ground_truth: 标准答案

        Returns:
            奖励值 (0.0 或 1.0)
        """
        # 1. 提取 \boxed{} 中的答案
        extracted_answer = self.extract_boxed_answer(solution_str)

        if extracted_answer is None:
            return 0.0

        # 2. 执行 Two-way Substring Exact Match
        is_correct = self.two_way_substring_match(extracted_answer, ground_truth)

        reward = 1.0 if is_correct else 0.0
        return reward


# 全局评估器实例
_evaluator = None


def get_evaluator():
    """获取全局评估器实例"""
    global _evaluator
    if _evaluator is None:
        _evaluator = KeyChainEvaluator()
    return _evaluator


def compute_score(solution_str: str, ground_truth: str, task_name: str = None, extra_info: Dict = None) -> Dict:
    """
    计算KeyChain任务的分数 - 统一VERL接口

    Args:
        solution_str (str): 模型预测结果
        ground_truth (str): 标准答案
        task_name (str): 任务名称 (应该包含 "keychain")
        extra_info (Dict): 额外信息

    Returns:
        Dict: 包含score, acc, format的字典
    """
    evaluator = get_evaluator()

    try:
        # 计算奖励
        reward = evaluator.calculate_reward(solution_str, ground_truth)

        return {
            "score": reward,
            "acc": reward,
            "format": 1 if evaluator.extract_boxed_answer(solution_str) is not None else 0
        }

    except Exception as e:
        return {
            "score": 0.0,
            "acc": 0.0,
            "format": 0
        }
