import os
import re
from typing import Dict, Tuple, Optional
import math
from sympy import Rational
import numpy as np
import sys
import re
import string
from collections import Counter, defaultdict
import pickle
from pathlib import Path
import jsonlines
import json
from tqdm import tqdm

def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall

def sub_em(prediction, ground_truth):
    ground_truth = normalize_answer(ground_truth)
    prediction = normalize_answer(prediction) 
    return (ground_truth in prediction) or (prediction in ground_truth)

def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def update_answer(metrics, prediction, gold):
    em = exact_match_score(prediction, gold)
    subem = sub_em(prediction, gold)

    f1, prec, recall = f1_score(prediction, gold)
    metrics['sub_em'] += subem
    metrics['em'] += float(em)
    metrics['f1'] += f1
    metrics['prec'] += prec
    metrics['recall'] += recall
    metrics['total_num'] += 1
    return em, prec, recall

def calc_metrics(predictions, goldens):
    assert len(predictions) == len(goldens)
    metrics = {'f1': 0, 'prec': 0, 'recall': 0, 'em': 0, 'sub_em': 0, 'total_num': 0}
    for pred, gold in zip(predictions, goldens):
        update_answer(metrics, pred, gold)
    for k, _ in metrics.items():
        if k == 'total_num':
            continue
        metrics[k] = round((metrics[k]/metrics['total_num']), 2)
    return metrics

def extract_solution(solution_str: str) -> Tuple[Optional[str], str]:
    """Extracts the final answer from the model's response string.
    
    Args:
        solution_str: Raw response string from the language model
        
    Returns:
        Tuple containing (extracted_answer, processed_string)
    """
    
    # Extract final answer using \boxed{} format
    boxed_pattern = re.compile(r'\\boxed\{([^}]*)\}', re.IGNORECASE)
    matches = boxed_pattern.findall(solution_str)
    
    if matches:
        # 返回最后一个 \boxed{} 中的内容
        final_answer = matches[-1].strip()
        return final_answer, solution_str
    else:
        print("[Error] No \\boxed{} found in response")
        return None, solution_str

def parse_model_answer(response: str) -> Optional[str]:
    """Parses the final answer from the model's response text.
    
    Args:
        response: Text extracted from the model's response
        
    Returns:
        The final answer (string), or None if not found
    """
    # Remove any asterisks or other unwanted characters
    response = response.replace('*', '').strip()
    
    # Return the cleaned answer directly (already extracted from \boxed{})
    if response:
        return response
    else:
        return None
    
def validate_response_structure(processed_str: str) -> bool:
    """Performs comprehensive validation of response structure.
    
    Args:
        processed_str: Processed response string from the model
        
    Returns:
        Boolean indicating whether all formatting requirements are met
    """
    print("\n[Structure Validation]")
    validation_passed = True

    # Check required tags
    tags = {
        'think_start': ('<think>', 1),
        'think_end': ('</think>', 1)
    }

    positions = {}
    for tag_name, (tag_str, expected_count) in tags.items():
        count = processed_str.count(tag_str)
        positions[tag_name] = pos = processed_str.find(tag_str)
        
        print(f"  {tag_str}: count={count}, position={pos}")
        
        if count != expected_count:
            print(f"  [Error] {tag_str} appears {count} times (expected {expected_count})")
            validation_passed = False

    # Verify tag order
    if (positions['think_start'] > positions['think_end']):
        print("  [Error] Incorrect tag order: Expected <think>...</think>")
        validation_passed = False
    else:
        print("  Tag sequence validation passed")

    return validation_passed

def compute_score(solution_str: str, ground_truth: str, task_name: str = None, extra_info: Dict = None) -> Dict:
    """
    计算QA任务的分数 - 统一VERL接口

    Args:
        solution_str (str): 模型预测结果
        ground_truth (str): 标准答案
        task_name (str): 任务名称
        extra_info (Dict): 额外信息

    Returns:
        Dict: 包含score, acc, format的字典
    """
    print("\n" + "="*80)
    print(" Processing New Sample ".center(80, '='))
    
    try:
        # Extract model answer
        answer_text, processed_str = extract_solution(solution_str)
        print(f"\n[Model Response]\n{processed_str}")
        
        # Check format
        format_valid = 1 if answer_text is not None else 0
        
        # Validate answer content
        answer_score = 0.0
        if answer_text:
            pred_status = parse_model_answer(answer_text)
            gt_status = ground_truth  # 直接用，已经是干净的字符串
            
            if pred_status:
                print(f"\n[Content Validation]")
                print(f"  Expected: {gt_status}")
                print(f"  Predicted: {pred_status}")
                metrics = calc_metrics([pred_status], [gt_status])
                metric = metrics['sub_em']
                answer_score = float(metric)
                print(f"  Answer Score: {answer_score}")
            else:
                answer_score = 0.0
                print("Fail to parse answer")
        else:
            print("\n[Content Validation] Skipped due to format errors or missing answer")
        
        print("\n" + "-"*80)
        print(f" Final Score ".center(80, '-'))
        print(f"  Score: {answer_score}")
        print(f"  Format: {format_valid}")
        print("="*80 + "\n")
        
        return {
            "score": answer_score,
            "acc": answer_score,
            "format": format_valid
        }
    
    except Exception as e:
        print(f"Exception: {e}")
        return {
            "score": 0.0,
            "acc": 0.0,
            "format": 0
        }
