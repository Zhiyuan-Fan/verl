import os
import re
from typing import Dict, Tuple, Optional
import math
from sympy import Rational
import numpy as np

def round_up_to_decimal(number, decimals):
    factor = 10 ** decimals
    return math.ceil(number * factor) / factor

def is_number(string):
    pattern = r'^[-+]?(\d{1,3}(,\d{3})*|(\d+))(\.\d+)?$'
    match = re.match(pattern, string)
    return bool(match)

def is_scientific_number(string):
    pattern = r'^[-+]?\d+(\.\d+)?e[-]?\d+$'
    match = re.match(pattern, string)
    return bool(match)

def normalize(prediction: str):
    # Preprocessing the string [Stage 1]
    prediction = prediction.strip()
    prediction = prediction.rstrip('.')
    if not isinstance(prediction, str):
        prediction = str(prediction) if prediction is not None else '0'

    for money in ["£", "€", "¥", "million", "billion", "thousand", "US", "USD", "RMB"]:
        prediction = prediction.replace(money, '')
        
    # Replace special tokens
    if '=' in prediction:
        prediction = prediction.split('=')[-1].strip()
    if '≈' in prediction:
        prediction = prediction.split('≈')[-1].strip()
    if '`' in prediction:
        prediction = prediction.replace('`', '')
    if '%' in prediction:
        prediction = prediction.replace('%', '')
    if '$' in prediction:
        prediction = prediction.replace('$', '')
    if '°' in prediction:
        prediction = prediction.replace('°', '')

    # Detect the boolean keyword in the generation
    if prediction in ['true', 'yes', 'false', 'no']:
        if prediction == 'true' or prediction == 'yes':
            prediction = 'True'
        else:
            prediction = 'False'
    if 'True' in prediction or 'False' in prediction:
        prediction = 'True' if 'True' in prediction else 'False'

    # Detect the approximation keyword
    if 'approximately' in prediction:
        prediction = prediction.replace('approximately', '').strip()
    if ' or ' in prediction:
        prediction = prediction.split(' or ')[0]

    # Drop the units before and after the number
    if re.match(r'[-+]?(?:[\d,]*\.*\d+) [^0-9 ]+$', prediction):
        prediction = re.search(r'([-+]?(?:[\d,]*\.*\d+)) [^0-9 ]+$', prediction).group(1)
    if re.match(r'[^0-9 ]+ [-+]?(?:[\d,]*\.*\d+)$', prediction):
        prediction = re.search(r'[^0-9 ]+ ([-+]?(?:[\d,]*\.*\d+))$', prediction).group(1)
    if re.match(r'[-+]?(?:[\d,]*\.*\d+)[^\d]{1,2}$', prediction):
        prediction = re.search(r'([-+]?(?:[\d,]*\.*\d+))[^\d]{1,2}$', prediction).group(1)
    if re.match(r'[^-+\d]{1,2}(?:[\d,]*\.*\d+)$', prediction):
        prediction = re.search(r'[^-+\d]{1,2}((?:[\d,]*\.*\d+))$', prediction).group(1)

    # Preprocessing the number [Stage 1]
    if '10^' in prediction:
        prediction = re.sub(r'10\^(-?\d+)', r'math.pow(10, \1)', prediction)
    if ' x ' in prediction:
        prediction = prediction.replace(' x ', '*')
    if ' × ' in prediction:
        prediction = prediction.replace(' × ', '*')
    if is_number(prediction):
        prediction = prediction.replace(',', '')

    # Preprocessing the option [Stage 3]
    if '(a)' in prediction or '(b)' in prediction or '(c)' in prediction or '(d)' in prediction:
        prediction = '"' + re.search(r'\([a-d]\)', prediction).group(0) + '"'

    # If the prediction is empty, use dummy '0'
    if not prediction:
        prediction = '0'

    # Converting the string answer to a number/list/bool/option
    try:
        prediction = eval(prediction)
    except Exception:
        # TO CHECK
        prediction = 0 

    # Performing common type conversion
    if isinstance(prediction, (set, tuple)):
        prediction = list(prediction)
        if isinstance(prediction[0], complex):
            prediction = [tmp.real for tmp in prediction]
        elif isinstance(prediction[0], Rational):
            prediction = [float(tmp) for tmp in prediction]
    elif isinstance(prediction, np.ndarray):
        prediction = prediction.tolist()
    else:
        if isinstance(prediction, complex):
            prediction = prediction.real
        elif isinstance(prediction, Rational):
            prediction = float(prediction)

    return prediction

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
    response = response.replace('*', '')
    response = response.strip()
    
    # Return the cleaned answer
    if response:
        return response
    else:
        return None

def within_eps(pred: float, gt: float):
    eps = abs(gt) * 0.0015
    if pred >= gt - eps and pred <= gt + eps:
        return True
    else:
        return False

def compare_two_numbers(p, gt):
    if isinstance(p, int) or isinstance(p, float):
        pass
    elif isinstance(p, list) or isinstance(p, bool) or isinstance(p, str):
        return False
    elif isinstance(p, tuple) or isinstance(p, complex) or isinstance(p, dict):
        return False
    else:
        raise ValueError(p)

    try:
        v1, v2 = max(abs(gt), abs(p)), min(abs(gt), abs(p))
        if (v1 !=0 and v2 != 0) and int(math.log10(v1) - math.log10(v2)) == (math.log10(v1) - math.log10(v2)):
            return True

        if v2 <= v1 / 50 and within_eps(pred=v2*100, gt=v1):
            return True
        elif v2 <= v1 / 500 and within_eps(pred=v2*1000, gt=v1):
            return True
        elif v2 <= v1 / 50000 and within_eps(pred=v2*100000, gt=v1):
            return True

        if round_up_to_decimal(v1, 3) == round_up_to_decimal(v2, 3):
            return True

        return within_eps(pred=p, gt=gt)
    except OverflowError:
        return False

def get_acc(prediction, gt, cot=True):
    print(f"get_acc({prediction}, {gt})")
    
    if cot:
        prediction = normalize(prediction)
        gt = normalize(gt)
    else:
        prediction = float(prediction)
    
    print(f"after normalize pre = {prediction}")
    print(f"after normalize gt = {gt}")
    
    answer_type = type(gt).__name__
    print(f"answer_type::{answer_type}")
    assert answer_type in ["int", "float", "float64", "bool"], answer_type
    
    if isinstance(prediction, (str, int, float, bool)) or isinstance(prediction, list):
        # Comparing prediction against the reference
        if answer_type in ['bool']:
            acc = int(prediction == gt)
        elif answer_type == 'int':
            acc = int(compare_two_numbers(prediction, gt))
        elif answer_type == 'float' or answer_type == 'float64':
            acc = int(compare_two_numbers(prediction, gt))
        else:
            acc = 0
    else:
        acc = 0
        print("Error: ", prediction, type(prediction))
    
    return acc

def compute_score(solution_str: str, ground_truth: str, task_name: str = None, extra_info: Dict = None) -> Dict:
    """
    计算数学题任务的分数 - 统一VERL接口

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
                acc = get_acc(pred_status, gt_status)
                answer_score = float(acc)
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
