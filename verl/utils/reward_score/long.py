import re
from typing import Dict, Tuple, Optional

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
    """Parses model's answer text to extract option letter.
    
    Args:
        response: Text extracted from \boxed{}
        
    Returns:
        Option letter (A/B/C/D), or None if not found
    """
    response = response.replace('*', '').strip()
    
    # Try to match option in various formats
    # Format 1: (A), (B), (C), (D)
    match = re.search(r'\(([A-D])\)', response, re.IGNORECASE)
    if match:
        return match.group(1)
    
    # Format 2: Just A, B, C, D (single letter)
    match = re.search(r'^([A-D])$', response, re.IGNORECASE)
    if match:
        return match.group(1)
    
    # Format 3: Letter at the beginning or end
    match = re.search(r'\b([A-D])\b', response, re.IGNORECASE)
    if match:
        return match.group(1)
    
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
    计算选择题任务的分数 - 统一VERL接口

    Args:
        solution_str (str): 模型预测结果
        ground_truth (str): 标准答案 (A/B/C/D)
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
        answer_score = 0
        if answer_text:
            pred_status = parse_model_answer(answer_text)
            gt_status = ground_truth  # 直接用，已经是干净的
            
            if pred_status:
                pred_status = pred_status.lower()
                gt_status = gt_status.lower()
                
                print(f"\n[Content Validation]")
                print(f"  Expected: {gt_status}")
                print(f"  Predicted: {pred_status}")

                if pred_status == gt_status:
                    answer_score = 1.0
                    print("  Content validation: FULL MATCH")
                else:
                    answer_score = 0.0
                    print("  Content validation: MISMATCH")
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
