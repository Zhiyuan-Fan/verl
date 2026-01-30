import json
import re
from typing import Dict, List, Optional, Union, Any

try:
    import numpy as np
except Exception:
    np = None


class BookRULEREvaluator:
    """
    Book RULER 数据集评估器
    - Multi-key: single value, two-way substring match (0/1)
    - Multi-value: list of values, set-based partial credit ([0,1])
    """

    def __init__(self):
        self.boxed_pattern = re.compile(r'\\boxed\{([^}]*)\}', re.IGNORECASE)

    # ---------- 新增：类型归一化工具 ----------

    def _is_numpy_array(self, x: Any) -> bool:
        return (np is not None) and isinstance(x, np.ndarray)

    def _to_py_obj(self, x: Any) -> Any:
        """把 numpy/arrow/其他奇怪对象尽量转成 Python 原生类型，不保证语义，仅保证不炸。"""
        if x is None:
            return None
        if self._is_numpy_array(x):
            return x.tolist()
        return x

    def _maybe_parse_json_list(self, s: str) -> Optional[List[Any]]:
        """若 s 是一个 JSON list 字符串则解析，否则返回 None。"""
        if not isinstance(s, str):
            return None
        t = s.strip()
        if not (t.startswith("[") and t.endswith("]")):
            return None
        try:
            obj = json.loads(t)
            if isinstance(obj, list):
                return obj
        except Exception:
            return None
        return None

    def _normalize_ground_truth(self, ground_truth: Any, variant: Optional[str]) -> Union[str, List[str]]:
        """
        把 ground_truth 统一成:
        - multi_key: str
        - multi_value: List[str]
        允许输入为 str/list/np.ndarray，以及 list/ndarray 内部元素非 str 的情况。
        """
        gt = self._to_py_obj(ground_truth)

        # 如果没有 variant，就用类型粗略推断（但要兼容 ndarray）
        if variant is None:
            if isinstance(gt, list):
                variant = "multi_value"
            else:
                variant = "multi_key"

        is_multi_key = ("multi_key" in variant) or (variant == "multi_key")

        # 1) Multi-key：强制成 str
        if is_multi_key:
            # 有些数据可能错误给了 list/ndarray，取第一个元素兜底
            if isinstance(gt, list):
                gt = gt[0] if len(gt) > 0 else ""
            return "" if gt is None else str(gt)

        # 2) Multi-value：强制成 List[str]
        # 如果是 JSON 字符串（有人会把 list dump 成字符串存 parquet），这里也支持
        if isinstance(gt, str):
            parsed = self._maybe_parse_json_list(gt)
            if parsed is not None:
                gt = parsed
            else:
                # 不是 JSON list，则当作单元素列表
                gt = [gt]

        gt = self._to_py_obj(gt)
        if not isinstance(gt, list):
            gt = [gt]

        # 元素全部转 str（None -> ""），避免后续 lower()/split() 崩
        out: List[str] = []
        for item in gt:
            if item is None:
                out.append("")
            else:
                out.append(str(self._to_py_obj(item)))
        return out

    def _normalize_solution_str(self, solution_str: Any) -> str:
        if solution_str is None:
            return ""
        solution_str = self._to_py_obj(solution_str)
        # 兜底：如果是 list/ndarray，拼起来
        if isinstance(solution_str, list):
            return " ".join(str(self._to_py_obj(x)) for x in solution_str)
        return str(solution_str)

    # ---------- 原有逻辑（小改：永不抛异常） ----------

    def extract_boxed_answer(self, text: str) -> Optional[str]:
        matches = self.boxed_pattern.findall(text or "")
        if matches:
            return matches[-1].strip()
        return None

    def normalize_text(self, text: Any) -> str:
        """
        标准化文本 - 转小写，去除多余空格
        这里必须保证任何类型都不炸。
        """
        if text is None:
            return ""

        text = self._to_py_obj(text)
        # 如果传进来是 list/ndarray，拼成字符串（主要用于异常输入兜底）
        if isinstance(text, list):
            text = " ".join(str(self._to_py_obj(x)) for x in text)

        try:
            s = str(text)
            return " ".join(s.lower().strip().split())
        except Exception:
            # 彻底兜底：不要 raise，直接返回空串，避免丢样本
            return ""

    def two_way_substring_match(self, predicted: str, ground_truth: str) -> bool:
        predicted_norm = self.normalize_text(predicted)
        ground_truth_norm = self.normalize_text(ground_truth)
        if not predicted_norm or not ground_truth_norm:
            return False
        return (ground_truth_norm in predicted_norm) or (predicted_norm in ground_truth_norm)

    def parse_list_answer(self, text: str) -> List[str]:
        if not text:
            return []
        items = [item.strip() for item in str(text).split(",")]
        normalized_items = [self.normalize_text(item) for item in items if item.strip()]
        return normalized_items

    def evaluate_multi_value_with_partial_credit(
        self,
        predicted_list: List[str],
        ground_truth_list: List[str]
    ) -> float:
        if not ground_truth_list or not predicted_list:
            return 0.0

        recall_hits = 0
        for gt_value in ground_truth_list:
            for pred_value in predicted_list:
                if self.two_way_substring_match(pred_value, gt_value):
                    recall_hits += 1
                    break

        precision_hits = 0
        for pred_value in predicted_list:
            for gt_value in ground_truth_list:
                if self.two_way_substring_match(pred_value, gt_value):
                    precision_hits += 1
                    break

        max_possible = 2 * len(ground_truth_list)
        score = (recall_hits + precision_hits) / max_possible
        return min(max(score, 0.0), 1.0)

    def calculate_reward(
        self,
        solution_str: Any,
        ground_truth: Any,
        extra_info: Dict = None
    ) -> float:
        # 1) 统一 solution_str 类型
        solution_str = self._normalize_solution_str(solution_str)

        # 2) 判定 variant（若缺失则后续 _normalize_ground_truth 会推断）
        variant = None
        if isinstance(extra_info, dict):
            variant = extra_info.get("variant", None)

        # 3) 统一 ground_truth 类型
        gt_norm = self._normalize_ground_truth(ground_truth, variant)

        # 4) 提取 boxed
        extracted_answer = self.extract_boxed_answer(solution_str)
        if extracted_answer is None:
            return 0.0

        is_multi_key = (variant is None and isinstance(gt_norm, str)) or (variant is not None and ("multi_key" in variant or variant == "multi_key"))

        if is_multi_key:
            # Multi-key
            gt_str = gt_norm if isinstance(gt_norm, str) else (gt_norm[0] if gt_norm else "")
            return 1.0 if self.two_way_substring_match(extracted_answer, gt_str) else 0.0
        else:
            # Multi-value
            gt_list = gt_norm if isinstance(gt_norm, list) else [str(gt_norm)]
            predicted_list = self.parse_list_answer(extracted_answer)
            return self.evaluate_multi_value_with_partial_credit(predicted_list, gt_list)


_evaluator = None


def get_evaluator():
    global _evaluator
    if _evaluator is None:
        _evaluator = BookRULEREvaluator()
    return _evaluator


def compute_score(
    solution_str: Any,
    ground_truth: Any,
    task_name: str = None,
    extra_info: Dict = None
) -> Dict:
    """
    返回 dict: {"score": float, "acc": float, "format": 0/1}
    关键：任何异常都要吞掉并返回 0，不能 raise（否则会造成 batch 内 reward 数量不齐）
    """
    evaluator = get_evaluator()
    try:
        reward = evaluator.calculate_reward(solution_str, ground_truth, extra_info)
        has_format = 1 if evaluator.extract_boxed_answer(evaluator._normalize_solution_str(solution_str)) is not None else 0
        return {"score": float(reward), "acc": float(reward), "format": int(has_format)}
    except Exception:
        # 必须兜底，绝不抛出
        return {"score": 0.0, "acc": 0.0, "format": 0}

