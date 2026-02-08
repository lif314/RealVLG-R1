# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
# ---------------------------------------------------------

import re
import json
import numpy as np
from typing import List, Dict, Any


def format_reward(response: str) -> float:
    """
    Check whether the model output follows this strict format:
    <think> ... </think>
    <answer>(x1, y1), (x2, y2)</answer>
    Supports integers or floats in coordinates.
    """
    re_str = (
        r"<think>.*?</think>\s*"
        r"<answer>\s*\(\s*[\d\.eE+-]+\s*,\s*[\d\.eE+-]+\s*\)\s*,\s*"
        r"\(\s*[\d\.eE+-]+\s*,\s*[\d\.eE+-]+\s*\)\s*</answer>"
    )
    pattern = re.compile(re_str, re.DOTALL)
    return 1.0 if re.fullmatch(pattern, response.strip()) else 0.0


def accuracy_reward(predict: str, ground_truth: str) -> float:
    """
    Compute IoU between predicted bbox and ground-truth bbox.

    predict: model output like "<think>...</think><answer>(x1, y1), (x2, y2)</answer>"
    ground_truth: JSON string, e.g.
        "[x1, y1, x2, y2]"   or   '{"bbox": [x1, y1, x2, y2]}'
    """
    ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
    COORD_PATTERN = re.compile(
        r"\(\s*([\d\.eE+-]+)\s*,\s*([\d\.eE+-]+)\s*\)\s*,\s*\(\s*([\d\.eE+-]+)\s*,\s*([\d\.eE+-]+)\s*\)"
    )

    def parse_bbox(coord_str: str):
        match = COORD_PATTERN.fullmatch(coord_str.strip())
        if not match:
            raise ValueError(f"Invalid bbox format: {coord_str}")
        x1, y1, x2, y2 = list(map(float, match.groups()))
        if (x2 - x1) < 1e-6 or (y2 - y1) < 1e-6:
            raise ValueError(f"Degenerate bbox: {coord_str}")
        return x1, y1, x2, y2

    def bbox_iou(boxA, boxB):
        """Compute IoU between two rectangular bboxes."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        inter_w = max(0, xB - xA)
        inter_h = max(0, yB - yA)
        inter_area = inter_w * inter_h

        boxA_area = max(0, (boxA[2] - boxA[0])) * max(0, (boxA[3] - boxA[1]))
        boxB_area = max(0, (boxB[2] - boxB[0])) * max(0, (boxB[3] - boxB[1]))

        union_area = boxA_area + boxB_area - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    try:
        # Extract <answer> content
        answer_match = ANSWER_PATTERN.search(predict)
        if not answer_match:
            return 0.0
        answer_str = answer_match.group(1).strip()

        # Parse predicted box
        x1, y1, x2, y2 = parse_bbox(answer_str)
        pred_box = (x1, y1, x2, y2)

        # Parse ground-truth box (support dict or list)
        try:
            gt_data = json.loads(ground_truth)
        except json.JSONDecodeError:
            return 0.0

        if isinstance(gt_data, dict) and "bbox" in gt_data:
            gt_box = gt_data["bbox"]
        elif isinstance(gt_data, list) and len(gt_data) == 4:
            gt_box = gt_data
        else:
            return 0.0

        if not all(isinstance(v, (int, float)) for v in gt_box):
            return 0.0

        return bbox_iou(pred_box, gt_box)

    except Exception:
        return 0.0


def compute_score(
    reward_inputs: List[Dict[str, Any]], format_weight: float = 0.1
) -> List[Dict[str, float]]:
    """
    Compute combined (format + accuracy) reward for a batch of responses.

    Args:
        reward_inputs: list of dicts, e.g.
          {
            "response": "<think>...</think><answer>(x1,y1),(x2,y2)</answer>",
            "ground_truth": "[x1,y1,x2,y2]" or '{"bbox":[x1,y1,x2,y2]}'
          }
        format_weight: relative weight for format correctness in final score.
    """
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")

    scores = []
    for reward_input in reward_inputs:
        response = reward_input["response"]
        # Normalize only tag spacing, not coordinate spaces
        response = re.sub(r"\s*(</?think>)\s*", r"\1", response)
        response = re.sub(r"\s*(</?answer>)\s*", r"\1", response)

        format_score = format_reward(response)
        accuracy_score = accuracy_reward(response, reward_input["ground_truth"])
        overall = (1 - format_weight) * accuracy_score + format_weight * format_score

        scores.append(
            {
                "overall": float(overall),
                "format": float(format_score),
                "accuracy": float(accuracy_score),
            }
        )

    return scores
