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
from typing import Dict, List, Any
import numpy as np
import shapely.geometry
import json

def format_reward(response: str) -> float:
    re_str = r"<think>.*?</think>\s*<answer>\s*\(\s*\d+\s*,\s*\d+\s*\)\s*,\s*\(\s*\d+\s*,\s*\d+\s*\)\s*</answer>"
    pattern = re.compile(re_str, re.DOTALL)
    format_match = re.fullmatch(pattern, response)
    return 1.0 if format_match else 0.0

def accuracy_reward(predict: str, ground_truth: str) -> float:
    COORD_PATTERN = re.compile(
        r"\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*,\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)"
    )
    ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

    def parse_touch_points(answer_str):
        match = COORD_PATTERN.fullmatch(answer_str.strip())
        if not match:
            raise ValueError(f"Incorrect Coordinate format: {answer_str}")
        return list(map(int, match.groups()))

    def compute_grasp_bbox(x1, y1, x2, y2, gripper_width=80):
        try:
            p1, p2 = np.array([x1, y1], dtype=float), np.array([x2, y2], dtype=float)
            theta = np.arctan2(p2[1]-p1[1], p2[0]-p1[0])
            dx = (gripper_width / 2) * np.sin(theta)
            dy = (gripper_width / 2) * np.cos(theta)
            return np.array([
                [p1[0]-dx, p1[1]+dy],
                [p1[0]+dx, p1[1]-dy],
                [p2[0]+dx, p2[1]-dy],
                [p2[0]-dx, p2[1]+dy]
            ])
        except Exception as e:
            raise ValueError(f"Grasping box calculation failed: {e} | Input coordinates: ({x1},{y1}),({x2},{y2})")

    def polygon_iou(poly1, poly2):
        try:
            poly1 = shapely.geometry.Polygon(poly1).buffer(0)
            poly2 = shapely.geometry.Polygon(poly2).buffer(0)
            if not poly1.is_valid or not poly2.is_valid:
                return 0.0
            intersection = poly1.intersection(poly2).area
            union = poly1.union(poly2).area
            return intersection / union if union > 1e-6 else 0.0  # Avoid divided by 0
        except Exception as e:
            return 0.0

    reward = 0.0    
    try:
        answer_match = ANSWER_PATTERN.search(predict)
        if not answer_match:
            return reward
            
        student_answer = answer_match.group(1).strip()
        try:
            x1, y1, x2, y2 = parse_touch_points(student_answer)
        except ValueError as e:
            return reward
        
        json_data = json.loads(ground_truth)
        sol_points = json_data["contact_points"]
        student_bbox = compute_grasp_bbox(x1, y1, x2, y2)
        iou_scores = []
        for sol_p in sol_points:
            try:
                sol_bbox = compute_grasp_bbox(sol_p[0], sol_p[1], sol_p[2], sol_p[3])
                iou = polygon_iou(student_bbox, sol_bbox)
                iou_scores.append(iou)
            except Exception as e:
                continue
                
        reward = max(iou_scores) if iou_scores else 0.0
        
    except Exception as e:
        reward = 0.0

    return reward


def compute_score(reward_inputs: list[dict[str, Any]], format_weight: float = 0.1) -> list[dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")

    scores = []
    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])  # handle qwen2.5vl-32b format
        format_score = format_reward(response)
        accuracy_score = accuracy_reward(response, reward_input["ground_truth"])
        scores.append(
            {
                "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
                "format": format_score,
                "accuracy": accuracy_score,
            }
        )

    return scores
