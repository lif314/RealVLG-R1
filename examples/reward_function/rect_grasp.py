# -*- coding: utf-8 -*-
import re
import json
import numpy as np
import shapely.geometry
from typing import Any, List, Dict

_EPS = np.spacing(1)

# ==========================
# XY-Theta-Width -> 8 points
# ==========================
def rect_to_points8(x: float, y: float, theta: float, width: float, gripper_depth: float = 40.0) -> np.ndarray:
    """
    Convert (x, y, theta, width) to 8 points [x0,y0,...,x3,y3]
    """
    theta_rad = np.deg2rad(theta)
    dx = (width / 2) * np.cos(theta_rad)
    dy = (width / 2) * np.sin(theta_rad)
    dz = gripper_depth / 2

    points = np.array([
        x - dx - dz*np.sin(theta_rad), y - dy + dz*np.cos(theta_rad),
        x + dx - dz*np.sin(theta_rad), y + dy + dz*np.cos(theta_rad),
        x + dx + dz*np.sin(theta_rad), y + dy - dz*np.cos(theta_rad),
        x - dx + dz*np.sin(theta_rad), y - dy - dz*np.cos(theta_rad)
    ])
    return points

# ==========================
# 8 points -> XY-Theta-Width
# ==========================
def points8_to_rect(points8: np.ndarray) -> dict:
    """
    Convert 8 points [x0,y0,...,x3,y3] to (x, y, theta, width)
    """
    if points8.shape[0] != 8:
        raise ValueError("points8 should have 8 elements.")

    corners = points8.reshape(4,2)

    # center
    x = np.mean(corners[:,0])
    y = np.mean(corners[:,1])

    # width: distance between midpoints of opposing edges
    c1 = np.mean(corners[[1,2]], axis=0)
    c2 = np.mean(corners[[0,3]], axis=0)
    width = np.linalg.norm(c1 - c2)

    # angle: vector from corner 0->1
    dx = corners[1,0] - corners[0,0]
    dy = corners[1,1] - corners[0,1]
    theta = np.rad2deg(np.arctan2(dy, dx))

    return x, y, theta, width

# ==========================
# Format Reward
# ==========================
def format_reward(response: str) -> float:
    """
    Check if the response matches the expected format:
    <think>...</think>\n<answer>(x, y, theta, width)</answer>
    Supports floats, negative numbers, and scientific notation.
    """
    float_pattern = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
    re_str = (
        rf"<think>.*?</think>\s*"
        rf"<answer>\s*\(\s*{float_pattern}\s*,\s*{float_pattern}\s*,\s*{float_pattern}\s*,\s*{float_pattern}\s*\)\s*</answer>"
    )
    pattern = re.compile(re_str, re.DOTALL)
    return 1.0 if re.fullmatch(pattern, response.strip()) else 0.0

# ==========================
# Huber loss per-component
# ==========================
def huber_single(x: float, y: float, delta: float = 1.0) -> float:
    """Single-element Huber / Smooth L1 loss (like PyTorch F.smooth_l1_loss)"""
    diff = x - y
    abs_diff = np.abs(diff)
    if abs_diff <= delta:
        return 0.5 * diff ** 2
    else:
        return delta * (abs_diff - 0.5 * delta)

# ==========================
# Helper functions
# ==========================
def parse_rect(answer_str: str):
    """Parse (x, y, theta, width) from <answer>"""
    parts = re.findall(r"-?\d*\.?\d+(?:[eE][-+]?\d+)?", answer_str)
    if len(parts) != 4:
        raise ValueError(f"Invalid format: {answer_str}")
    return list(map(float, parts))  # [x, y, theta, width]


def polygon_iou(poly1, poly2):
    """Compute IoU of two polygons"""
    try:
        poly1 = shapely.geometry.Polygon(poly1).buffer(0)
        poly2 = shapely.geometry.Polygon(poly2).buffer(0)
        if not poly1.is_valid or not poly2.is_valid:
            return 0.0
        inter = poly1.intersection(poly2).area
        union = poly1.union(poly2).area
        return inter / union if union > 1e-6 else 0.0
    except:
        return 0.0

# ==========================
# Huber reward and IoU
# ==========================
def huber_reward(predict: str, ground_truth: str) -> Dict[str, float]:
    """
    Compute train_reward using per-component Huber (Smooth L1) loss,
    and IoU score for evaluation.
    Only IoU is used; angle difference is ignored.

    Returns dict with keys:
        'train_reward': [0,1], 1 is best
        'iou_score': max IoU for grasps
    """
    ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
    reward_dict = {'train_reward': 0.0, 'iou_score': 0.0}

    try:
        answer_match = ANSWER_PATTERN.search(predict)
        if not answer_match:
            return reward_dict

        x, y, theta, width = parse_rect(answer_match.group(1).strip())
        student_corners = rect_to_points8(x, y, theta, width)
        student_vec = np.array([x, y, np.cos(np.deg2rad(theta)), np.sin(np.deg2rad(theta)), width])

        gt_json = json.loads(ground_truth)
        gt_grasp_points_list = gt_json.get("grasps", [])

        best_reward = 0.0
        max_iou = 0.0

        for gt_points in gt_grasp_points_list:
            if len(gt_points) != 8:
                continue
            gt_points = np.array(gt_points)
            gt_x, gt_y, gt_theta, gt_width = points8_to_rect(gt_points)
            gt_vec = np.array([gt_x, gt_y, np.cos(np.deg2rad(gt_theta)), np.sin(np.deg2rad(gt_theta)), gt_width])

            # Huber loss per component
            total_loss = sum([huber_single(s, g) for s, g in zip(student_vec, gt_vec)])
            # Normalize reward to [0,1]: smaller loss -> reward closer to 1
            reward = 1 / (1 + total_loss)
            best_reward = max(best_reward, reward)

            # IoU evaluation (only IoU, ignore angle)
            iou = polygon_iou(student_corners.reshape(4,2), gt_points.reshape(4,2))
            max_iou = max(max_iou, iou)

        reward_dict['train_reward'] = best_reward
        reward_dict['iou_score'] = max_iou

    except Exception:
        pass

    return reward_dict

# ==========================
# Compute batch scores
# ==========================
def compute_score(reward_inputs: List[Dict[str, Any]], format_weight: float = 0.1) -> List[Dict[str, float]]:
    """
    Compute overall score for a batch of predictions.
    Each input dict should contain 'response' and 'ground_truth'.
    """
    if not isinstance(reward_inputs, list):
        raise ValueError("Please provide a list of reward_inputs.")

    scores = []
    for reward_input in reward_inputs:
        response = re.sub(r">\s+<", "><", reward_input["response"])  # normalize spacing
        format_score = format_reward(response)
        reward_dict = huber_reward(response, reward_input["ground_truth"])
        overall = (1 - format_weight) * reward_dict['train_reward'] + format_weight * format_score

        scores.append({
            "overall": overall,
            "format": format_score,
            "accuracy": reward_dict['train_reward'],
            "iou_score": reward_dict['iou_score']
        })

    return scores