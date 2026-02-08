import os
import re
import json
import numpy as np
from tqdm import tqdm
from shapely.geometry import Polygon
import argparse

from evaluation.dataset import GraspNetVLGDataset
from evaluation.qwen_infer import Qwen2_5VL_Infer


# ==========================================================
# Compute grasp rectangle from contact points
# ==========================================================
def compute_grasp_bbox(x1, y1, x2, y2, gripper_width=80):
    """
    Given two contact points (x1, y1), (x2, y2), construct a grasp rectangle.
    The rectangle width corresponds to the gripper finger span,
    and its orientation is defined by the line connecting the two points.
    """
    try:
        p1, p2 = np.array([x1, y1], dtype=float), np.array([x2, y2], dtype=float)
        theta = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
        dx = (gripper_width / 2) * np.sin(theta)
        dy = (gripper_width / 2) * np.cos(theta)
        return np.array([
            [p1[0] - dx, p1[1] + dy],
            [p1[0] + dx, p1[1] - dy],
            [p2[0] + dx, p2[1] - dy],
            [p2[0] - dx, p2[1] + dy]
        ])
    except Exception as e:
        raise ValueError(f"Grasp rectangle computation failed: {e} | Input: ({x1},{y1}),({x2},{y2})")


# ==========================================================
# IoU computation between polygons
# ==========================================================
def polygon_iou(poly1, poly2):
    """Compute IoU between two polygons."""
    try:
        poly1 = Polygon(poly1).buffer(0)
        poly2 = Polygon(poly2).buffer(0)
        if not poly1.is_valid or not poly2.is_valid:
            return 0.0
        inter = poly1.intersection(poly2).area
        union = poly1.union(poly2).area
        return inter / union if union > 1e-6 else 0.0
    except Exception:
        return 0.0


# ==========================================================
# Parse contact points from model output
# ==========================================================
def parse_touch_points(answer_str):
    """
    Parse two 2D contact points from model output.
    Expected format inside <answer>...</answer>: "(x1, y1), (x2, y2)"
    """
    # Step 1. Extract the <answer> content if exists
    answer_match = re.search(r"<answer>(.*?)</answer>", answer_str, re.DOTALL)
    if answer_match:
        content = answer_match.group(1).strip()
    else:
        # fallback: use the raw string
        content = answer_str.strip()

    # Step 2. Remove unnecessary whitespace, newlines
    content = re.sub(r"\s+", " ", content)

    # Step 3. Match coordinate pairs
    COORD_PATTERN = re.compile(
        r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)\s*,\s*\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)"
    )
    match = COORD_PATTERN.search(content)
    if not match:
        raise ValueError(f"Failed to parse contact points from: {content}")

    return list(map(float, match.groups()))  # [x1, y1, x2, y2]

# ==========================================================
# Compute angular difference
# ==========================================================
def angular_diff(pred_theta, gt_theta):
    """
    Compute angular difference (degrees) between predicted and GT orientations.
    Robust to:
      - radians vs degrees
      - 180° periodicity
      - sign ambiguity
    """
    # Detect radians
    if abs(pred_theta) <= np.pi and abs(gt_theta) <= np.pi:
        pred_rad, gt_rad = pred_theta, gt_theta
    else:
        pred_rad, gt_rad = np.deg2rad(pred_theta), np.deg2rad(gt_theta)

    v_pred = np.array([np.cos(pred_rad), np.sin(pred_rad)])
    v_gt = np.array([np.cos(gt_rad), np.sin(gt_rad)])

    cos_sim = np.clip(np.dot(v_pred, v_gt), -1.0, 1.0)
    angle_diff = np.rad2deg(np.arccos(cos_sim))

    # Fold into [0, 90] range (gripper symmetry)
    if angle_diff > 90:
        angle_diff = 180 - angle_diff
    return angle_diff

def to_serializable(obj):
    """Recursively convert numpy types to native Python types."""
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    elif isinstance(obj, np.generic):  # np.float32, np.int64 等
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj
    
# ==========================================================
# Main evaluation loop
# ==========================================================
def evaluate_contact_grasp(model_path, data_root, split="seen", output_dir="outputs"):
    """
    Evaluate 2D contact-point predictions by converting both predicted and GT pairs
    into rectangular grasp representations for unified comparison.
    """
    os.makedirs(output_dir, exist_ok=True)
    result_path = os.path.join(output_dir, f"{split}_contact_eval.json")

    dataset = GraspNetVLGDataset(data_root, split=split)
    model = Qwen2_5VL_Infer(model_path)

    iou_list, acc_list, results = [], [], []

    for i, sample in tqdm(enumerate(dataset), total=len(dataset), desc=f"Evaluating {split}"):
        gt_contact_list = sample["contact_points"]  # list of [[x1, y1, x2, y2], ...]
        image_path = sample["image_path"]
        image_name = sample["image_name"]
        scene = sample.get("scene", "")
        object_id = sample.get("object_id", "")
        description = sample["description"]

        prompt = (
            f"Predict one stable two-finger grasp contact pair (two 2D coordinates) for the target object "
            f"described in the instruction: \"{description.strip()}\". "
            f"First output your reasoning in <think></think> tags, "
            f"then provide the final grasp in <answer></answer> tags. "
            f"Follow the format: <think> thinking process </think> <answer>(x1,y1),(x2,y2)</answer>"
        )

        try:
            output = model.infer(image_path, prompt)
            x1, y1, x2, y2 = parse_touch_points(output)
            pred_box = compute_grasp_bbox(x1, y1, x2, y2)
            pred_theta = np.rad2deg(np.arctan2(y2 - y1, x2 - x1))
        except Exception as e:
            results.append({
                "scene": scene,
                "image_name": image_name,
                "object_id": object_id,
                "description": description,
                "pred": None,
                "response": str(e),
                "iou": None,
                "angle_diff": None,
                "acc": None,
            })
            continue

        # Compare with all GT contact pairs
        best_iou, best_angle = 0.0, 999
        for gt in gt_contact_list:
            try:
                gx1, gy1, gx2, gy2 = gt
                gt_box = compute_grasp_bbox(gx1, gy1, gx2, gy2)
                gt_theta = np.rad2deg(np.arctan2(gy2 - gy1, gx2 - gx1))

                iou = polygon_iou(pred_box, gt_box)
                angle_diff = angular_diff(pred_theta, gt_theta)

                if iou > best_iou:
                    best_iou = iou
                    best_angle = angle_diff
            except Exception:
                continue

        is_correct = int(best_iou > 0.25 and best_angle < 30)
        iou_list.append(best_iou)
        acc_list.append(is_correct)

        results.append({
            "scene": scene,
            "image_name": image_name,
            "object_id": object_id,
            "description": description,
            "response": output,
            "pred": pred_box,
            "iou": best_iou,
            "angle_diff": best_angle,
            "acc": is_correct,
        })

    # ==========================================================
    # Aggregated metrics
    # ==========================================================
    mIoU = np.mean(iou_list) if iou_list else 0.0
    gAcc = np.mean(acc_list) if acc_list else 0.0

    validity_rate = len(iou_list) / len(results)
    summary = {
        "split": split,
        "Validity_Rate": validity_rate, 
        "mIoU": mIoU,
        "gAcc": gAcc,
        "num_samples": len(results),
        "results": results,
    }

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(to_serializable(summary), f, indent=2, ensure_ascii=False)

    print(f"\n📊 Split: {split}")
    print(f"Mean IoU (mIoU): {mIoU:.4f}")
    print(f"Grasp Accuracy (gAcc): {gAcc:.4f}")
    print(f"💾 Results saved to: {result_path}")

    return mIoU, gAcc


# ==========================================================
# CLI entry point
# ==========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate contact-point grasp predictions using Qwen2.5-VL.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to Qwen2.5-VL model.")
    parser.add_argument("--data_root", type=str, required=True, help="Root directory of the GraspNet-VLG dataset.")
    parser.add_argument("--output_dir", type=str, default="./outputs/evaluation/grasp", help="Directory to save results.")
    parser.add_argument("--split", type=str, default=None, choices=["seen", "similar", "novel"], help="Dataset split to evaluate.")
    args = parser.parse_args()

    splits = [args.split] if args.split else ["seen", "similar", "novel"]
    for split in splits:
        evaluate_contact_grasp(
            model_path=args.model_path,
            data_root=args.data_root,
            split=split,
            output_dir=args.output_dir
        )