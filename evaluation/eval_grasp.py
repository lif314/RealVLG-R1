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
# Convert (x, y, theta, width) → 8 corner points representation
# ==========================================================
def rect_to_points8(x: float, y: float, theta: float, width: float, gripper_depth: float = 40.0) -> np.ndarray:
    """Convert grasp parameters (x, y, theta, width) into 8-point rectangle coordinates."""
    theta_rad = np.deg2rad(theta)
    dx = (width / 2) * np.cos(theta_rad)
    dy = (width / 2) * np.sin(theta_rad)
    dz = gripper_depth / 2

    points = np.array([
        x - dx - dz * np.sin(theta_rad), y - dy + dz * np.cos(theta_rad),
        x + dx - dz * np.sin(theta_rad), y + dy + dz * np.cos(theta_rad),
        x + dx + dz * np.sin(theta_rad), y + dy - dz * np.cos(theta_rad),
        x - dx + dz * np.sin(theta_rad), y - dy - dz * np.cos(theta_rad)
    ])
    return points


# ==========================================================
# Convert 8 corner points → (x, y, theta, width)
# ==========================================================
def points8_to_rect(points8: np.ndarray):
    """Convert 8 points [x0, y0, ..., x3, y3] back into (x, y, theta, width)."""
    if points8.shape[0] != 8:
        raise ValueError("points8 should have 8 elements (4 points × 2 coords).")

    corners = points8.reshape(4, 2)
    x = np.mean(corners[:, 0])
    y = np.mean(corners[:, 1])

    c1 = np.mean(corners[[1, 2]], axis=0)
    c2 = np.mean(corners[[0, 3]], axis=0)
    width = np.linalg.norm(c1 - c2)

    dx = corners[1, 0] - corners[0, 0]
    dy = corners[1, 1] - corners[0, 1]
    theta = np.rad2deg(np.arctan2(dy, dx))

    return x, y, theta, width


# ==========================================================
# Parse (x, y, theta, width) from model text output
# ==========================================================
def parse_rect(answer_str: str):
    """
    Parse numerical grasp parameters (x, y, theta, width) from model output.
    Expected format inside <answer>...</answer>: (x, y, theta, width)
    """
    # Step 1: Extract <answer> content if exists
    answer_match = re.search(r"<answer>(.*?)</answer>", answer_str, re.DOTALL)
    if answer_match:
        content = answer_match.group(1).strip()
    else:
        content = answer_str.strip()

    # Step 2: Remove extra whitespace/newlines
    content = re.sub(r"\s+", " ", content)

    # Step 3: Match all numbers (integer, decimal, scientific notation)
    parts = re.findall(r"-?\d*\.?\d+(?:[eE][-+]?\d+)?", content)
    if len(parts) != 4:
        raise ValueError(f"Invalid format: {content}")

    return list(map(float, parts))  # [x, y, theta, width]


# ==========================================================
# Compute IoU and angular difference
# ==========================================================
def compute_iou(points1, points2):
    """Compute IoU between two grasp rectangles defined by 8-point coordinates."""
    poly1 = Polygon(points1.reshape(4, 2))
    poly2 = Polygon(points2.reshape(4, 2))
    if not poly1.is_valid or not poly2.is_valid:
        return 0.0
    inter_area = poly1.intersection(poly2).area
    union_area = poly1.union(poly2).area
    return inter_area / union_area if union_area > 0 else 0.0


def angular_diff(pred_theta, gt_theta):
    """
    Compute angular difference (degrees) between predicted and ground-truth θ.
    The comparison uses cosine and sine values to ensure robustness to:
      - radians vs degrees
      - periodicity (θ and θ+180° equivalence)
      - negative/positive angle conventions
    """

    # Automatically detect whether the value is in radians
    if abs(pred_theta) <= np.pi and abs(gt_theta) <= np.pi:
        pred_rad, gt_rad = pred_theta, gt_theta
    else:
        pred_rad, gt_rad = np.deg2rad(pred_theta), np.deg2rad(gt_theta)

    # Convert to 2D unit vectors
    v_pred = np.array([np.cos(pred_rad), np.sin(pred_rad)])
    v_gt = np.array([np.cos(gt_rad), np.sin(gt_rad)])

    # Compute cosine similarity and clamp for numerical stability
    cos_sim = np.clip(np.dot(v_pred, v_gt), -1.0, 1.0)

    # Convert to angular difference in degrees
    angle_diff = np.rad2deg(np.arccos(cos_sim))

    # Fold into [0, 90] range because θ and θ+180° represent the same gripper orientation
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
def evaluate_rect_grasp(model_path, data_root, split="seen", output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    result_path = os.path.join(output_dir, f"{split}_rect_eval.json")

    dataset = GraspNetVLGDataset(data_root, split=split)
    model = Qwen2_5VL_Infer(model_path)

    iou_list, acc_list, results = [], [], []

    for i, sample in tqdm(enumerate(dataset), total=len(dataset), desc=f"Evaluating {split}"):
        gt_grasp_list = sample["grasps"]  # list of Nx8 numpy arrays
        image_path = sample["image_path"]
        image_name = sample["image_name"]
        scene = sample.get("scene", "")
        object_id = sample.get("object_id", "")
        description = sample["description"]

        prompt = (
            f"Predict a stable 2D rectangular grasp pose for the target object "
            f"based on the instruction: \"{description.strip()}\". "
            f"First output the reasoning in <think></think> tags and then "
            f"output the final grasp pose in <answer></answer> tags. "
            f"Follow the format: <think>...</think> <answer>(x, y, theta, width)</answer>"
        )

        try:
            output = model.infer(image_path, prompt)
            pred_x, pred_y, pred_theta, pred_width = parse_rect(output)
            pred_points = rect_to_points8(pred_x, pred_y, pred_theta, pred_width)
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

        # Compare with all GT grasps and keep the best IoU match
        best_iou, best_angle = 0.0, 999
        for gt_points in gt_grasp_list:
            gt_x, gt_y, gt_theta, gt_w = points8_to_rect(np.array(gt_points))
            iou = compute_iou(pred_points, np.array(gt_points))
            angle_diff = angular_diff(pred_theta, gt_theta)
            if iou > best_iou:
                best_iou = iou
                best_angle = angle_diff

        is_correct = int(best_iou > 0.25 and best_angle < 30)
        iou_list.append(best_iou)
        acc_list.append(is_correct)

        results.append({
            "scene": scene,
            "image_name": image_name,
            "object_id": object_id,
            "description": description,
            "pred": [pred_x, pred_y, pred_theta, pred_width],
            "response": output,
            "iou": best_iou,
            "angle_diff": best_angle,
            "acc": is_correct,
        })

    # ==========================================================
    # Compute aggregated metrics
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
    parser = argparse.ArgumentParser(description="Evaluate rectangular grasp pose predictions using Qwen2.5-VL.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to Qwen2.5-VL model.")
    parser.add_argument("--data_root", type=str, required=True, help="Root directory of the GraspNet-VLG dataset.")
    parser.add_argument("--output_dir", type=str, default="./outputs/evaluation/grasp", help="Directory to save results.")
    parser.add_argument("--split", type=str, default=None, choices=["seen", "similar", "novel"], help="Dataset split to evaluate.")
    args = parser.parse_args()

    splits = [args.split] if args.split else ["seen", "similar", "novel"]
    for split in splits:
        evaluate_rect_grasp(
            model_path=args.model_path,
            data_root=args.data_root,
            split=split,
            output_dir=args.output_dir
        )
