import os
import re
import json
import numpy as np
from tqdm import tqdm
from shapely.geometry import box
import argparse

from evaluation.dataset import GraspNetVLGDataset
from evaluation.qwen_infer import Qwen2_5VL_Infer

# ============ Evaluation Metrics =============
def parse_bbox_from_output(text):
    """
    Parse bounding box coordinates (x_min, y_min, x_max, y_max) from model output.
    Expected format inside <answer>...</answer>:
        (x_min, y_min), (x_max, y_max)
    """
    # Step 1: Extract <answer> content if exists
    answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if answer_match:
        content = answer_match.group(1).strip()
    else:
        # fallback to full text
        content = text.strip()

    # Step 2: Remove extra whitespace/newlines
    content = re.sub(r"\s+", " ", content)

    # Step 3: Match coordinates
    bbox_pattern = re.compile(
        r"\(?\s*(-?\d+(?:\.\d+)?)\s*[,，]\s*(-?\d+(?:\.\d+)?)\s*\)?\s*[,，]\s*\(?\s*(-?\d+(?:\.\d+)?)\s*[,，]\s*(-?\d+(?:\.\d+)?)\s*\)?"
    )
    match = bbox_pattern.search(content)
    if not match:
        return None

    return [float(match.group(1)), float(match.group(2)),
            float(match.group(3)), float(match.group(4))]


def compute_giou(boxA, boxB):
    """Generalized IoU"""
    assert len(boxA) == 4 and len(boxB) == 4, "Invalid bbox format"
    boxA = box(*boxA)
    boxB = box(*boxB)
    inter = boxA.intersection(boxB).area
    union = boxA.union(boxB).area
    iou = inter / union if union > 0 else 0
    # Enclosing box
    x_min = min(boxA.bounds[0], boxB.bounds[0])
    y_min = min(boxA.bounds[1], boxB.bounds[1])
    x_max = max(boxA.bounds[2], boxB.bounds[2])
    y_max = max(boxA.bounds[3], boxB.bounds[3])
    enclosing = box(x_min, y_min, x_max, y_max)
    giou = iou - (enclosing.area - union) / enclosing.area
    return giou


def compute_ciou(boxA, boxB):
    """Complete IoU (approximation)"""
    boxA = np.array(boxA)
    boxB = np.array(boxB)

    # IoU
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - inter
    iou = inter / union if union > 0 else 0

    # Center distance
    centerA = [(boxA[0] + boxA[2]) / 2, (boxA[1] + boxA[3]) / 2]
    centerB = [(boxB[0] + boxB[2]) / 2, (boxB[1] + boxB[3]) / 2]
    center_dist = np.sum((np.array(centerA) - np.array(centerB)) ** 2)

    # Enclosing diagonal
    x_min = min(boxA[0], boxB[0])
    y_min = min(boxA[1], boxB[1])
    x_max = max(boxA[2], boxB[2])
    y_max = max(boxA[3], boxB[3])
    c = (x_max - x_min) ** 2 + (y_max - y_min) ** 2

    ciou = iou - center_dist / (c + 1e-7)
    return ciou

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
    
# ============ Evaluation Loop =============
def evaluate_bbox(model_path, data_root, split="seen", output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    result_path = os.path.join(output_dir, f"{split}_bbox_eval.json")

    dataset = GraspNetVLGDataset(data_root, split=split)
    model = Qwen2_5VL_Infer(model_path)

    giou_scores, ciou_scores = [], []
    results = []

    for i, sample in tqdm(enumerate(dataset), total=len(dataset), desc=f"Evaluating {split}"):
        gt_bbox = sample["bbox"]
        if not gt_bbox or len(gt_bbox) != 4:
            continue

        prompt = (
            f"Predict the bounding box of the referred object in the image "
            f"based on the instruction: \"{sample['description'].strip()}\". "
            f"First output the thinking process in <think></think> tags and then "
            f"output the final answer in <answer></answer> tags. "
            f"Follow the format: <think>thinking</think>\n"
            f"<answer>(x_min,y_min),(x_max,y_max)</answer>"
        )

        output = model.infer(sample["image_path"], prompt)
        pred_bbox = parse_bbox_from_output(output)

        if pred_bbox is None:
            result = {
                "scene": sample["scene"],
                "image_name": sample["image_name"],
                "object_id": sample.get("object_id", ""),
                "description": sample["description"],
                "gt_bbox": gt_bbox,
                "pred_bbox": None,
                "response": output,
                "giou": None,
                "ciou": None,
            }
            results.append(result)
            continue

        giou = compute_giou(gt_bbox, pred_bbox)
        ciou = compute_ciou(gt_bbox, pred_bbox)
        giou_scores.append(giou)
        ciou_scores.append(ciou)

        result = {
            "scene": sample["scene"],
            "image_name": sample["image_name"],
            "object_id": sample.get("object_id", ""),
            "description": sample["description"],
            "gt_bbox": gt_bbox,
            "pred_bbox": pred_bbox,
            "response": output,
            "giou": giou,
            "ciou": ciou,
        }
        results.append(result)

    mean_giou = np.mean([r["giou"] for r in results if r["giou"] is not None]) if results else 0
    mean_ciou = np.mean([r["ciou"] for r in results if r["ciou"] is not None]) if results else 0

    validity_rate = len(giou_scores) / len(results)
    summary = {
        "split": split,
        "Validity_Rate": validity_rate, 
        "mean_giou": mean_giou,
        "mean_ciou": mean_ciou,
        "num_samples": len(results),
        "results": results,
    }

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(to_serializable(summary), f, indent=2, ensure_ascii=False)

    print(f"\n📊 Split: {split}")
    print(f"gIoU: {mean_giou:.4f}, cIoU: {mean_ciou:.4f}")
    print(f"💾 Results saved to: {result_path}")

    return mean_giou, mean_ciou

# ============ Example Usage ============
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Bbox with Qwen2.5-VL")
    parser.add_argument("--model_path", type=str, required=True, help="Path to Qwen model")
    parser.add_argument("--data_root", type=str, required=True, help="Root directory of GraspNet VLG dataset")
    parser.add_argument("--output_dir", type=str, default="./outputs/evaluation/seg", help="Directory to save results")
    parser.add_argument("--split", type=str, default=None, choices=["seen", "similar", "novel"], help="Specific split to evaluate (optional)")
    args = parser.parse_args()

    splits = [args.split] if args.split else ["seen", "similar", "novel"]
    for split in splits:
        evaluate_bbox(
            model_path=args.model_path,
            data_root=args.data_root,
            split=split,
            output_dir=args.output_dir
        )