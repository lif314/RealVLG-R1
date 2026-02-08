import os
import re
import json
import numpy as np
from tqdm import tqdm
from shapely.geometry import box
from PIL import Image
import argparse

from evaluation.qwen_infer import Qwen2_5VL_Infer
from verl.workers.reward.sam_wrapper import SAMWrapper
from evaluation.dataset import GraspNetVLGDataset

# ========= Utility Functions =========
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
    boxA = box(*boxA)
    boxB = box(*boxB)
    inter = boxA.intersection(boxB).area
    union = boxA.union(boxB).area
    iou = inter / union if union > 0 else 0
    x_min = min(boxA.bounds[0], boxB.bounds[0])
    y_min = min(boxA.bounds[1], boxB.bounds[1])
    x_max = max(boxA.bounds[2], boxB.bounds[2])
    y_max = max(boxA.bounds[3], boxB.bounds[3])
    enclosing = box(x_min, y_min, x_max, y_max)
    giou = iou - (enclosing.area - union) / enclosing.area
    return giou

def compute_ciou(boxA, boxB):
    boxA = np.array(boxA)
    boxB = np.array(boxB)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - inter
    iou = inter / union if union > 0 else 0
    centerA = [(boxA[0] + boxA[2]) / 2, (boxA[1] + boxA[3]) / 2]
    centerB = [(boxB[0] + boxB[2]) / 2, (boxB[1] + boxB[3]) / 2]
    center_dist = np.sum((np.array(centerA) - np.array(centerB)) ** 2)
    x_min = min(boxA[0], boxB[0])
    y_min = min(boxA[1], boxB[1])
    x_max = max(boxA[2], boxB[2])
    y_max = max(boxA[3], boxB[3])
    c = (x_max - x_min) ** 2 + (y_max - y_min) ** 2
    ciou = iou - center_dist / (c + 1e-7)
    return ciou

# ========= Segmentation Metrics =========
def f_measure(pred_mask, gt_mask):
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, np.logical_not(gt)).sum()
    fn = np.logical_and(np.logical_not(pred), gt).sum()
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)
    return f1

def s_measure(pred_mask, gt_mask, alpha=0.5):
    pred = pred_mask.astype(np.float32)
    gt = gt_mask.astype(np.float32)
    fg = pred * gt
    bg = (1 - pred) * (1 - gt)
    u_fg = fg.mean()
    u_bg = bg.mean()
    So = alpha * u_fg + (1 - alpha) * u_bg
    h, w = pred.shape
    h_mid, w_mid = h // 2, w // 2
    regions = [
        (0, h_mid, 0, w_mid),
        (0, h_mid, w_mid, w),
        (h_mid, h, 0, w_mid),
        (h_mid, h, w_mid, w),
    ]
    Ss = 0
    for (r0, r1, c0, c1) in regions:
        pred_r = pred[r0:r1, c0:c1]
        gt_r = gt[r0:r1, c0:c1]
        u_r = pred_r.mean()
        gt_mean_r = gt_r.mean()
        Ss += 2 * u_r * gt_mean_r / (u_r ** 2 + gt_mean_r ** 2 + 1e-7)
    Ss /= 4.0
    S = alpha * So + (1 - alpha) * Ss
    return S

def e_measure(pred_mask, gt_mask):
    pred = pred_mask.astype(np.float32)
    gt = gt_mask.astype(np.float32)
    pred_mean = pred.mean()
    gt_mean = gt.mean()
    align_matrix = 2 * (pred - pred_mean) * (gt - gt_mean) / (
        (pred - pred_mean) ** 2 + (gt - gt_mean) ** 2 + 1e-7
    )
    return np.mean((align_matrix + 1) / 2)

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


# ========= Evaluation Loop =========
def evaluate_seg(model_path, sam_model_path, data_root, split="seen", output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    result_path = os.path.join(output_dir, f"{split}_seg_eval.json")

    dataset = GraspNetVLGDataset(data_root, split=split)
    model = Qwen2_5VL_Infer(model_path)
    sam = SAMWrapper(sam_model_path)

    results = []
    f_scores, s_scores, e_scores = [], [], []
    giou_scores, ciou_scores = [], []

    for sample in tqdm(dataset, desc=f"Evaluating {split}"):
        gt_bbox = sample.get("bbox")
        gt_mask = None
        mask_path = sample.get("mask_path")
        if mask_path and os.path.exists(mask_path):
            mask_img = Image.open(mask_path).convert("L")
            gt_mask = np.array(mask_img)
            gt_mask = (gt_mask > 128).astype(np.uint8)

        if gt_bbox is None or len(gt_bbox) != 4 or gt_mask is None:
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
            results.append({
                "scene": sample["scene"],
                "image_name": sample["image_name"],
                "object_id": sample.get("object_id", ""),
                "description": sample["description"],
                "gt_bbox": gt_bbox,
                "pred_bbox": None,
                "response": output,
                "giou": None,
                "ciou": None,
                "f_measure": None,
                "s_measure": None,
                "e_measure": None,
            })
            continue

        giou = compute_giou(gt_bbox, pred_bbox)
        ciou = compute_ciou(gt_bbox, pred_bbox)
        giou_scores.append(giou)
        ciou_scores.append(ciou)

        sam.set_image(sample["image_path"])
        pred_mask, _ = sam.predict(pred_bbox, multimask_output=False)
        pred_mask = pred_mask.astype(bool)

        f1 = f_measure(pred_mask, gt_mask)
        s = s_measure(pred_mask, gt_mask)
        e = e_measure(pred_mask, gt_mask)
        f_scores.append(f1)
        s_scores.append(s)
        e_scores.append(e)

        results.append({
            "scene": sample["scene"],
            "image_name": sample["image_name"],
            "object_id": sample.get("object_id", ""),
            "description": sample["description"],
            "gt_bbox": gt_bbox,
            "pred_bbox": pred_bbox,
            "response": output,
            "giou": giou,
            "ciou": ciou,
            "f_measure": f1,
            "s_measure": s,
            "e_measure": e,
        })

    validity_rate = len(giou_scores) / len(results)
    summary = {
        "split": split,
        "Validity_Rate": validity_rate, 
        "mean_giou": np.mean(giou_scores) if giou_scores else 0,
        "mean_ciou": np.mean(ciou_scores) if ciou_scores else 0,
        "mean_f_measure": np.mean(f_scores) if f_scores else 0,
        "mean_s_measure": np.mean(s_scores) if s_scores else 0,
        "mean_e_measure": np.mean(e_scores) if e_scores else 0,
        "num_samples": len(results),
        "results": results,
    }

    # with open(result_path, "w", encoding="utf-8") as f:
    #     json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(to_serializable(summary), f, indent=2, ensure_ascii=False)


    print(f"\n📊 Split: {split}")
    print(
        f"gIoU: {summary['mean_giou']:.4f}, "
        f"cIoU: {summary['mean_ciou']:.4f}, "
        f"F-measure: {summary['mean_f_measure']:.4f}, "
        f"S-measure: {summary['mean_s_measure']:.4f}, "
        f"E-measure: {summary['mean_e_measure']:.4f}"
    )
    print(f"💾 Results saved to: {result_path}")
    return summary

# ========= CLI =========
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Segmentation with Qwen2.5-VL + SAM")
    parser.add_argument("--model_path", type=str, required=True, help="Path to Qwen model")
    parser.add_argument("--sam_model_path", type=str, required=True, help="Path to SAM model")
    parser.add_argument("--data_root", type=str, required=True, help="Root directory of GraspNet VLG dataset")
    parser.add_argument("--output_dir", type=str, default="./outputs/evaluation/seg", help="Directory to save results")
    parser.add_argument("--split", type=str, default=None, choices=["seen", "similar", "novel"], help="Specific split to evaluate (optional)")
    args = parser.parse_args()

    splits = [args.split] if args.split else ["seen", "similar", "novel"]
    for split in splits:
        evaluate_seg(
            model_path=args.model_path,
            sam_model_path=args.sam_model_path,
            data_root=args.data_root,
            split=split,
            output_dir=args.output_dir
        )
