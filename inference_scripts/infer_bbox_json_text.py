import argparse
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from PIL import Image as PILImage
import cv2
import numpy as np
import re
import os
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="pretrain_models/bbox/qwen3b_cov_grpo_global_step_106/actor/huggingface")
    parser.add_argument("--text", type=str, default="The yellow banana.")
    parser.add_argument("--image_path", type=str, default="assets/0000.png")
    parser.add_argument("--save_dir", type=str, default="inference_scripts/outputs/bbox")
    return parser.parse_args()


# === 坐标解析函数 ===
def parse_bbox_from_text(text):
    """解析形如 <answer>(x1,y1),(x2,y2)</answer> 的输出"""
    match = re.search(
        r"<answer>\s*\(?(\d+)\s*,\s*(\d+)\)?\s*,\s*\(?(\d+)\s*,\s*(\d+)\)?\s*</answer>",
        text
    )
    if match:
        return tuple(map(int, match.groups()))
    return None


def parse_bbox_from_json(text):
    """解析形如 <answer>{"bbox": [x1, y1, x2, y2]}</answer> 的输出"""
    # 提取 JSON 子串
    json_match = re.search(r"\{[^{}]+\}", text)
    if not json_match:
        return None

    try:
        data = json.loads(json_match.group(0))
        bbox_key = next((k for k in data if "bbox" in k.lower()), None)
        if bbox_key and isinstance(data[bbox_key], (list, tuple)) and len(data[bbox_key]) == 4:
            return tuple(map(int, data[bbox_key]))
    except json.JSONDecodeError:
        return None
    return None


def extract_bbox(output_text):
    """自动选择解析方式"""
    # 优先尝试 JSON
    bbox = parse_bbox_from_json(output_text)
    if bbox:
        print("[Info] Detected JSON bbox format.")
        return bbox

    # 否则尝试 (x1,y1),(x2,y2)
    bbox = parse_bbox_from_text(output_text)
    if bbox:
        print("[Info] Detected (x1,y1),(x2,y2) bbox format.")
        return bbox

    return None


# === 可视化保存函数 ===
def draw_and_save_bbox(image_path, bbox, save_dir, label="prediction"):
    os.makedirs(save_dir, exist_ok=True)
    img = cv2.imread(image_path)
    if img is None:
        print(f"[Warning] Failed to load image: {image_path}")
        return

    x1, y1, x2, y2 = bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.putText(img, label, (x1, max(y1 - 10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    save_path = os.path.join(save_dir, os.path.basename(image_path))
    cv2.imwrite(save_path, img)
    print(f"[Saved] Visualization with bbox → {save_path}")


# === 主函数 ===
def main():
    args = parse_args()

    # 加载模型
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2"
    ).eval()

    # 加载处理器
    processor = AutoProcessor.from_pretrained(args.model_path, padding_side="left")

    # Prompt 模板，允许输出 JSON 或坐标形式
    QUESTION_TEMPLATE = (
        "Please find '{Question}' in the image and predict its bounding box. "
        "Output the reasoning in <think> </think> and final result in <answer> </answer>. "
        "You can output either: "
        "(1) <answer>(x1,y1),(x2,y2)</answer> or "
        "(2) <answer>{\"bbox\": [x1, y1, x2, y2]}</answer>."
    )

    image = PILImage.open(args.image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": QUESTION_TEMPLATE.format(Question=args.text.strip("."))}
            ]
        }
    ]

    text = [processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)]
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    # === 推理 ===
    generated_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    generated_ids_trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]

    print("\n=== Model Output ===")
    print(output_text)

    # === 解析 bbox ===
    bbox = extract_bbox(output_text)

    if bbox:
        print(f"✅ Predicted BBox: {bbox}")
        draw_and_save_bbox(args.image_path, bbox, args.save_dir)
    else:
        print("⚠️ No valid bbox detected in output.")


if __name__ == "__main__":
    main()
