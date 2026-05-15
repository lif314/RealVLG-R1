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
    parser.add_argument("--save_dir", type=str, default="inference_scripts/outputs/bbox")  # 保存预测图像的目录
    return parser.parse_args()


def parse_bbox(text):
    """
    <answer>(x1,y1),(x2,y2)</answer>
    """
    match = re.search(r"<answer>\s*\(?(\d+)\s*,\s*(\d+)\)?\s*,\s*\(?(\d+)\s*,\s*(\d+)\)?\s*</answer>", text)
    if not match:
        return None
    x1, y1, x2, y2 = map(int, match.groups())
    return (x1, y1, x2, y2)

def extract_bbox_from_json(output_text):
    json_pattern = r'{[^}]+}'
    json_match = re.search(json_pattern, output_text)
    if json_match:
        data = json.loads(json_match.group(0))
        bbox_key = next((key for key in data.keys() if 'bbox' in key.lower()), None)
        if bbox_key and len(data[bbox_key]) == 4:
            content_bbox = data[bbox_key]
            return tuple(content_bbox)

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


def main():
    args = parse_args()

    # Load model
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2"
    ).eval()
    
    # Load processor
    processor = AutoProcessor.from_pretrained(args.model_path, padding_side="left")

    # QUESTION_TEMPLATE = (
    #     "Given the instruction: '{Question}', predict the bounding box of the target object in the image. "
    #     "First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. "
    #     "Following \"<think> thinking process </think>\\n<answer>(x1,y1),(x2,y2)</answer>\" format."
    # )

    QUESTION_TEMPLATE = \
        "Please find '{Question}' with bbox." \
        "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags." \
        "Output the one bbox in JSON format." \
        "i.e., <think> thinking process here </think>" \
        "<answer>{Answer}</answer>"

    image = PILImage.open(args.image_path).convert("RGB")

    # 构造输入
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                 # {"type": "text", "text": QUESTION_TEMPLATE.format(Question=args.text.strip("."))}
                {"text": QUESTION_TEMPLATE.format(Question=args.text.lower().strip("."), 
                                                Answer="{'bbox': [10,100,200,210]")}
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

    # 推理
    generated_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
    generated_ids_trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]

    print("\n=== Model Output ===")
    print(output_text)

    # bbox = parse_bbox(output_text)

    bbox = extract_bbox_from_json(output_text)

    if bbox:
        print(f"Predicted BBox: {bbox}")
        draw_and_save_bbox(args.image_path, bbox, args.save_dir)
    else:
        print("⚠️ No valid bbox detected in output.")


if __name__ == "__main__":
    main()
