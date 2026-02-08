import json
import os

def update_validity_rate(json_path):
    """
    从已保存的 bbox 评估 JSON 中重新计算 Validity Rate
    并写回 JSON 顶层，排在第二个键位置
    """
    if not os.path.exists(json_path):
        print(f"文件不存在: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", [])
    if not results:
        print(f"{json_path} 中没有结果数据")
        return

    # 删除 results 中可能存在的 Validity_Rate
    for r in results:
        r.pop("Validity_Rate", None)

    # 计算 VR
    num_valid = sum(1 for r in results if r.get("pred_bbox") is not None) # bbox sam
    # num_valid = sum(1 for r in results if r.get("iou") is not None) # grasp contact
    validity_rate = num_valid / len(results)

    # 构造新的有序字典，将 Validity_Rate 排在第二位
    new_data = {}
    keys = list(data.keys())
    if "split" in keys:
        new_data["split"] = data["split"]
        new_data["Validity_Rate"] = validity_rate
        for k in keys:
            if k not in ["split", "Validity_Rate"]:
                new_data[k] = data[k]
    else:
        # 如果没有 split，就直接放在最前面
        new_data["Validity_Rate"] = validity_rate
        for k in keys:
            if k != "Validity_Rate":
                new_data[k] = data[k]

    # 写回文件
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False)

    print(f"✅ 已更新 {json_path}，Validity_Rate = {validity_rate:.4f}")


# ============ Example Usage ============
if __name__ == "__main__":
    json_files = [
        "./outputs/evaluation/bbox/RealVLG-GRPO-3B/seen_bbox_eval.json",
        "./outputs/evaluation/bbox/RealVLG-GRPO-3B/similar_bbox_eval.json",
        "./outputs/evaluation/bbox/RealVLG-GRPO-3B/novel_bbox_eval.json",
    ]

    for jf in json_files:
        update_validity_rate(jf)
