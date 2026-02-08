import os
import json
from torch.utils.data import Dataset


class GraspNetVLGDataset(Dataset):
    """
    GraspNet Visual-Language Grounding Dataset Loader
    仅包含三个测试划分：
        - seen:    scene_0100 ~ scene_0119
        - similar: scene_0120 ~ scene_0149
        - novel:   scene_0150 ~ scene_0189
    每个场景仅加载第一帧 0000.json
    """

    def __init__(self, data_root, camera_mode="kinect", split="seen"):
        assert split in ["seen", "similar", "novel"], \
            f"❌ Invalid split: {split}, must be one of ['seen', 'similar', 'novel']"

        self.data_root = data_root
        self.metadata_dir = os.path.join(data_root, "metadata", camera_mode)
        assert os.path.exists(self.metadata_dir), f"❌ Metadata dir not found: {self.metadata_dir}"

        self.samples = []

        split_ranges = {
            "seen": range(100, 130),
            "similar": range(130, 160),
            "novel": range(160, 190),
        }
        scene_range = split_ranges[split]

        for scene_id in scene_range:
            scene_name = f"scene_{scene_id:04d}"
            scene_dir = os.path.join(self.metadata_dir, scene_name)
            if not os.path.exists(scene_dir):
                continue

            json_path = os.path.join(scene_dir, "0000.json")
            if not os.path.exists(json_path):
                continue

            with open(json_path, "r", encoding="utf-8") as f:
                objs = json.load(f)

            for obj in objs:
                grasps = obj.get("grasps", [])
                if not grasps:
                    continue

                image_path = os.path.join(self.data_root, obj.get("image_path"))
                mask_path = os.path.join(self.data_root, obj.get("mask_path", ""))

                self.samples.append({
                    "scene": scene_name,
                    "image_name": obj.get("image_name", ""),
                    "object_id": obj.get("object_id", ""),
                    "image_path": image_path,
                    "mask_path": mask_path,
                    "description": obj.get("description", ""),
                    "bbox": obj.get("bbox", []),
                    "grasps": grasps,
                    "contact_points": obj.get("contact_points", []),
                })

        print(f"✅ Loaded {len(self.samples)} samples for split='{split}'")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]