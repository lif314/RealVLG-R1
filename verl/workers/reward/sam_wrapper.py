import numpy as np
from PIL import Image
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

class SAMWrapper:
    def __init__(self, model_path: str, device: str = None):
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_sam2(model_cfg, model_path, device=self.device)
        self.predictor = SAM2ImagePredictor(self.model)

    def set_image(self, image):
        if isinstance(image, str):
            rgb_image = np.array(Image.open(image).convert("RGB"), dtype=np.uint8)
        elif isinstance(image, Image.Image):
            rgb_image = np.array(image.convert("RGB"), dtype=np.uint8)
        elif isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
            rgb_image = image
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        self.predictor.set_image(rgb_image)

    def predict(self, bbox: list, multimask_output=False):
        mask_pred, score, _ = self.predictor.predict(
            box=np.array([bbox], dtype=np.float32),
            multimask_output=multimask_output
        )
        return mask_pred[0], score[0]
