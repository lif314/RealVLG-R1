import cv2
import numpy as np
import torch
from PIL import Image as PILImage
import torchvision.transforms.functional as TF
import re

from typing import Optional, List, Tuple, Any
import numpy as np
_EPS = np.spacing(1)    # the different implementation of epsilon (extreme min value) between numpy and matlab
_TYPE = np.float64
RESIZE_SIZE = (768, 768)

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import json

def resize_image_and_mask(example: dict) -> dict:
    """Resize input image and corresponding mask to specified dimensions.
    
    Args:
        example: Dictionary containing 'image' and 'gt_path' keys
        
    Returns:
        Dictionary with resized 'image' (PIL Image) and 'gt_mask' (numpy array)
    """
    image = example["image"].convert("RGB")
    image_resized = TF.resize(image, RESIZE_SIZE)

    # Load and process mask
    mask = PILImage.open(example["gt_path"]).convert("L")
    mask_resized = TF.resize(mask, RESIZE_SIZE)
    gt_mask_np = (np.array(mask_resized) > 127).astype(np.uint8)  # Binarize mask

    return {
        "image": image_resized,
        "gt_mask": gt_mask_np
    }

class SAMWrapper:
    
    def __init__(self, model_path: str, device: Optional[str] = None):
        """Initialize SAM2 model and predictor.
        
        Args:
            model_path: Path to SAM2 model checkpoint
            device: Device to run model on (e.g. "cuda", "cuda:0", "cpu"). 
                   If None, will auto-detect available device.
        """
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        self.model = build_sam2(model_cfg, model_path)
        
        # Device configuration
        self.device = torch.device(
            device if device else 
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = self.model.to(self.device)
        
        self.predictor = SAM2ImagePredictor(self.model)
        self.last_mask = None
        
    def predict(self, 
               image: PILImage.Image, 
               bbox: List[Tuple[int, int]]) -> Tuple[np.ndarray, float]:
        """Run segmentation prediction with given prompts.
        
        Args:
            image: Input PIL Image
            bbox: List of (x1, y1, x2, y2) coordinate points
            labels: List of point labels (1=foreground, 0=background)
            
        Returns:
            Tuple of (predicted_mask, confidence_score)
        """
        # Convert and preprocess image
        image_np = np.array(image)
        rgb_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        
        # Run prediction
        self.predictor.set_image(rgb_image)
        mask_pred, score, _ = self.predictor.predict(
            box=bbox,
            multimask_output=False,
        )
        
        self.last_mask = mask_pred[0]
        return mask_pred[0], score[0]
    
class Smeasure(object):
    def __init__(self, alpha: float = 0.5):
        """
        S-measure(Structure-measure) of SOD.
        ::
            @inproceedings{Smeasure,
                title={Structure-measure: A new way to eval foreground maps},
                author={Fan, Deng-Ping and Cheng, Ming-Ming and Liu, Yun and Li, Tao and Borji, Ali},
                booktitle=ICCV,
                pages={4548--4557},
                year={2017}
            }
        :param alpha: the weight for balancing the object score and the region score
        """
        self.sms = []
        self.alpha = alpha

    def step(self, pred: np.ndarray, gt: np.ndarray):
        pred, gt = _prepare_data(pred=pred, gt=gt)

        sm = self.cal_sm(pred, gt)
        self.sms.append(sm)

    def cal_sm(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """
        Calculate the S-measure.
        :return: s-measure
        """
        y = np.mean(gt)
        if y == 0:
            sm = 1 - np.mean(pred)
        elif y == 1:
            sm = np.mean(pred)
        else:
            sm = self.alpha * self.object(pred, gt) + (1 - self.alpha) * self.region(pred, gt)
            sm = max(0, sm)
        return sm

    def object(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """
        Calculate the object score.
        """
        fg = pred * gt
        bg = (1 - pred) * (1 - gt)
        u = np.mean(gt)
        object_score = u * self.s_object(fg, gt) + (1 - u) * self.s_object(bg, 1 - gt)
        return object_score

    def s_object(self, pred: np.ndarray, gt: np.ndarray) -> float:
        x = np.mean(pred[gt == 1])
        sigma_x = np.std(pred[gt == 1], ddof=1)
        score = 2 * x / (np.power(x, 2) + 1 + sigma_x + _EPS)
        return score

    def region(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """
        Calculate the region score.
        """
        x, y = self.centroid(gt)
        part_info = self.divide_with_xy(pred, gt, x, y)
        w1, w2, w3, w4 = part_info["weight"]
        # assert np.isclose(w1 + w2 + w3 + w4, 1), (w1 + w2 + w3 + w4, pred.mean(), gt.mean())

        pred1, pred2, pred3, pred4 = part_info["pred"]
        gt1, gt2, gt3, gt4 = part_info["gt"]
        score1 = self.ssim(pred1, gt1)
        score2 = self.ssim(pred2, gt2)
        score3 = self.ssim(pred3, gt3)
        score4 = self.ssim(pred4, gt4)

        return w1 * score1 + w2 * score2 + w3 * score3 + w4 * score4

    def centroid(self, matrix: np.ndarray) -> tuple:
        """
        To ensure consistency with the matlab code, one is added to the centroid coordinate,
        so there is no need to use the redundant addition operation when dividing the region later,
        because the sequence generated by ``1:X`` in matlab will contain ``X``.
        :param matrix: a bool data array
        :return: the centroid coordinate
        """
        h, w = matrix.shape
        area_object = np.count_nonzero(matrix)
        if area_object == 0:
            x = np.round(w / 2)
            y = np.round(h / 2)
        else:
            # More details can be found at: https://www.yuque.com/lart/blog/gpbigm
            y, x = np.argwhere(matrix).mean(axis=0).round()
        return int(x) + 1, int(y) + 1

    def divide_with_xy(self, pred: np.ndarray, gt: np.ndarray, x: int, y: int) -> dict:
        """
        Use (x,y) to divide the ``pred`` and the ``gt`` into four submatrices, respectively.
        """
        h, w = gt.shape
        area = h * w

        gt_LT = gt[0:y, 0:x]
        gt_RT = gt[0:y, x:w]
        gt_LB = gt[y:h, 0:x]
        gt_RB = gt[y:h, x:w]

        pred_LT = pred[0:y, 0:x]
        pred_RT = pred[0:y, x:w]
        pred_LB = pred[y:h, 0:x]
        pred_RB = pred[y:h, x:w]

        w1 = x * y / area
        w2 = y * (w - x) / area
        w3 = (h - y) * x / area
        w4 = 1 - w1 - w2 - w3

        return dict(
            gt=(gt_LT, gt_RT, gt_LB, gt_RB),
            pred=(pred_LT, pred_RT, pred_LB, pred_RB),
            weight=(w1, w2, w3, w4),
        )

    def ssim(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """
        Calculate the ssim score.
        """
        h, w = pred.shape
        N = h * w

        x = np.mean(pred)
        y = np.mean(gt)

        sigma_x = np.sum((pred - x) ** 2) / (N - 1)
        sigma_y = np.sum((gt - y) ** 2) / (N - 1)
        sigma_xy = np.sum((pred - x) * (gt - y)) / (N - 1)

        alpha = 4 * x * y * sigma_xy
        beta = (x ** 2 + y ** 2) * (sigma_x + sigma_y)

        if alpha != 0:
            score = alpha / (beta + _EPS)
        elif alpha == 0 and beta == 0:
            score = 1
        else:
            score = 0
        return score

    def get_results(self) -> dict:
        """
        Return the results about S-measure.
        :return: dict(sm=sm)
        """
        sm = np.mean(np.array(self.sms, dtype=_TYPE))
        return dict(sm=sm)


def _prepare_data(pred: np.ndarray, gt: np.ndarray) -> tuple:
    """
    A numpy-based function for preparing ``pred`` and ``gt``.
    - for ``pred``, it looks like ``mapminmax(im2double(...))`` of matlab;
    - ``gt`` will be binarized by 128.
    :param pred: prediction
    :param gt: mask
    :return: pred, gt
    """
    gt = gt > 128
    # im2double, mapminmax
    pred = pred / 255
    if pred.max() != pred.min():
        pred = (pred - pred.min()) / (pred.max() - pred.min())
    return pred, gt


def segmentation_reward(
    completions: List[str],
    gt_mask: List[np.ndarray],
    image: List[PILImage.Image],
    **kwargs
) -> List[float]:
    """Calculate segmentation quality reward.
    
    Args:
        completions: List of model completion strings
        gt_mask: List of ground truth masks
        image: List of input images
        
    Returns:
        List of reward scores combining IOU and S-measure metrics
    """
    rewards = []
    sm_calculator = Smeasure(alpha=kwargs.get('s_measure_alpha', 0.5))
    sam_config = kwargs.get("sam_config")[0]
    sam = SAMWrapper(
            model_path=sam_config["model_path"],
            device=sam_config["device"]
        )
    
    for completion, gt_mask, img in zip(completions, gt_mask, image):
        content = completion[0]["content"]
        points, labels = parse_custom_format(content)
        # print(f"Parsed points: {points}, labels: {labels}")

        iou_reward = 0.0
        sm_reward = 0.0
        # point_penalty = 0.0
        
        if points is not None and len(points) > 0:

            if not isinstance(img, PILImage.Image):
                img = PILImage.fromarray(img)
            
            mask_pred, score = sam.predict(img, points.tolist(), labels.tolist())
            # pred_mask = PILImage.fromarray((mask_pred * 255).astype(np.uint8))
            # pred_mask_path = "pred_mask.png"
            # pred_mask.save(pred_mask_path)            
            intersection = np.logical_and(mask_pred, gt_mask).sum()
            union = np.logical_or(mask_pred, gt_mask).sum()
            iou_reward = intersection / union if union > 0 else 0.0

            mask_pred = np.array(mask_pred).astype(_TYPE)
            gt_np = np.array(gt_mask).astype(_TYPE)
            mask_pred, gt_np = _prepare_data(mask_pred, gt_np)
            
            sm = sm_calculator.cal_sm(mask_pred, gt_np)
            sm_reward = max(0.0, min(1.0, sm))  
        total_reward = 0.7*iou_reward  + 0.3*sm_reward 
        rewards.append(total_reward)
        
        if os.getenv("DEBUG") == "1":
            log_str = f"""
            Content: {content}
            Points: {points}
            Labels: {labels}
            IOU: {iou_reward:.2f}
            Final Reward: {total_reward:.2f}
            {'-'*40}
            """
            with open("training.log", "a") as f:
                f.write(log_str)
    
    return rewards

def format_reward(response: str) -> float:
    re_str = r"<think>.*?</think>\s*<answer>\s*\(\s*\d+\s*,\s*\d+\s*\)\s*,\s*\(\s*\d+\s*,\s*\d+\s*\)\s*</answer>"
    pattern = re.compile(re_str, re.DOTALL)
    format_match = re.fullmatch(pattern, response)
    return 1.0 if format_match else 0.0


def parse_format(response: str):
    ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
    COORD_PATTERN = re.compile(
        r"\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*,\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)"
    )

    def parse_bbox(coord_str: str):
        match = COORD_PATTERN.fullmatch(coord_str.strip())
        if not match:
            raise ValueError(f"Invalid bbox format: {coord_str}")
        x1, y1, x2, y2 = list(map(int, match.groups()))
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid bbox (x2<=x1 or y2<=y1): {coord_str}")
        return x1, y1, x2, y2
    
    # Extract <answer> content
    answer_match = ANSWER_PATTERN.search(response)
    if not answer_match:
        return 0.0

    answer_str = answer_match.group(1).strip()
    x1, y1, x2, y2 = parse_bbox(answer_str)
    pred_box = [x1, y1, x2, y2]

    return pred_box

def accuracy_reward(predict: str, ground_truth: str) -> float:
    """
    Compute IoU between predicted bbox and ground-truth bbox.
    predict: model output like "<think>...</think><answer>(x1, y1), (x2, y2)</answer>"
    ground_truth: JSON string, e.g. "[x1, y1, x2, y2]"
    """

    ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
    COORD_PATTERN = re.compile(
        r"\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*,\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)"
    )

    def parse_bbox(coord_str: str):
        match = COORD_PATTERN.fullmatch(coord_str.strip())
        if not match:
            raise ValueError(f"Invalid bbox format: {coord_str}")
        x1, y1, x2, y2 = list(map(int, match.groups()))
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid bbox (x2<=x1 or y2<=y1): {coord_str}")
        return x1, y1, x2, y2

    def bbox_iou(boxA, boxB):
        """Compute IoU between two rectangular bboxes."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        inter_w = max(0, xB - xA)
        inter_h = max(0, yB - yA)
        inter_area = inter_w * inter_h

        boxA_area = max(0, (boxA[2] - boxA[0])) * max(0, (boxA[3] - boxA[1]))
        boxB_area = max(0, (boxB[2] - boxB[0])) * max(0, (boxB[3] - boxB[1]))

        union_area = boxA_area + boxB_area - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    try:
        # Extract <answer> content
        answer_match = ANSWER_PATTERN.search(predict)
        if not answer_match:
            return 0.0

        answer_str = answer_match.group(1).strip()
        x1, y1, x2, y2 = parse_bbox(answer_str)
        pred_box = (x1, y1, x2, y2)

        # Parse GT bbox
        gt_box = json.loads(ground_truth)
        if not isinstance(gt_box, list) or len(gt_box) != 4:
            return 0.0

        return bbox_iou(pred_box, gt_box)

    except Exception:
        return 0.0

def compute_score(reward_inputs: list[dict[str, Any]], format_weight: float = 0.1) -> list[dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")

    scores = []
    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])  # handle qwen2.5vl-32b format
        format_score = format_reward(response)
        accuracy_score = accuracy_reward(response, reward_input["ground_truth"])
        scores.append(
            {
                "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
                "format": format_score,
                "accuracy": accuracy_score,
            }
        )

    return scores