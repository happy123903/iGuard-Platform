"""
VisionGuard V3 — YOLOv11 智能構圖與視角驗證模組 (CompositionGuard)

解決痛點：
1. 拍攝距離偏差（太近局部裁切、太遠車身過小特徵丟失）
2. 拍錯車輛或鏡頭未對準車輛
3. 視角顛倒（如應拍車內卻拍了車外、應拍左前卻拍成右後）
4. 提供車輛分割遮罩（Mask），精準剔除背景雜訊，供後續車損辨識管線使用

硬體配置：RTX 5090, YOLOv11x-seg FP16, imgsz=1280, 延遲 < 25ms
"""

import json
import logging
from pathlib import Path
from typing import Optional, Union, Tuple
import cv2
import numpy as np
import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# 設定檔路徑
CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "thresholds.json"
DEFAULT_MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "yolo11x-seg.pt"

# COCO 類別對應：2: car, 3: motorcycle, 5: bus, 7: truck
VEHICLE_CLASS_IDS = {2: "car", 5: "bus", 7: "truck"}

# ImageType 定義
IMAGE_TYPE_NAMES = {
    1: "左前 (Front-Left)",
    2: "右前 (Front-Right)",
    3: "左後 (Rear-Left)",
    4: "右後 (Rear-Right)",
    5: "其他/損傷特寫",
    6: "其他/底盤車頂",
    7: "車況細節補充",
    8: "備份視角",
    9: "備份視角",
    10: "前車內 (Front Interior)",
    11: "後車內 (Rear Interior)",
}


def _load_composition_config() -> dict:
    """載入構圖閾值設定"""
    defaults = {
        "min_car_occupancy": 0.20,
        "max_car_occupancy": 0.95,
        "yolo_confidence": 0.15,
        "yolo_imgsz": 640
    }
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                return cfg.get("composition", defaults)
        except Exception as e:
            logger.warning(f"讀取構圖設定檔失敗: {e}，使用預設值")
    return defaults


class CompositionGuard:
    """
    YOLOv11 智能構圖與視角驗證守門員。
    使用 YOLOv11x-seg 進行車輛偵測、實例分割與構圖驗證。
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        conf_threshold: Optional[float] = None,
        imgsz: Optional[int] = None
    ):
        self.config = _load_composition_config()
        self.min_occupancy = self.config.get("min_car_occupancy", 0.20)
        self.max_occupancy = self.config.get("max_car_occupancy", 0.95)
        self.conf = conf_threshold or self.config.get("yolo_confidence", 0.15)
        self.imgsz = imgsz or self.config.get("yolo_imgsz", 640)

        # 決定運算裝置
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # 載入模型
        resolved_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        if not resolved_path.exists():
            logger.warning(f"模型檔案不存在: {resolved_path}，使用 yolo11x-seg.pt")
            resolved_path = "yolo11x-seg.pt"

        logger.info(f"正在載入 YOLOv11 模型: {resolved_path} (device={self.device})...")
        self.model = YOLO(str(resolved_path))
        self.model.to(self.device)
        logger.info("YOLOv11 構圖模型載入完成")

    def predict_image(self, image: np.ndarray, auto_rotate: bool = True):
        """執行 YOLO 推論並返回原始 results 物件，若未偵測到車輛則自動嘗試自適應暗部增強與旋轉校正"""
        res = self.model.predict(
            image,
            imgsz=self.imgsz,
            conf=self.conf,
            device=self.device,
            verbose=False
        )[0]

        has_car = any(int(b.cls[0].item()) in VEHICLE_CLASS_IDS for b in res.boxes) if res.boxes is not None else False
        rot_map = {90: cv2.ROTATE_90_CLOCKWISE, 270: cv2.ROTATE_90_COUNTERCLOCKWISE, 180: cv2.ROTATE_180}
        if not has_car and auto_rotate:
            for r, code in rot_map.items():
                cur = cv2.rotate(image, code)
                res_r = self.model.predict(cur, imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False)[0]
                if res_r.boxes is not None and any(int(b.cls[0].item()) in VEHICLE_CLASS_IDS for b in res_r.boxes):
                    return res_r

        # 若仍未檢出且影像較暗（例如夜間借還車死角），使用 HSV-CLAHE 提亮暗部邊緣後重新檢測
        if not has_car:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if np.mean(gray) < 60:
                try:
                    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
                    hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
                    brightened = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                    res_b = self.model.predict(brightened, imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False)[0]
                    if res_b.boxes is not None and any(int(b.cls[0].item()) in VEHICLE_CLASS_IDS for b in res_b.boxes):
                        return res_b
                    # 同步嘗試提亮後的旋轉
                    if auto_rotate:
                        for r, code in rot_map.items():
                            cur = cv2.rotate(brightened, code)
                            res_br = self.model.predict(cur, imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False)[0]
                            if res_br.boxes is not None and any(int(b.cls[0].item()) in VEHICLE_CLASS_IDS for b in res_br.boxes):
                                return res_br
                except Exception as e:
                    logger.debug(f"暗部強化檢測略過: {e}")

        return res

    def verify_viewpoint(
        self,
        image: np.ndarray,
        expected_image_type: int
    ) -> dict:
        """
        驗證影像拍攝構圖是否合規，並檢查是否符合預期視角 (ImageType 1-11)。

        Args:
            image: BGR 或 RGB 格式的 numpy array
            expected_image_type: 預期的 ImageType (1-4: 車外必拍, 5-9: 選拍/特寫, 10-11: 車內必拍)

        Returns:
            dict: {
                "passed": bool,
                "occupancy_ratio": float or None,
                "mask_occupancy_ratio": float or None,
                "message": str,
                "reason": str or None,
                "primary_box": list[float] or None,  # [x1, y1, x2, y2]
                "detected_vehicles": int,
                "is_centered": bool,
                "has_mask": bool,
                "latency_ms": float
            }
        """
        import time
        start_time = time.perf_counter()

        if image is None or image.size == 0:
            return {
                "passed": False,
                "occupancy_ratio": None,
                "mask_occupancy_ratio": None,
                "message": "影像數據為空",
                "reason": "傳入無效或空的影像數據",
                "primary_box": None,
                "detected_vehicles": 0,
                "is_centered": False,
                "has_mask": False,
                "latency_ms": 0.0
            }

        h, w = image.shape[:2]
        screen_area = float(h * w)

        # 執行 YOLO 推論
        results = self.predict_image(image)
        latency_ms = (time.perf_counter() - start_time) * 1000

        # 篩選車輛 (car, bus, truck)
        car_candidates = []
        if results.boxes is not None and len(results.boxes) > 0:
            for idx, box in enumerate(results.boxes):
                cls_id = int(box.cls[0].item())
                if cls_id in VEHICLE_CLASS_IDS:
                    conf = float(box.conf[0].item())
                    xyxy = [float(v) for v in box.xyxy[0].tolist()]
                    box_w = xyxy[2] - xyxy[0]
                    box_h = xyxy[3] - xyxy[1]
                    box_area = box_w * box_h
                    car_candidates.append({
                        "index": idx,
                        "class_name": VEHICLE_CLASS_IDS[cls_id],
                        "conf": conf,
                        "xyxy": xyxy,
                        "area": box_area,
                        "occupancy": box_area / screen_area
                    })

        # ============================================================
        # 情況 A：車外四視角 (ImageType 1: 左前, 2: 右前, 3: 左後, 4: 右後)
        # ============================================================
        if expected_image_type in (1, 2, 3, 4):
            if not car_candidates:
                return {
                    "passed": False,
                    "occupancy_ratio": 0.0,
                    "mask_occupancy_ratio": 0.0,
                    "message": "未偵測到車輛",
                    "reason": "畫面中未偵測到車輛主體，請將手機鏡頭對準租賃車輛",
                    "primary_box": None,
                    "detected_vehicles": 0,
                    "is_centered": False,
                    "has_mask": False,
                    "latency_ms": round(latency_ms, 1)
                }

            # 取面積與置信度綜合最高者為主車輛 (確保選中前景租賃車)
            primary_car = max(car_candidates, key=lambda c: c["area"] * (c["conf"] + 0.2))
            occupancy = round(primary_car["occupancy"], 3)
            xyxy = primary_car["xyxy"]

            # 計算分割遮罩面積比例
            mask_occupancy = None
            has_mask = False
            if results.masks is not None and len(results.masks) > primary_car["index"]:
                try:
                    mask_tensor = results.masks[primary_car["index"]].data[0]
                    mask_pixels = float(torch.sum(mask_tensor > 0.5).item())
                    orig_h, orig_w = mask_tensor.shape[:2]
                    mask_occupancy = round(mask_pixels / (orig_h * orig_w), 3)
                    has_mask = True
                except Exception as e:
                    logger.debug(f"計算遮罩面積異常: {e}")

            # 居中性檢查（車體中心是否落在畫面中段 10% ~ 90% 區間）
            center_x = (xyxy[0] + xyxy[2]) / 2.0 / w
            center_y = (xyxy[1] + xyxy[3]) / 2.0 / h
            is_centered = (0.10 <= center_x <= 0.90) and (0.10 <= center_y <= 0.90)

            # 佔比過小：拍太遠
            if occupancy < self.min_occupancy:
                return {
                    "passed": False,
                    "occupancy_ratio": occupancy,
                    "mask_occupancy_ratio": mask_occupancy,
                    "message": "距離過遠",
                    "reason": f"車身在畫面中佔比過小 ({occupancy:.0%})，請向前走 2-3 步拍攝全車外觀",
                    "primary_box": [round(v, 1) for v in xyxy],
                    "detected_vehicles": len(car_candidates),
                    "is_centered": is_centered,
                    "has_mask": has_mask,
                    "latency_ms": round(latency_ms, 1)
                }

            # 佔比過大：拍太近嚴重局部裁切（對於車外必拍角度）
            if occupancy > self.max_occupancy:
                return {
                    "passed": False,
                    "occupancy_ratio": occupancy,
                    "mask_occupancy_ratio": mask_occupancy,
                    "message": "距離過近",
                    "reason": f"距離車輛過近 ({occupancy:.0%})，導致車身邊界遭裁切，請退後拍攝完整視角",
                    "primary_box": [round(v, 1) for v in xyxy],
                    "detected_vehicles": len(car_candidates),
                    "is_centered": is_centered,
                    "has_mask": has_mask,
                    "latency_ms": round(latency_ms, 1)
                }

            # 邊緣嚴重切齊檢查（若頂邊、底邊、左右雙邊同時碰壁，可能取景不完整）
            edge_cut_count = 0
            if xyxy[0] <= 5: edge_cut_count += 1
            if xyxy[1] <= 5: edge_cut_count += 1
            if xyxy[2] >= w - 5: edge_cut_count += 1
            if xyxy[3] >= h - 5: edge_cut_count += 1

            if edge_cut_count >= 3 and occupancy > 0.94:
                return {
                    "passed": False,
                    "occupancy_ratio": occupancy,
                    "mask_occupancy_ratio": mask_occupancy,
                    "message": "構圖邊緣裁切過多",
                    "reason": "車輛多處邊緣超出畫面，請稍微退後將車身完整納入取景框",
                    "primary_box": [round(v, 1) for v in xyxy],
                    "detected_vehicles": len(car_candidates),
                    "is_centered": is_centered,
                    "has_mask": has_mask,
                    "latency_ms": round(latency_ms, 1)
                }

            # 車外構圖合規通過
            view_name = IMAGE_TYPE_NAMES.get(expected_image_type, f"視角 {expected_image_type}")
            return {
                "passed": True,
                "occupancy_ratio": occupancy,
                "mask_occupancy_ratio": mask_occupancy,
                "message": f"構圖優良 ({view_name})",
                "reason": None,
                "primary_box": [round(v, 1) for v in xyxy],
                "detected_vehicles": len(car_candidates),
                "is_centered": is_centered,
                "has_mask": has_mask,
                "latency_ms": round(latency_ms, 1)
            }

        # ============================================================
        # 情況 B：車內兩視角 (ImageType 10: 前車內, 11: 後車內)
        # ============================================================
        elif expected_image_type in (10, 11):
            view_name = IMAGE_TYPE_NAMES.get(expected_image_type, "車內照")

            # 若畫面中偵測到巨大的車外全車主體（佔比 > 0.35），表示使用者在車外拍錯上傳至車內欄位
            if car_candidates:
                primary_car = max(car_candidates, key=lambda c: c["area"])
                if primary_car["occupancy"] > 0.35:
                    return {
                        "passed": False,
                        "occupancy_ratio": round(primary_car["occupancy"], 3),
                        "mask_occupancy_ratio": None,
                        "message": "上傳視角錯誤 (應為車內照)",
                        "reason": f"目前為【{view_name}】拍照環節，畫面偵測為車外車身，請打開車門進入車內拍攝",
                        "primary_box": [round(v, 1) for v in primary_car["xyxy"]],
                        "detected_vehicles": len(car_candidates),
                        "is_centered": False,
                        "has_mask": False,
                        "latency_ms": round(latency_ms, 1)
                    }

            # 車內照無大面積車外車身干擾，構圖合格
            return {
                "passed": True,
                "occupancy_ratio": 0.0,
                "mask_occupancy_ratio": None,
                "message": f"車內構圖良好 ({view_name})",
                "reason": None,
                "primary_box": None,
                "detected_vehicles": len(car_candidates),
                "is_centered": True,
                "has_mask": False,
                "latency_ms": round(latency_ms, 1)
            }

        # ============================================================
        # 情況 C：其他選拍 / 車況局部特寫 (ImageType 5-9)
        # ============================================================
        else:
            view_name = IMAGE_TYPE_NAMES.get(expected_image_type, "車況特寫")
            primary_box = None
            occupancy = 0.0
            if car_candidates:
                primary_car = max(car_candidates, key=lambda c: c["area"])
                primary_box = [round(v, 1) for v in primary_car["xyxy"]]
                occupancy = round(primary_car["occupancy"], 3)

            return {
                "passed": True,
                "occupancy_ratio": occupancy,
                "mask_occupancy_ratio": None,
                "message": f"特寫/補充視角構圖通過 ({view_name})",
                "reason": None,
                "primary_box": primary_box,
                "detected_vehicles": len(car_candidates),
                "is_centered": True,
                "has_mask": results.masks is not None,
                "latency_ms": round(latency_ms, 1)
            }

    def get_vehicle_mask(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        獲取主車輛的二值化精確分割遮罩 (uint8, 0 或 255)。
        供後續環節二 SuperPoint 對齊與車損檢測去除背景干擾。

        Args:
            image: BGR 影像

        Returns:
            np.ndarray: 與原圖相同 HxW 大小的二值化遮罩，若無車輛則返回 None
        """
        h, w = image.shape[:2]
        results = self.predict_image(image)

        if results.masks is None or len(results.masks) == 0:
            return None

        # 找最大車輛
        best_idx = None
        max_area = 0
        if results.boxes is not None:
            for idx, box in enumerate(results.boxes):
                cls_id = int(box.cls[0].item())
                if cls_id in VEHICLE_CLASS_IDS:
                    xyxy = box.xyxy[0].tolist()
                    area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
                    if area > max_area:
                        max_area = area
                        best_idx = idx

        if best_idx is None:
            return None

        # 提取遮罩並 resize 回原圖尺寸
        mask_raw = results.masks[best_idx].data[0].cpu().numpy()
        mask_resized = cv2.resize(mask_raw, (w, h), interpolation=cv2.INTER_LINEAR)
        binary_mask = (mask_resized > 0.5).astype(np.uint8) * 255
        return binary_mask

    def crop_vehicle(
        self,
        image: np.ndarray,
        margin_ratio: float = 0.05
    ) -> Tuple[Optional[np.ndarray], Optional[list]]:
        """
        擷取主車輛的局部影像區域（含邊緣預留 margin），供車牌 OCR 或車損分析。

        Args:
            image: BGR 影像
            margin_ratio: 邊界擴展比例 (預設 5%)

        Returns:
            (cropped_image, bbox_coordinates): 裁切影像與 [x1, y1, x2, y2]
        """
        h, w = image.shape[:2]
        results = self.predict_image(image)

        if results.boxes is None or len(results.boxes) == 0:
            return None, None

        best_box = None
        max_area = 0
        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            if cls_id in VEHICLE_CLASS_IDS:
                xyxy = box.xyxy[0].tolist()
                area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
                if area > max_area:
                    max_area = area
                    best_box = xyxy

        if best_box is None:
            return None, None

        x1, y1, x2, y2 = best_box
        dx = (x2 - x1) * margin_ratio
        dy = (y2 - y1) * margin_ratio

        cx1 = max(0, int(x1 - dx))
        cy1 = max(0, int(y1 - dy))
        cx2 = min(w, int(x2 + dx))
        cy2 = min(h, int(y2 + dy))

        cropped = image[cy1:cy2, cx1:cx2]
        return cropped, [cx1, cy1, cx2, cy2]
