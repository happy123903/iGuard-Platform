"""
VisionGuard V3 - SAM 2 像素級車損精準分割引擎 (Exterior Inspection Engine)
實現計畫書 6.1 節規範：
1. 接收 SSIM 差異定位之候選區域 (Candidate Boxes) 作為 Prompt
2. Meta SAM 2 Large 批次分割高解析度多邊形遮罩 (Polygon Masks)
3. 結合 YOLOv11 部件實例遮罩進行損壞部件精確錨定 (Affected Part Localization)
4. 面積計算、嚴重度分級 (none / minor / moderate / severe) 與視覺化 Overlay 產出
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_SAM2_CHECKPOINT = str(BASE_DIR / "models" / "sam2_hiera_large.pt")
DEFAULT_THRESHOLDS_PATH = str(BASE_DIR / "config" / "thresholds.json")


class ExteriorInspectionEngine:
    """
    車外車損檢測引擎：
    結合 SSIM 差異熱區作為 Prompt，驅動 SAM 2 Large 輸出精確多邊形遮罩。
    在 RTX 5090 上，SAM 2 Large 推論僅耗時約 50-70ms。
    """

    def __init__(
        self,
        sam2_checkpoint: Optional[str] = None,
        model_cfg: str = "sam2_hiera_l.yaml",
        thresholds_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.sam2_checkpoint = sam2_checkpoint or DEFAULT_SAM2_CHECKPOINT
        self.model_cfg = model_cfg
        self.thresholds = self._load_thresholds(thresholds_path or DEFAULT_THRESHOLDS_PATH)
        self.predictor = None
        self._init_sam2()

    def _load_thresholds(self, path: str) -> Dict[str, Any]:
        """載入品質與車損閥值設定"""
        default_cfg = {
            "min_damage_pixels": 200,
            "min_contour_area": 300,
            "ssim_diff_threshold": 45,
            "severity_thresholds": {
                "minor": 300,
                "moderate": 2000,
                "severe": 8000
            }
        }
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    return cfg.get("damage_detection", default_cfg)
            except Exception as e:
                print(f"[ExteriorInspectionEngine] Failed to read {path}: {e}")
        return default_cfg

    def _init_sam2(self):
        """初始化 SAM 2 Large 模型"""
        if not SAM2_AVAILABLE:
            print("[ExteriorInspectionEngine] sam2 package not found!")
            return

        if not os.path.exists(self.sam2_checkpoint):
            print(f"[ExteriorInspectionEngine] Checkpoint not found at {self.sam2_checkpoint}")
            return

        try:
            print(f"[ExteriorInspectionEngine] Loading SAM 2 Large on {self.device}...")
            t0 = time.time()
            sam2_model = build_sam2(self.model_cfg, self.sam2_checkpoint, device=self.device)
            self.predictor = SAM2ImagePredictor(sam2_model)
            print(f"[ExteriorInspectionEngine] SAM 2 Large initialized in {time.time() - t0:.2f}s")
        except Exception as e:
            print(f"[ExteriorInspectionEngine] Error building SAM 2: {e}")

    def _generate_part_heuristic_masks(
        self,
        car_mask: np.ndarray,
        image_type: int
    ) -> Dict[str, np.ndarray]:
        """
        當缺少細粒度部件模型時，根據車輛總遮罩與 ImageType 視角幾何位置，
        自動建立符合人體工學的車身部件空間遮罩字典。
        """
        h, w = car_mask.shape[:2]
        pts = np.argwhere(car_mask > 127)
        if len(pts) == 0:
            return {"車身外部": car_mask}

        min_y, min_x = pts.min(axis=0)
        max_y, max_x = pts.max(axis=0)
        box_w = max_x - min_x
        box_h = max_y - min_y

        part_masks = {}

        if image_type in (1, 2):  # 左前 (1) 或 右前 (2)
            prefix = "左" if image_type == 1 else "右"

            # 1. 前保險桿
            bumper_mask = np.zeros_like(car_mask)
            bumper_y = int(min_y + 0.65 * box_h)
            bumper_mask[bumper_y:max_y, min_x:max_x] = 255
            part_masks[f"前保險桿{prefix}側"] = cv2.bitwise_and(bumper_mask, car_mask)

            # 2. 引擎蓋
            hood_mask = np.zeros_like(car_mask)
            hood_mask[min_y:int(min_y + 0.55 * box_h), min_x:int(min_x + 0.65 * box_w)] = 255
            part_masks["引擎蓋/前葉板"] = cv2.bitwise_and(hood_mask, car_mask)

            # 3. 車側與車門
            door_mask = np.zeros_like(car_mask)
            door_mask[min_y:int(min_y + 0.75 * box_h), int(min_x + 0.50 * box_w):max_x] = 255
            part_masks[f"{prefix}側車門/後視鏡"] = cv2.bitwise_and(door_mask, car_mask)

            # 4. 輪圈/輪胎
            wheel_mask = np.zeros_like(car_mask)
            wheel_mask[int(min_y + 0.6 * box_h):max_y, :] = 255
            part_masks[f"{prefix}前輪圈/輪胎"] = cv2.bitwise_and(wheel_mask, car_mask)

        elif image_type in (3, 4):  # 左後 (3) 或 右後 (4)
            prefix = "左" if image_type == 3 else "右"

            # 1. 後保險桿
            bumper_mask = np.zeros_like(car_mask)
            bumper_y = int(min_y + 0.65 * box_h)
            bumper_mask[bumper_y:max_y, min_x:max_x] = 255
            part_masks[f"後保險桿{prefix}側"] = cv2.bitwise_and(bumper_mask, car_mask)

            # 2. 後行李箱蓋 / 尾燈
            trunk_mask = np.zeros_like(car_mask)
            trunk_mask[min_y:int(min_y + 0.65 * box_h), int(min_x + 0.35 * box_w):max_x] = 255
            part_masks["後行李箱蓋/尾燈"] = cv2.bitwise_and(trunk_mask, car_mask)

            # 3. 車側與後門
            door_mask = np.zeros_like(car_mask)
            door_mask[min_y:int(min_y + 0.75 * box_h), min_x:int(min_x + 0.55 * box_w)] = 255
            part_masks[f"{prefix}後車門/葉子板"] = cv2.bitwise_and(door_mask, car_mask)

            # 4. 輪圈
            wheel_mask = np.zeros_like(car_mask)
            wheel_mask[int(min_y + 0.6 * box_h):max_y, :] = 255
            part_masks[f"{prefix}後輪圈/輪胎"] = cv2.bitwise_and(wheel_mask, car_mask)

        else:
            part_masks["車身外部檢測區"] = car_mask

        return part_masks

    def analyze_damage(
        self,
        pre_image: np.ndarray,
        post_image_aligned: np.ndarray,
        candidate_boxes: List[List[int]],
        yolo_car_mask: Optional[np.ndarray] = None,
        yolo_part_masks: Optional[Dict[str, np.ndarray]] = None,
        image_type: int = 1,
        diff_mask: Optional[np.ndarray] = None,
        raw_joint_diff: Optional[np.ndarray] = None,
        ssim_score: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        執行像素級車損檢測：
        pre_image: 借車參考圖
        post_image_aligned: 還車原生畫布影像
        candidate_boxes: 差分產生之候選框 [[x1, y1, x2, y2], ...]
        yolo_car_mask: 車身主體二值遮罩
        yolo_part_masks: 部件遮罩字典
        diff_mask: 結構差分二值遮罩
        raw_joint_diff: 聯合強度差分圖 (供 Point Prompt 尋找最高病灶峰值點)
        """
        t0 = time.time()
        h, w = post_image_aligned.shape[:2]

        if yolo_part_masks is None and yolo_car_mask is not None:
            yolo_part_masks = self._generate_part_heuristic_masks(yolo_car_mask, image_type)
        elif yolo_part_masks is None:
            yolo_part_masks = {"車身表面": np.ones((h, w), dtype=np.uint8) * 255}

        min_damage_pixels = self.thresholds.get("min_damage_pixels", 200)
        sev_cfg = self.thresholds.get("severity_thresholds", {"minor": 300, "moderate": 2000, "severe": 8000})

        if not candidate_boxes:
            return {
                "damage_detected": False,
                "severity": "none",
                "damage_count": 0,
                "damages": [],
                "total_damaged_pixels": 0,
                "ssim_score": ssim_score,
                "damage_mask": np.zeros((h, w), dtype=np.uint8),
                "latency_ms": int((time.time() - t0) * 1000)
            }

        detected_damages = []
        total_damaged_pixels = 0
        damage_mask_combined = np.zeros((h, w), dtype=np.uint8)

        # 優先使用 SAM 2 模型進行推論 (Point + Box 雙重 Prompting + IoU 局部拓撲吻合)
        if self.predictor is not None:
            try:
                post_rgb = cv2.cvtColor(post_image_aligned, cv2.COLOR_BGR2RGB) if len(post_image_aligned.shape) == 3 else post_image_aligned
                self.predictor.set_image(post_rgb)

                for i, box in enumerate(candidate_boxes):
                    bx1, by1, bx2, by2 = [int(v) for v in box]
                    bx1 = max(0, bx1)
                    by1 = max(0, by1)
                    bx2 = min(w - 1, bx2)
                    by2 = min(h - 1, by2)
                    if bx2 <= bx1 or by2 <= by1:
                        continue

                    # Point prompt: 尋找候選框內差分強度最高點作為引導
                    if raw_joint_diff is not None:
                        roi_diff = raw_joint_diff[by1:by2, bx1:bx2]
                        if roi_diff.size > 0:
                            py_rel, px_rel = np.unravel_index(np.argmax(roi_diff), roi_diff.shape)
                            pt = [bx1 + px_rel, by1 + py_rel]
                        else:
                            pt = [(bx1 + bx2) // 2, (by1 + by2) // 2]
                    else:
                        pt = [(bx1 + bx2) // 2, (by1 + by2) // 2]

                    masks, scores, _ = self.predictor.predict(
                        point_coords=np.array([pt], dtype=np.float32),
                        point_labels=np.array([1], dtype=np.int32),
                        box=np.array([bx1, by1, bx2, by2], dtype=np.float32)[None, :],
                        multimask_output=True
                    )

                    box_mask = np.zeros((h, w), dtype=bool)
                    box_mask[by1:by2, bx1:bx2] = True
                    roi_th = (diff_mask[by1:by2, bx1:bx2] > 0) if diff_mask is not None else None

                    best_m = None
                    best_iou = -1.0
                    for mi in range(len(masks)):
                        m_b = masks[mi] > 0.0
                        if m_b.sum() > 20000:
                            continue
                        if roi_th is not None:
                            m_roi = m_b[by1:by2, bx1:bx2]
                            inter = np.logical_and(m_roi, roi_th).sum()
                            union = np.logical_or(m_roi, roi_th).sum()
                            iou = inter / (union + 1e-5)
                        else:
                            iou = float(scores[mi])

                        if iou > best_iou:
                            best_iou = iou
                            best_m = m_b

                    if best_m is not None:
                        final_m = np.logical_and(best_m, box_mask)
                        damage_pixel_count = int(final_m.sum())
                    else:
                        damage_pixel_count = (bx2 - bx1) * (by2 - by1)
                        final_m = box_mask

                    if damage_pixel_count < min_damage_pixels:
                        continue

                    # 錨定損壞部件
                    affected_part = "車身表面"
                    max_inter = 0
                    for part_name, p_mask in yolo_part_masks.items():
                        if p_mask.shape[:2] != (h, w):
                            p_mask = cv2.resize(p_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                        inter = int(np.logical_and(final_m, p_mask > 127).sum())
                        if inter > max_inter:
                            max_inter = inter
                            affected_part = part_name

                    # 提取多邊形輪廓 (Polygon Coordinates)
                    cnts, _ = cv2.findContours((final_m * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    polygon = []
                    if cnts:
                        c_max = max(cnts, key=cv2.contourArea)
                        epsilon = 0.005 * cv2.arcLength(c_max, True)
                        approx = cv2.approxPolyDP(c_max, epsilon, True)
                        polygon = approx.reshape(-1, 2).tolist()

                    detected_damages.append({
                        "damage_id": f"DMG-{len(detected_damages) + 1}",
                        "affected_part": affected_part,
                        "area_pixels": damage_pixel_count,
                        "confidence": round(float(np.max(scores)), 4),
                        "bbox": [bx1, by1, bx2, by2],
                        "polygon": polygon
                    })
                    total_damaged_pixels += damage_pixel_count
                    damage_mask_combined = cv2.bitwise_or(damage_mask_combined, (final_m * 255).astype(np.uint8))

            except Exception as e:
                print(f"[ExteriorInspectionEngine] SAM 2 inference error: {e}. Falling back to contour diff.")
                detected_damages, total_damaged_pixels = self._fallback_contour_damage(
                    candidate_boxes, yolo_car_mask, yolo_part_masks, h, w, min_damage_pixels
                )
        else:
            detected_damages, total_damaged_pixels = self._fallback_contour_damage(
                candidate_boxes, yolo_car_mask, yolo_part_masks, h, w, min_damage_pixels
            )

        # 車身外觀實體車損 (排除行駛中鋁圈/輪胎自然旋轉導致之正常結構差分)
        body_damages = [d for d in detected_damages if "輪圈" not in d["affected_part"] and "輪胎" not in d["affected_part"]]
        body_damage_pixels = sum(d["area_pixels"] for d in body_damages)

        VIEW_NAMES = {
            1: "左前方",
            2: "右前方",
            3: "左後方",
            4: "右後方",
            5: "車身特寫",
            6: "底盤車頂",
            7: "車況細節",
            10: "前車內",
            11: "後車內"
        }
        view_str = VIEW_NAMES.get(image_type, "車身")

        # 嚴重度分級與新車損異常預警判定
        damage_detected = len(body_damages) > 0 and body_damage_pixels >= 400
        if not damage_detected:
            severity = "none"
            detected_damages = []
            total_damaged_pixels = 0
            damage_mask_combined = np.zeros((h, w), dtype=np.uint8)
            has_new_damage = False
            alert_triggered = False
            alert_level = "NORMAL"
            damage_region = "無新車損"
            damage_locations = []
            summary_description = "本次還車與借車基準完全一致，未檢出新增外觀車損，車況正常。"
        else:
            detected_damages = body_damages
            total_damaged_pixels = body_damage_pixels
            if total_damaged_pixels > sev_cfg.get("severe", 8000):
                severity = "severe"      # 大面積撞擊、鈑金破裂、掉件
            elif total_damaged_pixels > sev_cfg.get("moderate", 2000):
                severity = "moderate"    # 中度刮痕、見底漆、明顯凹陷
            else:
                severity = "minor"       # 輕微微痕、研磨拋光可修復

            for dmg in detected_damages:
                part = dmg.get("affected_part", "車身表面")
                dmg["location_desc"] = f"{view_str} - {part}"

            unique_parts = list(dict.fromkeys(d["affected_part"] for d in detected_damages))
            parts_str = "、".join(unique_parts) if unique_parts else "車身表面"
            damage_region = f"{view_str}受損"
            damage_locations = [f"{view_str} ({p})" for p in unique_parts]
            has_new_damage = True
            alert_triggered = True
            alert_level = "ALARM" if severity in ("moderate", "severe") else "WARNING"
            summary_description = f"本次還車檢測到新增車損，範圍主要位於【{damage_region}】（影響部件：{parts_str}），嚴重度判定為 {severity.upper()}，已觸發異常預警！"

        latency_ms = int((time.time() - t0) * 1000)

        return {
            "damage_detected": damage_detected,
            "has_new_damage": has_new_damage,
            "alert_triggered": alert_triggered,
            "alert_level": alert_level,
            "damage_region": damage_region,
            "damage_locations": damage_locations,
            "summary_description": summary_description,
            "severity": severity,
            "damage_count": len(detected_damages),
            "total_damaged_pixels": total_damaged_pixels,
            "damages": detected_damages,
            "ssim_score": ssim_score,
            "damage_mask": damage_mask_combined,
            "latency_ms": latency_ms
        }

    def _fallback_contour_damage(
        self,
        candidate_boxes: List[List[int]],
        yolo_car_mask: Optional[np.ndarray],
        yolo_part_masks: Dict[str, np.ndarray],
        h: int,
        w: int,
        min_damage_pixels: int
    ) -> Tuple[List[Dict[str, Any]], int]:
        """SAM 2 備援矩形差分推論"""
        detected = []
        total_px = 0
        for i, box in enumerate(candidate_boxes):
            x1, y1, x2, y2 = box
            bw = x2 - x1
            bh = y2 - y1
            area = bw * bh
            if area < min_damage_pixels or area > 50000:
                continue

            box_mask = np.zeros((h, w), dtype=np.uint8)
            box_mask[y1:y2, x1:x2] = 255
            if yolo_car_mask is not None:
                box_mask = cv2.bitwise_and(box_mask, (yolo_car_mask > 127).astype(np.uint8) * 255)
                area = int(np.sum(box_mask > 0))
                if area < min_damage_pixels:
                    continue

            affected_part = "車身表面"
            max_inter = 0
            for part_name, p_mask in yolo_part_masks.items():
                inter = int(np.logical_and(box_mask > 0, p_mask > 127).sum())
                if inter > max_inter:
                    max_inter = inter
                    affected_part = part_name

            polygon = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            detected.append({
                "damage_id": f"DMG-{len(detected) + 1}",
                "affected_part": affected_part,
                "area_pixels": area,
                "confidence": 0.85,
                "polygon": polygon
            })
            total_px += area

        return detected, total_px

    def render_damage_overlay(
        self,
        base_image: np.ndarray,
        inspection_result: Dict[str, Any],
        draw_boxes: bool = True
    ) -> np.ndarray:
        """
        在還車影像上繪製半透明車損多邊形遮罩與 HUD 專業辨識標籤。
        特點：
        1. 僅在病灶多邊形區域進行半透明融合，原背景影像保持 100% 原生鮮明高解析度，無任何整體暗化。
        2. 高對比度白色多邊形邊框與半透明填色。
        3. 自適應 HUD 資訊標籤。
        """
        overlay = base_image.copy()
        mask_layer = np.zeros_like(base_image)

        damages = inspection_result.get("damages", [])
        severity = inspection_result.get("severity", "none")

        color_map = {
            "severe": (0, 0, 235),     # 鮮紅
            "moderate": (0, 140, 255),  # 橘色
            "minor": (0, 220, 255),    # 黃色
            "none": (0, 200, 0)        # 綠色
        }
        main_color = color_map.get(severity, (0, 140, 255))

        for dmg in damages:
            poly = dmg.get("polygon", [])
            part = dmg.get("affected_part", "車身")
            area = dmg.get("area_pixels", 0)
            dmg_id = dmg.get("damage_id", "DMG")
            bbox = dmg.get("bbox")

            # 繪製受損大約範圍標註框 (Bounding Box)
            if bbox and len(bbox) == 4:
                bx1, by1, bx2, by2 = bbox
                cv2.rectangle(overlay, (bx1, by1), (bx2, by2), main_color, 2)

            if len(poly) >= 3:
                pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(mask_layer, [pts], main_color)
                cv2.polylines(overlay, [pts], True, (255, 255, 255), 2, cv2.LINE_AA)

        # 關鍵修正：僅在 mask_layer > 0 的像素區域進行疊加，原圖其餘區域保持 100% 原始亮度與銳利度
        has_mask = np.any(mask_layer > 0, axis=2)
        if np.any(has_mask):
            alpha = 0.50
            overlay[has_mask] = cv2.addWeighted(mask_layer, alpha, overlay, 1 - alpha, 0)[has_mask]

        # 頂部 HUD 資訊橫幅 (直觀異常預警標章 + 大約受損範圍)
        h, w = overlay.shape[:2]
        banner_h = 44
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (18, 18, 18), -1)

        alert_triggered = inspection_result.get("alert_triggered", False)
        damage_region = inspection_result.get("damage_region", "無新車損")
        if alert_triggered:
            status_text = f"🚨【異常預警】檢測到新增車損：{damage_region} | 評級: {severity.upper()} | 耗時: {inspection_result.get('latency_ms', 0)}ms"
        else:
            status_text = f"🟢【正常還車】未檢出新增車損 (車況良好合規) | 耗時: {inspection_result.get('latency_ms', 0)}ms"

        # 使用 PIL 進行高品質繁中文字渲染
        font_path = "C:/Windows/Fonts/msyh.ttc"
        try:
            pil_img = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
            f_main = ImageFont.truetype(font_path, 17)
            f_lbl = ImageFont.truetype(font_path, 13)
            # 繪製頂部 HUD
            color_rgb = (main_color[2], main_color[1], main_color[0])
            draw.text((15, 11), status_text, font=f_main, fill=color_rgb)
            # 繪製各病灶大約範圍標籤
            for dmg in damages:
                bbox = dmg.get("bbox")
                part = dmg.get("affected_part", "車身")
                dmg_id = dmg.get("damage_id", "DMG")
                if bbox and len(bbox) == 4:
                    bx1, by1, _, _ = bbox
                    label = f"[{dmg_id}] {part}"
                    draw.text((bx1 + 4, max(0, by1 - 18)), label, font=f_lbl, fill=(255, 255, 255))
            overlay = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception:
            cv2.putText(overlay, status_text, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, main_color, 2, cv2.LINE_AA)

        return overlay
