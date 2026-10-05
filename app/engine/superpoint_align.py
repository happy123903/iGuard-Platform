"""
VisionGuard V3 - SuperPoint + LightGlue 深度特徵配對與車身影像對齊模組
提供:
1. SuperPoint 關鍵點提取 + LightGlue 深度注意力特徵配對 (GPU 加速)
2. 透視變換 (Homography) / 仿射變換 (Affine) 穩健幾何對齊與畸變防禦
3. 車身亮度正規化 (Illumination Normalization) 消除光影天色差異
4. 高頻結構差分 (High-Frequency Structural Diff) + 輪廓邊界過濾 + 候選框 (Candidate Boxes) 生成
"""

import os
import cv2
import numpy as np
import torch
from typing import Dict, Any, Tuple, Optional, List
from skimage.metrics import structural_similarity as ssim

try:
    from lightglue import LightGlue, SuperPoint
    from lightglue.utils import rbd
    LIGHTGLUE_AVAILABLE = True
except ImportError:
    LIGHTGLUE_AVAILABLE = False


class SuperPointAligner:
    """
    SuperPoint + LightGlue 影像配對與幾何對齊引擎
    """

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu", max_keypoints: int = 2048):
        self.device = torch.device(device)
        self.max_keypoints = max_keypoints
        self.extractor = None
        self.matcher = None
        self._init_models()

    def _init_models(self):
        """初始化 SuperPoint 與 LightGlue 模型"""
        if LIGHTGLUE_AVAILABLE:
            try:
                self.extractor = SuperPoint(max_num_keypoints=self.max_keypoints).eval().to(self.device)
                self.matcher = LightGlue(features="superpoint").eval().to(self.device)
                print(f"[SuperPointAligner] Loaded SuperPoint + LightGlue on {self.device}")
            except Exception as e:
                print(f"[SuperPointAligner] Failed to load SuperPoint/LightGlue: {e}. Will fallback to SIFT/ORB.")
        else:
            print("[SuperPointAligner] LightGlue not available. Will fallback to SIFT/ORB.")

    def _preprocess_tensor(self, image: np.ndarray, max_dim: int = 1280) -> Tuple[torch.Tensor, float]:
        """將 OpenCV BGR 影像轉換為 LightGlue 所需的標準灰階 Tensor [1, 1, H, W]"""
        h, w = image.shape[:2]
        scale = 1.0
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            resized = image

        if len(resized.shape) == 3:
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        else:
            gray = resized

        tensor = torch.from_numpy(gray).float()[None, None] / 255.0
        return tensor.to(self.device), scale

    def match_features(self, img0: np.ndarray, img1: np.ndarray) -> Dict[str, Any]:
        """提取並配對兩張影像的特徵點"""
        if self.extractor is not None and self.matcher is not None:
            try:
                t0, s0 = self._preprocess_tensor(img0)
                t1, s1 = self._preprocess_tensor(img1)

                with torch.no_grad():
                    feats0 = self.extractor({"image": t0})
                    feats1 = self.extractor({"image": t1})
                    matches01 = self.matcher({"image0": feats0, "image1": feats1})

                    feats0, feats1, matches01 = [
                        rbd(x) for x in [feats0, feats1, matches01]
                    ]

                kpts0 = feats0["keypoints"].cpu().numpy() / s0
                kpts1 = feats1["keypoints"].cpu().numpy() / s1
                matches = matches01["matches"].cpu().numpy()
                scores = matches01["scores"].cpu().numpy()

                m_kpts0 = kpts0[matches[:, 0]]
                m_kpts1 = kpts1[matches[:, 1]]

                return {
                    "pts0": m_kpts0,
                    "pts1": m_kpts1,
                    "scores": scores,
                    "num_matches": len(m_kpts0),
                    "method": "superpoint_lightglue"
                }
            except Exception as e:
                print(f"[SuperPointAligner] SuperPoint matching error: {e}. Falling back to SIFT.")

        # Fallback to OpenCV SIFT
        return self._fallback_sift_match(img0, img1)

    def _fallback_sift_match(self, img0: np.ndarray, img1: np.ndarray) -> Dict[str, Any]:
        """OpenCV SIFT 備援特徵匹配"""
        sift = cv2.SIFT_create(nfeatures=self.max_keypoints)
        gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY) if len(img0.shape) == 3 else img0
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1

        kp0, des0 = sift.detectAndCompute(gray0, None)
        kp1, des1 = sift.detectAndCompute(gray1, None)

        if des0 is None or des1 is None or len(kp0) < 4 or len(kp1) < 4:
            return {"pts0": np.empty((0, 2)), "pts1": np.empty((0, 2)), "scores": np.empty((0,)), "num_matches": 0, "method": "sift_failed"}

        bf = cv2.BFMatcher(cv2.NORM_L2)
        raw_matches = bf.knnMatch(des0, des1, k=2)

        good_pts0, good_pts1, scores = [], [], []
        for m_n in raw_matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.75 * n.distance:
                    good_pts0.append(kp0[m.queryIdx].pt)
                    good_pts1.append(kp1[m.trainIdx].pt)
                    scores.append(1.0 - (m.distance / 100.0))

        return {
            "pts0": np.array(good_pts0, dtype=np.float32) if good_pts0 else np.empty((0, 2)),
            "pts1": np.array(good_pts1, dtype=np.float32) if good_pts1 else np.empty((0, 2)),
            "scores": np.array(scores, dtype=np.float32) if scores else np.empty((0,)),
            "num_matches": len(good_pts0),
            "method": "sift"
        }

    def _is_homography_valid(self, H: np.ndarray, w: int, h: int) -> bool:
        """驗證透視變換矩陣是否合理（防止翻轉或過度扭曲）"""
        if H is None or H.shape != (3, 3):
            return False

        det = np.linalg.det(H[:2, :2])
        if det <= 0.15 or det >= 8.0:
            return False

        corners = np.array([
            [0, 0, 1],
            [w, 0, 1],
            [w, h, 1],
            [0, h, 1]
        ], dtype=np.float32).T

        proj_corners = H @ corners
        w_proj = proj_corners[2, :]
        if np.any(w_proj <= 0.05):
            return False

        proj_pts = (proj_corners[:2, :] / w_proj).T

        def poly_area(pts):
            x = pts[:, 0]
            y = pts[:, 1]
            return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

        orig_area = float(w * h)
        transformed_area = poly_area(proj_pts)
        area_ratio = transformed_area / orig_area

        if area_ratio < 0.35 or area_ratio > 3.0:
            return False

        return True

    def align_pair(
        self,
        pre_img: np.ndarray,
        post_img: np.ndarray,
        align_target: str = "post"
    ) -> Dict[str, Any]:
        """
        配對並對齊影像
        align_target="post": 將借車影像 pre_img 幾何對齊到還車影像 post_img 坐標系，
        確保還車影像保持 100% 原生高解析度無任何重取樣失真，提供最精確之車損分析。
        支援在內點不足時自適應搜尋最佳拍攝旋轉角度 (0°, 90°, 180°, 270°)。
        """
        rot_map = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}

        # 優先測試原始 0 度
        m0 = self.match_features(pre_img, post_img)
        pts_pre0, pts_post0 = m0["pts0"], m0["pts1"]
        cnt0 = 0
        M0 = None

        if m0["num_matches"] >= 4:
            src_p, dst_p = (pts_pre0, pts_post0) if align_target == "post" else (pts_post0, pts_pre0)
            M0, inls0 = cv2.estimateAffinePartial2D(src_p, dst_p, method=cv2.RANSAC, ransacReprojThreshold=5.0)
            cnt0 = int(np.sum(inls0)) if inls0 is not None else 0

        best_inls = cnt0
        best_M = M0
        best_post = post_img
        best_matches = m0["num_matches"]
        best_r = 0
        best_method = m0["method"]

        # 若 0 度內點數充足 (>= 20)，代表方向正確，直接使用以極大化推論速度
        if cnt0 < 20:
            for r in [90, 180, 270]:
                cur_post = cv2.rotate(post_img, rot_map[r])
                m_r = self.match_features(pre_img, cur_post)
                if m_r["num_matches"] >= 4:
                    src_p, dst_p = (m_r["pts0"], m_r["pts1"]) if align_target == "post" else (m_r["pts1"], m_r["pts0"])
                    M_r, inls_r = cv2.estimateAffinePartial2D(src_p, dst_p, method=cv2.RANSAC, ransacReprojThreshold=5.0)
                    cnt_r = int(np.sum(inls_r)) if inls_r is not None else 0
                    if cnt_r > best_inls:
                        best_inls = cnt_r
                        best_M = M_r
                        best_post = cur_post
                        best_matches = m_r["num_matches"]
                        best_r = r
                        best_method = m_r["method"]

        method_used = "affine_partial"
        # 若 Partial Affine 內點數偏低 (< 25)，評估 Full Affine 是否顯著改善 (例如具有較大視角透視畸變之特寫照)
        if best_inls < 25 and best_matches >= 4:
            src_p, dst_p = (pts_pre0, pts_post0) if align_target == "post" else (pts_post0, pts_pre0)
            M_full, inls_full = cv2.estimateAffine2D(src_p, dst_p, method=cv2.RANSAC, ransacReprojThreshold=5.0)
            cnt_full = int(np.sum(inls_full)) if inls_full is not None else 0
            if cnt_full >= 25:
                best_inls = cnt_full
                best_M = M_full
                method_used = "affine_full"

        h_pre, w_pre = pre_img.shape[:2]
        h_post, w_post = best_post.shape[:2]

        if align_target == "post":
            target_w, target_h = w_post, h_post
            warp_src = pre_img
            target_ref = best_post
        else:
            target_w, target_h = w_pre, h_pre
            warp_src = best_post
            target_ref = pre_img

        alignment_success = False

        if best_M is not None and best_inls >= 4:
            aligned_img = cv2.warpAffine(
                warp_src, best_M, (target_w, target_h),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
            )
            alignment_success = True
        else:
            aligned_img = cv2.resize(warp_src, (target_w, target_h))
            best_M = np.eye(2, 3)
            method_used = "resize_fallback"

        return {
            "aligned_image": aligned_img,
            "target_image": target_ref,
            "affine_matrix": best_M,
            "alignment_success": alignment_success,
            "method_used": method_used,
            "num_matches": best_matches,
            "inliers_count": best_inls,
            "rotation_applied": best_r,
            "feature_method": best_method
        }

    def compute_structural_diff(
        self,
        pre_image: np.ndarray,
        post_image_aligned: np.ndarray,
        vehicle_mask: Optional[np.ndarray] = None,
        vehicle_mask_src_aligned: Optional[np.ndarray] = None,
        plate_boxes: Optional[List[Dict[str, Any]]] = None,
        image_type: int = 1,
        diff_threshold: int = 55,
        min_contour_area: int = 295,
        merge_margin: int = 8
    ) -> Dict[str, Any]:
        """
        計算結構差異，結合光影正規化、語義車身過濾、接縫衰減、輪胎抑制與高頻缺陷提取，
        輸出高品質車損病灶候選框。
        """
        h, w = post_image_aligned.shape[:2]
        if pre_image.shape[:2] != (h, w):
            pre_image = cv2.resize(pre_image, (w, h))

        # 1. 雙向車輛輪廓語義遮罩融合
        if vehicle_mask is not None:
            if vehicle_mask.shape[:2] != (h, w):
                vehicle_mask = cv2.resize(vehicle_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            body_1 = cv2.erode((vehicle_mask > 127).astype(np.uint8) * 255, np.ones((7, 7), np.uint8))
        else:
            body_1 = np.ones((h, w), dtype=np.uint8) * 255

        if vehicle_mask_src_aligned is not None:
            if vehicle_mask_src_aligned.shape[:2] != (h, w):
                vehicle_mask_src_aligned = cv2.resize(vehicle_mask_src_aligned, (w, h), interpolation=cv2.INTER_NEAREST)
            body_2 = cv2.erode((vehicle_mask_src_aligned > 127).astype(np.uint8) * 255, np.ones((11, 11), np.uint8))
        else:
            body_2 = np.ones((h, w), dtype=np.uint8) * 255

        body_mask = cv2.bitwise_and(body_1, body_2)

        # 排除幾何變換超出畫布之邊界黑邊 (valid_pre)
        valid_pre = cv2.erode((pre_image.sum(axis=2) > 15).astype(np.uint8) * 255, np.ones((11, 11), np.uint8))
        body_mask = cv2.bitwise_and(body_mask, valid_pre)

        # 玻璃反射與地面陰影排除
        pts = np.argwhere(vehicle_mask > 127) if vehicle_mask is not None else np.array([])
        if len(pts) > 0:
            vh = pts[:, 0].max() - pts[:, 0].min()
            vw = pts[:, 1].max() - pts[:, 1].min()
            min_y = pts[:, 0].min()
            min_x = pts[:, 1].min()
            
            # 若為車頭/車尾特寫 (車體頂部貼近邊界且整體高度有限)，避免錯誤抹除引擎蓋或車頭飾板
            if image_type in (1, 2) and min_y < 20 and pts[:, 0].max() < h * 0.70:
                glass_ratio = 0.0
            else:
                glass_ratio = 0.30 if image_type in (3, 4) else 0.40

            if glass_ratio > 0:
                body_mask[min_y : int(min_y + glass_ratio * vh), :] = 0
            # 僅剔除底部 2% 地面陰影與邊界雜訊 (保護前/後保險桿下巴與擾流板)
            body_mask[int(pts[:, 0].max() - 0.02 * vh) : pts[:, 0].max() + 1, :] = 0
        else:
            vh, vw, min_y, min_x = h, w, 0, 0

        # 車牌擴張抑制 (25x25 適度保護車牌邊界與後行李箱車損)
        if plate_boxes:
            plate_mask = np.zeros((h, w), dtype=np.uint8)
            for p in plate_boxes:
                bbox = p.get("bbox", [])
                if len(bbox) == 4 and isinstance(bbox[0], (int, float)):
                    bx1, by1, bx2, by2 = [int(v) for v in bbox]
                    cv2.rectangle(plate_mask, (bx1, by1), (bx2, by2), 255, -1)
                elif len(bbox) >= 3:
                    cv2.fillPoly(plate_mask, [np.array(bbox, dtype=np.int32)], 255)
            plate_dilated = cv2.dilate(plate_mask, np.ones((25, 25), np.uint8))
            body_mask[plate_dilated > 0] = 0

        # 2. 灰階化與光影正規化
        gray_pre = cv2.cvtColor(pre_image, cv2.COLOR_BGR2GRAY) if len(pre_image.shape) == 3 else pre_image
        gray_post = cv2.cvtColor(post_image_aligned, cv2.COLOR_BGR2GRAY) if len(post_image_aligned.shape) == 3 else post_image_aligned

        # 車漆光影正規化 (黑車或強光/陰影巨幅光差)
        m_eval = (body_mask > 127)
        if np.any(m_eval):
            mean_post = np.mean(gray_post[m_eval])
            mean_pre = np.mean(gray_pre[m_eval])
            is_dark_car = (mean_post < 65)
            delta = abs(mean_post - mean_pre)
            if (is_dark_car and delta > 20.0) or (delta > 35.0):
                src_lab = cv2.cvtColor(pre_image, cv2.COLOR_BGR2LAB).astype(np.float32)
                tgt_lab = cv2.cvtColor(post_image_aligned, cv2.COLOR_BGR2LAB).astype(np.float32)
                for ch in range(3):
                    s_vals = src_lab[:, :, ch][m_eval]
                    t_vals = tgt_lab[:, :, ch][m_eval]
                    s_mean, s_std = np.mean(s_vals), np.std(s_vals) + 1e-5
                    t_mean, t_std = np.mean(t_vals), np.std(t_vals) + 1e-5
                    src_lab[:, :, ch] = (src_lab[:, :, ch] - s_mean) * (t_std / s_std) + t_mean
                pre_image = cv2.cvtColor(np.clip(src_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
                gray_pre = cv2.cvtColor(pre_image, cv2.COLOR_BGR2GRAY)

        edges_pre = cv2.Canny(gray_pre, 30, 100)
        edges_pre_dil = cv2.dilate(edges_pre, np.ones((7, 7), np.uint8))

        # 水箱散熱網高密度結構紋理抑制
        k_density = np.ones((31, 31), np.float32) / (31 * 31)
        edge_density = cv2.filter2D((edges_pre > 0).astype(np.float32), -1, k_density)
        dense_texture_mask = cv2.dilate((edge_density > 0.22).astype(np.uint8) * 255, np.ones((5, 5), np.uint8))

        # 3. 顏色差異與高頻結構差分
        diff_color = np.max(cv2.absdiff(post_image_aligned, pre_image), axis=2)
        hp_pre = cv2.absdiff(gray_pre, cv2.GaussianBlur(gray_pre, (21, 21), 0))
        hp_post = cv2.absdiff(gray_post, cv2.GaussianBlur(gray_post, (21, 21), 0))
        hp_diff = cv2.absdiff(hp_pre, hp_post)

        joint_diff = cv2.addWeighted(diff_color, 0.65, hp_diff, 0.35, 0)
        joint_diff[edges_pre_dil > 0] = (joint_diff[edges_pre_dil > 0] * 0.15).astype(np.uint8)
        joint_diff[dense_texture_mask > 0] = 0

        # 輪圈/輪胎橡膠抑制 (限制於車身最底部 30% 且縮小擴張，保護保險桿與水箱氣壩)
        if len(pts) > 0:
            tire_rubber = ((gray_post < 40) & (vehicle_mask > 0)).astype(np.uint8) * 255
            tire_rubber[:int(min_y + 0.60 * vh), :] = 0
            if image_type == 2:
                tire_rubber[:, :int(min_x + 0.70 * vw)] = 0
            tire_dil = cv2.dilate(tire_rubber, np.ones((30, 30), np.uint8))
            joint_diff[tire_dil > 0] = 0

        # 極端暗部與反光車漆過濾
        joint_diff[gray_post < 25] = 0
        joint_diff[(pre_image > 235).all(axis=2)] = 0
        joint_diff[(post_image_aligned > 230).all(axis=2)] = 0
        joint_diff = cv2.bitwise_and(joint_diff, joint_diff, mask=body_mask)

        # 4. 二值化門檻過濾與形態學連接 (15x15 閉運算橋接細微平行刮痕)
        _, raw_thresh = cv2.threshold(joint_diff, diff_threshold, 255, cv2.THRESH_BINARY)
        kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        kernel15 = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        opened = cv2.morphologyEx(raw_thresh, cv2.MORPH_OPEN, kernel3)
        diff_cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel15)

        # 5. 輪廓提取與病灶候選框生成
        contours, _ = cv2.findContours(diff_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        raw_boxes = []

        for c in contours:
            area = cv2.contourArea(c)
            if min_contour_area <= area <= 65000:
                x, y, bw, bh = cv2.boundingRect(c)
                aspect = max(bw, bh) / (min(bw, bh) + 1e-5)
                if aspect > 7 and min(bw, bh) <= 12:
                    continue

                x1 = max(0, x - merge_margin)
                y1 = max(0, y - merge_margin)
                x2 = min(w - 1, x + bw + merge_margin)
                y2 = min(h - 1, y + bh + merge_margin)

                raw_boxes.append([x1, y1, x2, y2])
                candidates.append({
                    "bbox": [x1, y1, x2, y2],
                    "raw_contour": c,
                    "area": float(area),
                    "aspect_ratio": round(float(aspect), 2)
                })

        # 6. 熱力圖可視化與 SSIM 結構相似度評估
        heatmap = cv2.applyColorMap(joint_diff, cv2.COLORMAP_JET)

        try:
            sw, sh = 320, int(320 * h / max(w, 1))
            gp_s = cv2.resize(gray_pre, (sw, sh))
            gpo_s = cv2.resize(gray_post, (sw, sh))
            ssim_score = float(ssim(gp_s, gpo_s))
        except Exception:
            ssim_score = 0.85

        return {
            "candidate_boxes": raw_boxes,
            "candidates": candidates,
            "diff_mask": diff_cleaned,
            "diff_heatmap": heatmap,
            "raw_joint_diff": joint_diff,
            "num_candidates": len(raw_boxes),
            "ssim_score": round(ssim_score, 4)
        }

    def _merge_overlapping_boxes(self, boxes: List[List[int]]) -> List[List[int]]:
        """合併重疊或相鄰的邊界框"""
        if not boxes:
            return []

        boxes = sorted(boxes, key=lambda b: (b[0], b[1]))
        merged = []

        while boxes:
            curr = boxes.pop(0)
            has_merge = False
            for i, other in enumerate(boxes):
                inter_x1 = max(curr[0], other[0])
                inter_y1 = max(curr[1], other[1])
                inter_x2 = min(curr[2], other[2])
                inter_y2 = min(curr[3], other[3])

                if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                    curr = [
                        min(curr[0], other[0]),
                        min(curr[1], other[1]),
                        max(curr[2], other[2]),
                        max(curr[3], other[3])
                    ]
                    boxes.pop(i)
                    has_merge = True
                    break

            if has_merge:
                boxes.append(curr)
            else:
                merged.append(curr)

        return merged
