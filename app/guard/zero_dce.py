"""
VisionGuard V3 — Zero-DCE 零參考深度曲線估計微光增強模組

解決痛點：大量 iRent 使用者在地下停車場（B2/B3）拍照，光線極差，
傳統 CV 邊緣檢測直接失效。Zero-DCE 能在 <10ms 內無損提亮暗部細節。

架構說明：
- 若 models/zero_dce.pth 存在 → 使用官方預訓練 Zero-DCE 神經網絡（最佳效果）
- 若模型不存在 → 自動降級使用 CLAHE 自適應直方圖增強（備援方案）

模型來源：Li-Chongyi/Zero-DCE (GitHub) — Epoch99.pth
論文：Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement (CVPR 2020)
"""

import os
import json
import logging
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

# 載入閾值設定
CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "thresholds.json"


def _load_config() -> dict:
    """載入品質閾值設定"""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"quality": {"zero_dce_trigger_mean": 60}}


# ============================================================
# Zero-DCE 官方神經網絡架構 (enhance_net_nopool)
# 來源：https://github.com/Li-Chongyi/Zero-DCE/blob/master/Zero-DCE_code/model.py
# ============================================================

class DCENet(nn.Module):
    """
    Zero-DCE 深度曲線估計網絡（官方 enhance_net_nopool 架構）。
    輸入低光照影像，輸出 8 組光照增強曲線參數 (Light Enhancement Curves)。
    網絡極輕量（僅 ~79K 參數），在 RTX 5090 上推論 <10ms。

    架構特點：
    - 7 層 3×3 卷積 + ReLU
    - 對稱跳接 (Symmetrical Skip Connection)
    - 輸出 24 通道（3 RGB × 8 迭代）
    - 迭代式曲線增強（8 次迭代逐步提亮）
    """

    def __init__(self):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)

        number_f = 32
        # 編碼路徑：逐層提取光照特徵
        self.e_conv1 = nn.Conv2d(3, number_f, 3, 1, 1, bias=True)
        self.e_conv2 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv4 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)

        # 解碼路徑：帶對稱跳接
        self.e_conv5 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.e_conv6 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)

        # 輸出層：24 通道 = 3 RGB × 8 組曲線參數
        self.e_conv7 = nn.Conv2d(number_f * 2, 24, 3, 1, 1, bias=True)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: 輸入影像 tensor [B, 3, H, W]，值域 [0, 1]

        Returns:
            enhance_image_1: 中間增強結果（第 4 迭代後）
            enhance_image: 最終增強結果（第 8 迭代後）
            r: 全部 8 組曲線參數 [B, 24, H, W]
        """
        # 編碼路徑
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))

        # 解碼路徑（帶跳接）
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))

        # 輸出曲線參數，用 tanh 限制在 [-1, 1]
        x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))

        # 拆分為 8 組 RGB 曲線參數
        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(x_r, 3, dim=1)

        # 迭代應用光照增強曲線 (LE-Curve)
        x = x + r1 * (torch.pow(x, 2) - x)
        x = x + r2 * (torch.pow(x, 2) - x)
        x = x + r3 * (torch.pow(x, 2) - x)
        enhance_image_1 = x + r4 * (torch.pow(x, 2) - x)
        x = enhance_image_1 + r5 * (torch.pow(enhance_image_1, 2) - enhance_image_1)
        x = x + r6 * (torch.pow(x, 2) - x)
        x = x + r7 * (torch.pow(x, 2) - x)
        enhance_image = x + r8 * (torch.pow(x, 2) - x)

        r = torch.cat([r1, r2, r3, r4, r5, r6, r7, r8], 1)
        return enhance_image_1, enhance_image, r


# ============================================================
# Zero-DCE 增強器（含自動降級備援）
# ============================================================

class ZeroDCEEnhancer:
    """
    Zero-DCE 零參考深度曲線估計微光增強器。

    功能：
    - 自動偵測是否為暗光場景（平均亮度 < zero_dce_trigger_mean）
    - 若有預訓練模型 → GPU 加速 Zero-DCE 增強（<10ms）
    - 若無模型 → 降級為 CLAHE 自適應直方圖增強
    - 增強後保留車身真實幾何，不引入假刮痕

    使用範例：
        enhancer = ZeroDCEEnhancer()
        result = enhancer.enhance(bgr_image)
        if result["enhanced"]:
            enhanced_image = result["image"]
    """

    def __init__(
        self,
        model_path: str = "models/zero_dce.pth",
        device: str = "cuda"
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = None
        self.use_neural = False
        self.config = _load_config()
        self.trigger_mean = self.config["quality"].get("zero_dce_trigger_mean", 60)

        # 嘗試載入 Zero-DCE 官方預訓練模型
        model_full_path = Path(__file__).parent.parent.parent / model_path
        if model_full_path.exists():
            try:
                self.model = DCENet()
                state_dict = torch.load(str(model_full_path), map_location=self.device, weights_only=True)
                self.model.load_state_dict(state_dict)
                self.model = self.model.to(self.device).eval().half()
                self.use_neural = True
                logger.info(f"Zero-DCE model loaded: {model_full_path} (device={self.device})")
            except Exception as e:
                logger.warning(f"Zero-DCE model load failed, fallback to CLAHE: {e}")
                self.model = None
                self.use_neural = False
        else:
            logger.info(
                f"Zero-DCE model not found ({model_full_path}), using CLAHE fallback."
            )

        # CLAHE 備援增強器
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    def is_low_light(self, bgr_image: np.ndarray) -> bool:
        """
        判斷影像是否為暗光場景。
        灰階平均亮度 < trigger_mean（預設 60）時觸發增強。
        """
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        return float(gray.mean()) < self.trigger_mean

    def enhance(self, bgr_image: np.ndarray, force: bool = False) -> dict:
        """
        智能微光增強：自動判斷是否需要增強，選擇最佳增強策略。

        Args:
            bgr_image: BGR 格式輸入影像
            force: 是否強制增強（忽略亮度判斷）

        Returns:
            dict: {
                "image": 增強後的 BGR 影像,
                "enhanced": 是否執行了增強,
                "method": "zero_dce" | "clahe" | "none",
                "original_mean": 原始平均亮度,
                "enhanced_mean": 增強後平均亮度
            }
        """
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        original_mean = float(gray.mean())

        # 判斷是否需要增強
        if not force and original_mean >= self.trigger_mean:
            return {
                "image": bgr_image,
                "enhanced": False,
                "method": "none",
                "original_mean": round(original_mean, 1),
                "enhanced_mean": round(original_mean, 1)
            }

        # 優先使用 Zero-DCE 神經網絡
        if self.use_neural and self.model is not None:
            enhanced = self._enhance_neural(bgr_image)
            method = "zero_dce"
        else:
            enhanced = self._enhance_clahe(bgr_image)
            method = "clahe"

        enhanced_gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        enhanced_mean = float(enhanced_gray.mean())

        return {
            "image": enhanced,
            "enhanced": True,
            "method": method,
            "original_mean": round(original_mean, 1),
            "enhanced_mean": round(enhanced_mean, 1)
        }

    def _enhance_neural(self, bgr_image: np.ndarray) -> np.ndarray:
        """
        使用 Zero-DCE 深度曲線估計進行神經網絡增強。
        在 RTX 5090 上 FP16 推論僅需 ~8ms。
        """
        # BGR → RGB → Tensor [0, 1]
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        tensor = (
            torch.from_numpy(rgb.astype(np.float32))
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device)
            .half()
            / 255.0
        )

        with torch.no_grad():
            _, enhanced_tensor, _ = self.model(tensor)

        # Tensor → NumPy → BGR
        enhanced_rgb = (
            enhanced_tensor
            .squeeze(0)
            .permute(1, 2, 0)
            .clamp(0, 1)
            .cpu()
            .float()
            .numpy()
            * 255.0
        ).astype(np.uint8)

        return cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2BGR)

    def _enhance_clahe(self, bgr_image: np.ndarray) -> np.ndarray:
        """
        CLAHE 自適應直方圖增強（備援方案）。
        在 LAB 色彩空間的 L 通道進行增強，避免色偏。
        """
        lab = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        l_enhanced = self.clahe.apply(l_channel)
        lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    @property
    def info(self) -> dict:
        """回傳增強器狀態資訊"""
        return {
            "method": "zero_dce" if self.use_neural else "clahe",
            "device": self.device,
            "trigger_mean": self.trigger_mean,
            "model_loaded": self.use_neural
        }
