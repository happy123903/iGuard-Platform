"""
VisionGuard V3 — 基礎品質與曝光檢查模組

解決痛點：使用者常上傳模糊、過曝、死黑的照片，導致後續 AI 辨識完全失效。
此模組在拍照防呆管線的最前端進行快速篩檢（<15ms），不合格立即引導重拍。

檢測項目：
1. 模糊度檢測（Laplacian 變異數）
2. 整體曝光分析（平均亮度 + 暗部/亮部比例）
3. 九宮格區域亮度均勻性分析
4. 綜合品質評分

閾值來源：基於競賽提供的 240 張正常無損照片校準。
"""

import json
import logging
import cv2
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

# 載入閾值設定
CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "thresholds.json"


def _load_thresholds() -> dict:
    """載入品質閾值設定"""
    defaults = {
        "blur_threshold": 80.0,
        "dark_mean_threshold": 35,
        "dark_ratio_threshold": 0.4,
        "bright_mean_threshold": 225,
        "bright_ratio_threshold": 0.35,
        "zero_dce_trigger_mean": 60
    }
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = json.load(f)
            return config.get("quality", defaults)
    return defaults


# 模組層級載入閾值（避免每次呼叫都讀檔）
_THRESHOLDS = _load_thresholds()


def check_blur(gray: np.ndarray) -> dict:
    """
    模糊度檢測：使用 Laplacian 算子計算影像銳利度。

    原理：Laplacian 是二階微分算子，對焦清晰的影像其 Laplacian 變異數較高。
    閾值 80.0 由 240 張競賽正常照片的清晰度分佈校準。

    Args:
        gray: 灰階影像

    Returns:
        dict: {"score": 銳利度分數, "passed": 是否通過, "warning": 警告訊息或 None}
    """
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    threshold = _THRESHOLDS["blur_threshold"]
    passed = blur_score >= threshold

    return {
        "score": round(blur_score, 1),
        "passed": passed,
        "warning": "照片模糊，請拿穩手機並對焦後重新拍攝" if not passed else None
    }


def check_exposure(gray: np.ndarray) -> dict:
    """
    曝光分析：偵測整體過暗、過亮、死黑區域、過曝區域。

    分析指標：
    - mean_brightness: 灰階平均值（正常範圍 40~220）
    - dark_ratio: 像素值 < 25 的比例（死黑區域）
    - bright_ratio: 像素值 > 235 的比例（過曝區域）

    Args:
        gray: 灰階影像

    Returns:
        dict: 包含亮度指標與警告
    """
    mean_val = float(gray.mean())
    std_val = float(gray.std())
    dark_ratio = float((gray < 25).mean())
    bright_ratio = float((gray > 235).mean())

    warnings = []

    # 極度昏暗（地下室場景）
    if mean_val < _THRESHOLDS["dark_mean_threshold"] or dark_ratio > _THRESHOLDS["dark_ratio_threshold"]:
        warnings.append("環境極度昏暗，系統已嘗試自動增強，建議開啟手電筒補光")

    # 過曝/強光反射
    if mean_val > _THRESHOLDS["bright_mean_threshold"] or bright_ratio > _THRESHOLDS["bright_ratio_threshold"]:
        warnings.append("反光或強光過曝，請避開直射光源或調整拍攝角度")

    # 對比度極低（如濃霧、鏡頭沾水）
    if std_val < 15:
        warnings.append("影像對比度極低，請擦拭鏡頭或等待視野改善")

    return {
        "mean_brightness": round(mean_val, 1),
        "std_brightness": round(std_val, 1),
        "dark_ratio": round(dark_ratio, 3),
        "bright_ratio": round(bright_ratio, 3),
        "passed": len(warnings) == 0,
        "warnings": warnings
    }


def check_nine_grid(gray: np.ndarray) -> dict:
    """
    九宮格區域亮度均勻性分析。

    將影像分成 3×3 九個區域，檢查各區域亮度差異。
    若某區域亮度嚴重偏離整體平均，可能表示：
    - 局部遮擋（手指擋住鏡頭一角）
    - 局部反光（車身金屬反射太陽光）
    - 半邊陰影（車輛停在建築物邊界）

    Args:
        gray: 灰階影像

    Returns:
        dict: 九宮格分析結果
    """
    h, w = gray.shape
    grid_h, grid_w = h // 3, w // 3

    grid_means = []
    grid_details = []

    for row in range(3):
        for col in range(3):
            y_start = row * grid_h
            y_end = (row + 1) * grid_h if row < 2 else h
            x_start = col * grid_w
            x_end = (col + 1) * grid_w if col < 2 else w

            region = gray[y_start:y_end, x_start:x_end]
            region_mean = float(region.mean())
            grid_means.append(region_mean)
            grid_details.append({
                "position": f"({row},{col})",
                "mean": round(region_mean, 1)
            })

    overall_mean = float(np.mean(grid_means))
    max_deviation = float(max(abs(m - overall_mean) for m in grid_means))

    # 偏差超過 80 表示嚴重不均勻
    warnings = []
    if max_deviation > 80:
        warnings.append("影像亮度嚴重不均，可能有局部遮擋或強烈反光")
    elif max_deviation > 60:
        warnings.append("影像亮度分佈不均勻，建議調整拍攝位置")

    return {
        "grid_means": [round(m, 1) for m in grid_means],
        "overall_mean": round(overall_mean, 1),
        "max_deviation": round(max_deviation, 1),
        "uniformity_passed": max_deviation <= 80,
        "warnings": warnings
    }


def check_basic_quality(image: np.ndarray) -> dict:
    """
    綜合基礎品質檢查：整合模糊度、曝光、九宮格分析。

    此函數是環節一防呆管線的核心品質篩檢點，
    在 RTX 5090 上整體耗時 <5ms（純 CPU 運算）。

    Args:
        image: BGR 格式輸入影像

    Returns:
        dict: {
            "passed": 整體是否通過品質檢查,
            "blur_score": 銳利度分數（越高越清晰）,
            "mean_brightness": 平均亮度,
            "needs_enhancement": 是否建議進行微光增強,
            "warnings": 所有警告訊息列表,
            "details": {
                "blur": 模糊度檢查詳情,
                "exposure": 曝光檢查詳情,
                "nine_grid": 九宮格分析詳情
            }
        }
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 執行三項檢查
    blur_result = check_blur(gray)
    exposure_result = check_exposure(gray)
    grid_result = check_nine_grid(gray)

    # 彙整所有警告
    all_warnings = []
    if blur_result["warning"]:
        all_warnings.append(blur_result["warning"])
    all_warnings.extend(exposure_result["warnings"])
    all_warnings.extend(grid_result["warnings"])

    # 判斷是否需要 Zero-DCE 增強
    needs_enhancement = exposure_result["mean_brightness"] < _THRESHOLDS["zero_dce_trigger_mean"]

    # 整體品質判定
    #   - 模糊度必須通過
    #   - 曝光可以透過 Zero-DCE 補救，但極端過曝無法補救
    overall_passed = blur_result["passed"]
    if exposure_result["bright_ratio"] > _THRESHOLDS["bright_ratio_threshold"]:
        overall_passed = False  # 過曝無法補救

    return {
        "passed": overall_passed,
        "blur_score": blur_result["score"],
        "mean_brightness": exposure_result["mean_brightness"],
        "needs_enhancement": needs_enhancement,
        "warnings": all_warnings,
        "details": {
            "blur": blur_result,
            "exposure": exposure_result,
            "nine_grid": grid_result
        }
    }


def compute_quality_score(quality_result: dict) -> int:
    """
    將品質檢查結果轉換為 0-100 分的品質評分。
    用於 Smart Capture Score 使用者拍照品質積分系統。

    評分權重：
    - 模糊度：40%
    - 曝光適當性：30%
    - 亮度均勻性：30%

    Args:
        quality_result: check_basic_quality() 的回傳結果

    Returns:
        int: 0-100 品質分數
    """
    details = quality_result.get("details", {})

    # 模糊度分數 (0-40)
    blur_score = details.get("blur", {}).get("score", 0)
    blur_points = min(40, int(blur_score / _THRESHOLDS["blur_threshold"] * 40))

    # 曝光分數 (0-30)
    mean_bright = details.get("exposure", {}).get("mean_brightness", 128)
    # 最佳亮度約 100-160，偏離越多分數越低
    if 100 <= mean_bright <= 160:
        exposure_points = 30
    elif 60 <= mean_bright < 100 or 160 < mean_bright <= 200:
        exposure_points = 20
    elif 35 <= mean_bright < 60 or 200 < mean_bright <= 225:
        exposure_points = 10
    else:
        exposure_points = 0

    # 均勻性分數 (0-30)
    max_dev = details.get("nine_grid", {}).get("max_deviation", 0)
    if max_dev <= 30:
        uniformity_points = 30
    elif max_dev <= 50:
        uniformity_points = 20
    elif max_dev <= 80:
        uniformity_points = 10
    else:
        uniformity_points = 0

    total = blur_points + exposure_points + uniformity_points
    return max(0, min(100, total))
