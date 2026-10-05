"""
VisionGuard V3 — 環節一：拍照防呆管線 (Smart Capture Guard)
包含：
1. ZeroDCEEnhancer — 微光深度增強
2. Quality Checker — 模糊度、曝光與均勻度檢查
3. CompositionGuard — YOLOv11 構圖與視角驗證
4. PlateGuard — EasyOCR 車牌辨識與訂單自動核對
"""

from .zero_dce import ZeroDCEEnhancer
from .quality import (
    check_basic_quality,
    compute_quality_score,
    check_blur,
    check_exposure,
    check_nine_grid
)
from .composition import CompositionGuard
from .plate_ocr import PlateGuard

__all__ = [
    "ZeroDCEEnhancer",
    "check_basic_quality",
    "compute_quality_score",
    "check_blur",
    "check_exposure",
    "check_nine_grid",
    "CompositionGuard",
    "PlateGuard"
]
