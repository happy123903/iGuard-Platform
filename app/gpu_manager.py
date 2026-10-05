"""
iGuard — RTX 5090 GPU 顯存與模型生命週期管理器 (GPU Manager)
管理本機 RTX 5090 (32GB VRAM) 上的各環節 AI 模型常駐與顯存排程。

模型清單與顯存預算分佈：
1. Zero-DCE (微光增強): ~40 MB
2. CompositionGuard (YOLOv11x-seg): ~160 MB
3. PlateGuard (EasyOCR GPU): ~1,200 MB
4. SuperPointAligner (SuperPoint + LightGlue): ~320 MB
5. ExteriorInspectionEngine (Meta SAM 2 Large): ~1,850 MB
6. QwenInteriorEngine (Qwen2.5-VL-3B-Instruct BF16): ~7,200 MB
7. RiskDispatcher (營運三色分流調度引擎): ~10 MB
總計顯存佔用約 10.8 GB，RTX 5090 (32 GB) 仍保有 >21 GB 充裕空間。
"""

import logging
import threading
from typing import Dict, Any, Optional

import torch

logger = logging.getLogger("iguard.gpu_manager")

# 引入 VisionGuard 各環節模組
from app.guard.zero_dce import ZeroDCEEnhancer
from app.guard.composition import CompositionGuard
from app.guard.plate_ocr import PlateGuard
from app.engine.superpoint_align import SuperPointAligner
from app.engine.sam2_damage import ExteriorInspectionEngine
from app.engine.qwen_interior import QwenInteriorEngine
from app.engine.risk_dispatcher import RiskDispatcher


class GPUManager:
    """
    RTX 5090 模型與顯存管理器（單例模式）
    支援模型懶加載、線程安全初始化、即時 VRAM 監控與模型熱載入。
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(GPUManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        self.total_vram_gb = (
            torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            if torch.cuda.is_available()
            else 0.0
        )

        # 模型實例快取
        self._dce: Optional[ZeroDCEEnhancer] = None
        self._composition: Optional[CompositionGuard] = None
        self._plate: Optional[PlateGuard] = None
        self._aligner: Optional[SuperPointAligner] = None
        self._exterior: Optional[ExteriorInspectionEngine] = None
        self._interior: Optional[QwenInteriorEngine] = None
        self._dispatcher: Optional[RiskDispatcher] = None

        self._model_lock = threading.Lock()
        self._initialized = True
        logger.info(f"[GPUManager] Initialized on {self.device_name} ({self.total_vram_gb:.1f} GB VRAM)")

    # -------------------------------------------------------------------------
    # 懶加載屬性存取 (Thread-Safe Lazy Properties)
    # -------------------------------------------------------------------------
    @property
    def zero_dce(self) -> ZeroDCEEnhancer:
        if self._dce is None:
            with self._model_lock:
                if self._dce is None:
                    logger.info("[GPUManager] Loading Zero-DCE Enhancer...")
                    self._dce = ZeroDCEEnhancer()
        return self._dce

    @property
    def composition_guard(self) -> CompositionGuard:
        if self._composition is None:
            with self._model_lock:
                if self._composition is None:
                    logger.info("[GPUManager] Loading YOLOv11x-seg Composition Guard...")
                    self._composition = CompositionGuard()
        return self._composition

    @property
    def plate_guard(self) -> PlateGuard:
        if self._plate is None:
            with self._model_lock:
                if self._plate is None:
                    logger.info("[GPUManager] Loading EasyOCR Plate Guard (GPU)...")
                    self._plate = PlateGuard(enhancer=self.zero_dce)
        return self._plate

    @property
    def aligner(self) -> SuperPointAligner:
        if self._aligner is None:
            with self._model_lock:
                if self._aligner is None:
                    logger.info("[GPUManager] Loading SuperPoint + LightGlue Aligner...")
                    self._aligner = SuperPointAligner(device=self.device)
        return self._aligner

    @property
    def exterior_engine(self) -> ExteriorInspectionEngine:
        if self._exterior is None:
            with self._model_lock:
                if self._exterior is None:
                    logger.info("[GPUManager] Loading Meta SAM 2 Large Exterior Engine...")
                    self._exterior = ExteriorInspectionEngine(device=self.device)
        return self._exterior

    @property
    def interior_engine(self) -> QwenInteriorEngine:
        if self._interior is None:
            with self._model_lock:
                if self._interior is None:
                    logger.info("[GPUManager] Loading Qwen2.5-VL-3B-Instruct Interior Engine...")
                    self._interior = QwenInteriorEngine()
        return self._interior

    @property
    def dispatcher(self) -> RiskDispatcher:
        if self._dispatcher is None:
            with self._model_lock:
                if self._dispatcher is None:
                    logger.info("[GPUManager] Initializing Risk Dispatcher...")
                    self._dispatcher = RiskDispatcher()
        return self._dispatcher

    # -------------------------------------------------------------------------
    # 預熱與顯存狀態
    # -------------------------------------------------------------------------
    def warmup_guard_models(self):
        """預熱 Step 1~4 輕量守門員模組（構圖與車牌辨識）"""
        _ = self.zero_dce
        _ = self.composition_guard
        _ = self.plate_guard
        _ = self.dispatcher
        logger.info("[GPUManager] Guard models warmed up.")

    def warmup_heavy_models(self):
        """預熱 Step 5 (SAM 2) 與 Step 6 (Qwen2.5-VL) 深度推論大模型"""
        _ = self.aligner
        _ = self.exterior_engine
        _ = self.interior_engine
        logger.info("[GPUManager] Heavy models (SAM 2 & Qwen2.5-VL) warmed up.")

    def get_status(self) -> Dict[str, Any]:
        """獲取當前 GPU 硬體狀態與模型加載列表"""
        cuda_ok = torch.cuda.is_available()
        allocated_mb = torch.cuda.memory_allocated() / (1024 ** 2) if cuda_ok else 0.0
        reserved_mb = torch.cuda.memory_reserved() / (1024 ** 2) if cuda_ok else 0.0

        loaded_models = []
        if self._dce is not None: loaded_models.append("Zero-DCE")
        if self._composition is not None: loaded_models.append("YOLOv11x-seg")
        if self._plate is not None: loaded_models.append("EasyOCR-GPU")
        if self._aligner is not None: loaded_models.append("SuperPoint-LightGlue")
        if self._exterior is not None: loaded_models.append("Meta-SAM2-Large")
        if self._interior is not None: loaded_models.append("Qwen2.5-VL-3B")
        if self._dispatcher is not None: loaded_models.append("RiskDispatcher")

        return {
            "device": self.device,
            "device_name": self.device_name,
            "cuda_available": cuda_ok,
            "total_vram_gb": round(self.total_vram_gb, 2),
            "allocated_vram_mb": round(allocated_mb, 1),
            "reserved_vram_mb": round(reserved_mb, 1),
            "free_vram_gb": round(self.total_vram_gb - (reserved_mb / 1024), 2) if cuda_ok else 0.0,
            "loaded_models_count": len(loaded_models),
            "loaded_models": loaded_models
        }


# 全域單例
gpu_manager = GPUManager()
