"""
VisionGuard V3 — 工業級車牌辨識與訂單自動核對模組 (PlateGuard)

技術選型：EasyOCR (GPU 加速) + 台灣車牌混淆矩陣正規化與模糊比對
解決痛點：
1. 使用者拍錯車輛（非預訂車輛）
2. 車牌被遮擋、角度過斜或未入鏡
3. 繁體/台灣車牌特殊字元混淆容錯（如 'Z' 與 '2'、'I' 與 '1'、'O' 與 '0'、'S' 與 '5'）
4. 即時反饋指引（如「未完整拍到車牌，請調整角度確保車牌清晰」）

硬體配置：RTX 5090, EasyOCR GPU, 延遲 < 80ms
"""

import re
import time
import json
import logging
from pathlib import Path
from typing import Optional, Union, List, Tuple
import cv2
import numpy as np
import easyocr

logger = logging.getLogger(__name__)

# 設定檔路徑
CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "thresholds.json"

# 台灣車牌常見字元辨識混淆對應集合 (OCR Confusion Sets)
# 台灣租賃車輛常見 'R' 開頭編號（如 RCR-7661, RCG-2235, RCW-8160, RDS-6583, RFB-2091）
CONFUSION_SETS = [
    {'W', 'K', 'V', '1', 'I', 'M', 'H'},       # W 常被誤讀為 K, 1, I, V, H (如 RCW -> RCK, RC1, RCI)
    {'Q', 'O', '0', '9', 'D', 'C'},            # Q 常被誤讀為 O, 0, 9, D, C (如 RCQ -> RCO, RC9)
    {'Z', '2', '7'},                            # Z 常與 2, 7 混淆 (如 2235 -> ZZ35, 6760 -> 6Z60)
    {'I', '1', 'L', 'T', 'J'},                 # 1 與 I, L, T, J 混淆
    {'S', '5'},                                 # S 與 5 混淆
    {'G', '6', 'C', 'Q', '9'},                  # G 常與 6, C, 9 混淆 (如 RCG -> RC6, 9206 -> 920g)
    {'B', '8'},                                 # B 與 8 混淆
    {'U', 'V', 'Y'},                            # U/V/Y 混淆
    {'D', '0', 'O'},                            # D/0/O 混淆
    {'9', 'G', 'Q', 'P'},                       # 9 與 g, q, p 混淆
    {'E', '8', 'B', '3', 'Z'},                  # E, 8, B, 3, Z 混淆 (印刷體及歪斜邊緣常將 3 辨識為 Z, 8, B)
]

# 車身貼紙、車型名稱、品牌商標與非車牌黑名單（嚴格排除誤判）
BLACKLIST_SUBSTRINGS = [
    "IRENT", "IRAT", "IPETT", "IKENT", "IKERT", "TOYOTA", "YARIS",
    "4ARIS", "4ARI5", "YARI5", "SARIS", "YAR1S", "TOY0TA",
    "COROLLA", "CROSS", "VIOS", "SIENTA", "CAMRY", "RAV4", "ALTIS",
    "HOTAI", "WEMO", "GOSHUR", "CARPLUS", "ZIPCAR", "AVIS", "HERTZ",
    "HYBRID", "TURBO", "SPORT", "ELECTRIC", "PRIUS", "NISSAN", "HONDA"
]

# 台灣車牌常見正則表示法：
# 3碼英文+4碼數字 (e.g. RCG-2235, RDS-6583)
# 2碼英文+4碼數字 (e.g. AA-1234)
# 4碼數字+2碼英文 (e.g. 1234-AA)
# 3碼英文+3碼數字 (e.g. ABC-123)
PLATE_REGEX = re.compile(r'([A-Z]{2,4}[- ]?[0-9]{3,4}|[0-9]{3,4}[- ]?[A-Z]{2,4})')


def is_char_confusable(c1: str, c2: str) -> bool:
    """判斷兩字元是否在台灣車牌 OCR 混淆集合內相同或可互換"""
    if c1 == c2:
        return True
    for s in CONFUSION_SETS:
        if c1 in s and c2 in s:
            return True
    return False


def _load_plate_config() -> dict:
    """載入車牌 OCR 閾值設定"""
    defaults = {
        "confidence_threshold": 0.15,
        "match_similarity_threshold": 0.65,
        "gpu_enabled": True
    }
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                return cfg.get("plate_ocr", defaults)
        except Exception as e:
            logger.warning(f"讀取車牌 OCR 設定檔失敗: {e}，使用預設值")
    return defaults


def clean_plate_text(text: str) -> str:
    """去除特殊符號、連字號與空白，轉換為純大寫英數字元"""
    if not text:
        return ""
    # 先將豎線或斜線轉換為 1
    t = text.replace('|', '1').replace('/', '1')
    return re.sub(r'[^A-Z0-9]', '', t.upper())


def is_valid_plate_candidate(text: str) -> bool:
    """
    嚴格驗證候選文字是否符合台灣車牌基本特徵，並徹底過濾 iRent 等品牌車貼。
    規則：
    1. 純英文字串（如 IRENT, YARIS, TOYOTA）絕對不是車牌，直接剔除。
    2. 純數字字串（如客服電話 02-5588-0808）直接剔除。
    3. 長度必須介於 5 至 8 碼之間。
    4. 必須同時包含英文字母與數字。
    5. 不得包含已知品牌或車貼黑名單字串。
    """
    cleaned = clean_plate_text(text)
    if not (5 <= len(cleaned) <= 8):
        return False
    has_letters = any(c.isalpha() for c in cleaned)
    has_digits = any(c.isdigit() for c in cleaned)
    if not (has_letters and has_digits):
        return False
    for b in BLACKLIST_SUBSTRINGS:
        if b in cleaned:
            return False

    # 台灣車牌必定是英數分群結構（[英文][數字] 或 [數字][英文]）
    # 嚴格禁止「首尾皆數字、中間皆字母」（如 4ARI5 / 0YARIS1 等車身商標雜訊）
    stripped_ends = cleaned.strip("0123456789")
    if stripped_ends != cleaned and cleaned[0].isdigit() and cleaned[-1].isdigit() and len(stripped_ends) >= 2:
        return False

    return True


def compute_levenshtein_distance(s1: str, s2: str) -> int:
    """計算兩字串之 Levenshtein 編輯距離"""
    if len(s1) < len(s2):
        return compute_levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def normalize_with_target(candidate: str, target: str) -> tuple[bool, str, float]:
    """
    根據目標車牌與字元混淆對照表校正候選字串。
    支援：
    1. 完全比對
    2. 去除前綴/後綴噪聲字元（例如 IRCK9206 -> RCK9206 -> RCW9206）
    3. 逐字元混淆集合替換（例如 RCGZZ35 -> RCG2235, Rco6z60 -> RCQ6760）
    4. 模糊編輯距離容錯

    Returns:
        (is_matched, corrected_text, similarity)
    """
    cand = clean_plate_text(candidate)
    tgt = clean_plate_text(target)
    if not cand or not tgt:
        return False, cand, 0.0

    if cand == tgt:
        return True, tgt, 1.0

    # 候選字串變體生成（支援單字元噪聲前綴/後綴剝除、租賃車 R 前綴補全）
    variations = [cand]

    # 若首碼為常見邊界反光噪聲 'I' 或 '1'，且目標為 'R' 開頭（租賃車）：始終測試剝除首碼
    if cand.startswith(("I", "1")) and tgt.startswith("R") and len(cand) >= 5:
        variations.append(cand[1:])
    if cand.endswith(("I", "1")) and len(cand) >= 5:
        variations.append(cand[:-1])

    if len(cand) == len(tgt) + 1:
        variations.append(cand[1:])   # 去除前綴（如 'I'RCK9206）
        variations.append(cand[:-1])  # 去除後綴
    elif len(cand) == len(tgt) + 2:
        variations.append(cand[1:-1])
        variations.append(cand[2:])
    elif len(cand) == len(tgt) - 1:
        # 遺漏首碼 R（因左側螺絲、牌框或角度遮蔽，如 FG7213 / FGz2iz -> RFG7213）
        if tgt.startswith("R") and not cand.startswith("R"):
            variations.append("R" + cand)
        # 遺漏尾碼（如 RFG721 -> RFG7213）
        for digit in "0123456789":
            variations.append(cand + digit)
    elif len(cand) == len(tgt) - 2:
        if tgt.startswith("R") and not cand.startswith("R"):
            for digit in "0123456789":
                variations.append("R" + cand + digit)

    best_sim = 0.0
    best_corr = cand

    for cur_cand in variations:
        if cur_cand == tgt:
            return True, tgt, 1.0

        # 當長度相同時，執行逐字混淆矩陣對齊校正
        if len(cur_cand) == len(tgt):
            corr = []
            match_count = 0
            for c, t in zip(cur_cand, tgt):
                if c == t:
                    corr.append(t)
                    match_count += 1
                elif is_char_confusable(c, t):
                    corr.append(t)
                    match_count += 1
                else:
                    corr.append(c)
            sim = match_count / len(tgt)
            if sim > best_sim:
                best_sim = sim
                best_corr = "".join(corr)
                if match_count == len(tgt):
                    return True, tgt, 1.0

        # 子字串比對
        if tgt in cur_cand or cur_cand in tgt:
            sim = min(len(cur_cand), len(tgt)) / max(len(cur_cand), len(tgt))
            if sim > best_sim:
                best_sim = sim
                best_corr = tgt if sim >= 0.8 else cur_cand

        # Levenshtein 編輯距離
        if abs(len(cur_cand) - len(tgt)) <= 2:
            dist = compute_levenshtein_distance(cur_cand, tgt)
            max_len = max(len(cur_cand), len(tgt))
            sim = 1.0 - (dist / max_len)
            if sim > best_sim:
                best_sim = sim
                best_corr = tgt if sim >= 0.85 else cur_cand

    matched = best_sim >= 0.70
    final_text = tgt if matched else best_corr
    return matched, final_text, round(best_sim, 2)


class PlateGuard:
    """
    EasyOCR 工業級車牌辨識與訂單自動核對守門員。
    """

    def __init__(
        self,
        use_gpu: Optional[bool] = None,
        conf_threshold: Optional[float] = None,
        languages: Optional[List[str]] = None,
        enhancer=None
    ):
        """
        初始化 PlateGuard。

        Args:
            use_gpu: 是否啟用 GPU 加速（預設讀取 config 或 True）
            conf_threshold: 辨識信心門檻（預設 0.20）
            languages: 辨識語言列表，預設 ['en']
            enhancer: ZeroDCEEnhancer 實例（選填，用於暗光預先增強）
        """
        self.config = _load_plate_config()
        self.gpu_enabled = use_gpu if use_gpu is not None else self.config.get("gpu_enabled", True)
        self.conf_threshold = conf_threshold or self.config.get("confidence_threshold", 0.20)
        self.sim_threshold = self.config.get("match_similarity_threshold", 0.70)
        self.languages = languages or ["en"]
        self.enhancer = enhancer

        logger.info(f"正在初始化 EasyOCR (languages={self.languages}, gpu={self.gpu_enabled})...")
        self.reader = easyocr.Reader(self.languages, gpu=self.gpu_enabled)
        logger.info("EasyOCR 車牌辨識模組初始化完成")

    def read_text_raw(self, image: np.ndarray) -> List[Tuple[list, str, float]]:
        """執行 EasyOCR 辨識，返回 [(bbox, text, confidence), ...]"""
        return self.reader.readtext(image)

    def extract_plate_candidates(
        self,
        image: np.ndarray,
        car_bbox: Optional[list] = None
    ) -> List[dict]:
        """
        從影像中提取可能的車牌文字候選群。
        支援在整張影像與車身局部進行多尺度檢索。
        """
        candidates = []
        seen_texts = set()

        def add_candidate(text: str, conf: float, bbox):
            cleaned = clean_plate_text(text)
            if is_valid_plate_candidate(cleaned) and cleaned not in seen_texts:
                seen_texts.add(cleaned)
                candidates.append({
                    "raw_text": text,
                    "cleaned_text": cleaned,
                    "confidence": float(conf),
                    "bbox": bbox
                })

        # 1. 全圖 OCR
        raw_results = self.read_text_raw(image)
        for bbox, text, conf in raw_results:
            if conf >= self.conf_threshold:
                add_candidate(text, conf, bbox)

        # 2. 若傳入車身 bounding box，針對車身中下半部（牌照好發區域）進行裁切 OCR
        if car_bbox is not None:
            h, w = image.shape[:2]
            x1, y1, x2, y2 = [int(v) for v in car_bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            bh = y2 - y1
            crop_y1 = y1 + int(bh * 0.35)
            crop_y2 = y1 + int(bh * 0.95)
            cropped = image[crop_y1:crop_y2, x1:x2]

            if cropped.size > 0:
                crop_results = self.read_text_raw(cropped)
                for bbox, text, conf in crop_results:
                    if conf >= (self.conf_threshold * 0.75):
                        add_candidate(text, conf, bbox)

                # 局部車牌區域 2.0x 多尺度超解析度辨識（解決視角歪斜、字元微小沾黏問題）
                try:
                    resized_2x = cv2.resize(cropped, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
                    for bbox, text, conf in self.read_text_raw(resized_2x):
                        if conf >= (self.conf_threshold * 0.70):
                            add_candidate(text, conf, bbox)
                except Exception as e:
                    logger.debug(f"Crop 2.0x OCR 略過: {e}")

        # 3. 若仍未檢出且具備 enhancer (Zero-DCE)，嘗試增強後再檢測
        if not candidates and self.enhancer is not None:
            try:
                enh_result = self.enhancer.enhance(image, force=True)
                if enh_result.get("enhanced"):
                    enh_raw = self.read_text_raw(enh_result["image"])
                    for bbox, text, conf in enh_raw:
                        if conf >= (self.conf_threshold * 0.75):
                            add_candidate(text, conf, bbox)
            except Exception as e:
                logger.debug(f"Zero-DCE 輔助 OCR 略過: {e}")

        # 排序：優先以符合台灣車牌特徵（R開頭、長度6~7碼、高信心值）排序
        candidates.sort(
            key=lambda c: (
                c["cleaned_text"].startswith("R"),
                6 <= len(c["cleaned_text"]) <= 7,
                c["confidence"]
            ),
            reverse=True
        )

        return candidates

    def match_with_expected(
        self,
        candidates: List[dict],
        expected_plate: str
    ) -> Tuple[bool, float, Optional[dict], Optional[str]]:
        """
        將候選車牌列表與預期訂單車牌進行模糊比對。

        Returns:
            (is_matched, best_similarity, best_candidate, corrected_text)
            corrected_text: 經混淆矩陣校正後最接近目標的車牌文字
        """
        if not expected_plate:
            return False, 0.0, None, None

        target = clean_plate_text(expected_plate)
        if not target:
            return False, 0.0, None, None

        best_sim = 0.0
        best_cand = None
        best_corrected = None
        best_matched = False

        for cand in candidates:
            cleaned = cand["cleaned_text"]
            matched, corr, sim = normalize_with_target(cleaned, target)
            if sim > best_sim:
                best_sim = sim
                best_cand = cand
                best_corrected = corr
                best_matched = matched
                if matched and sim >= 0.99:
                    break

        return best_matched, round(best_sim, 2), best_cand, (expected_plate if best_matched else best_corrected)

    def verify_plate(
        self,
        image: np.ndarray,
        expected_plate: Optional[str] = None,
        image_type: Optional[int] = None,
        car_bbox: Optional[list] = None
    ) -> dict:
        """
        執行車牌辨識並與訂單車牌核對。

        Args:
            image: BGR 或 RGB 格式的 numpy array
            expected_plate: 預期訂單車牌（例如 "RCR-7661" 或 "RCG2235"）
            image_type: ImageType 代碼 (1-4: 車外, 5-9: 選拍, 10-11: 車內)
            car_bbox: 車輛邊界框 [x1, y1, x2, y2] (選填)

        Returns:
            dict: {
                "plate_detected": bool,
                "matched_order": bool,
                "detected_candidates": list[str],
                "best_match": str or None,
                "similarity": float,
                "confidence": float,
                "message": str,
                "reason": str or None,
                "latency_ms": float
            }
        """
        start_time = time.perf_counter()

        if image is None or image.size == 0:
            return {
                "plate_detected": False,
                "matched_order": False,
                "detected_candidates": [],
                "best_match": None,
                "similarity": 0.0,
                "confidence": 0.0,
                "message": "影像數據為空",
                "reason": "傳入無效或空的影像數據",
                "latency_ms": 0.0
            }

        # 提取車牌候選文字
        candidates = self.extract_plate_candidates(image, car_bbox=car_bbox)
        latency_ms = (time.perf_counter() - start_time) * 1000

        cand_texts = [c["cleaned_text"] for c in candidates]

        # ============================================================
        # 情況 A：未提供預期車牌（僅辨識當前車牌文字）
        # ============================================================
        if not expected_plate:
            plate_detected = len(candidates) > 0
            best_text = cand_texts[0] if cand_texts else None
            conf = candidates[0]["confidence"] if candidates else 0.0
            return {
                "plate_detected": plate_detected,
                "matched_order": True,
                "detected_candidates": cand_texts,
                "best_match": best_text,
                "similarity": 1.0 if plate_detected else 0.0,
                "confidence": round(conf, 2),
                "message": f"偵測到車牌: {best_text}" if plate_detected else "未偵測到車牌文字",
                "reason": None,
                "latency_ms": round(latency_ms, 1)
            }

        # ============================================================
        # 情況 B：提供預期車牌，進行自動核對比對
        # ============================================================
        matched, sim, best_cand, corrected_text = self.match_with_expected(candidates, expected_plate)

        # 若未成功匹配，且影像具備暗光特徵與 enhancer (Zero-DCE)，嘗試增強影像後再提取候選重新匹配
        if not matched and self.enhancer is not None:
            try:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                if np.mean(gray) < 80:
                    enh_res = self.enhancer.enhance(image, force=True)
                    if enh_res.get("enhanced"):
                        enh_cands = self.extract_plate_candidates(enh_res["image"], car_bbox=car_bbox)
                        if enh_cands:
                            m_enh, s_enh, c_enh, corr_enh = self.match_with_expected(enh_cands, expected_plate)
                            if s_enh > sim or m_enh:
                                for ec in enh_cands:
                                    if ec["cleaned_text"] not in cand_texts:
                                        candidates.append(ec)
                                        cand_texts.append(ec["cleaned_text"])
                                matched, sim, best_cand, corrected_text = m_enh, s_enh, c_enh, corr_enh
            except Exception as e:
                logger.debug(f"Zero-DCE 輔助重試略過: {e}")

        raw_text = best_cand["cleaned_text"] if best_cand else (cand_texts[0] if cand_texts else None)
        conf = best_cand["confidence"] if best_cand else (candidates[0]["confidence"] if candidates else 0.0)

        # 1. 成功核對符合
        if matched:
            target_clean = clean_plate_text(expected_plate)
            is_corrected = raw_text and raw_text != target_clean
            if is_corrected:
                msg = f"車牌核對無誤 (原始讀取: {raw_text}，成功校正為: {expected_plate})"
            else:
                msg = f"車牌核對無誤 ({expected_plate} 匹配成功)"

            return {
                "plate_detected": True,
                "matched_order": True,
                "detected_candidates": cand_texts,
                "best_match": expected_plate,
                "raw_ocr_text": raw_text,
                "corrected_plate": corrected_text or expected_plate,
                "plate_not_in_frame": False,
                "similarity": sim,
                "confidence": round(conf, 2),
                "message": msg,
                "reason": None,
                "latency_ms": round(latency_ms, 1)
            }

        # 2. 偵測到車牌文字，但與預約車牌不符（誤拍他人車輛）
        if candidates and not matched:
            suspicious_plate = cand_texts[0]
            return {
                "plate_detected": True,
                "matched_order": False,
                "detected_candidates": cand_texts,
                "best_match": suspicious_plate,
                "raw_ocr_text": raw_text,
                "corrected_plate": None,
                "plate_not_in_frame": False,
                "similarity": sim,
                "confidence": round(conf, 2),
                "message": f"偵測到車牌 ({suspicious_plate}) 與訂單車牌 ({expected_plate}) 不符",
                "reason": f"車輛車牌不一致，請確認是否為本次租用的指定車輛 ({expected_plate})",
                "latency_ms": round(latency_ms, 1)
            }

        # 3. 未偵測到車牌文字 — 車牌未入鏡
        # 依視角區分是否應嚴格要求車牌
        is_exterior = image_type in (1, 2, 3, 4) if image_type is not None else True

        if is_exterior:
            message = "⚠️ 車牌未入鏡，請調整角度確保車牌完整可見"
            reason = "照片中未拍攝到車牌區域，請移動至車頭正面或車尾正面拍攝，確保牌照清晰入鏡"
            matched_order = False
        else:
            # 車內照 (10-11) 或車況特寫 (5-9) 不強制要求車牌
            message = "此視角無需強制比對車牌"
            reason = None
            matched_order = True

        return {
            "plate_detected": False,
            "matched_order": matched_order,
            "detected_candidates": [],
            "best_match": None,
            "raw_ocr_text": None,
            "corrected_plate": None,
            "plate_not_in_frame": is_exterior,
            "similarity": 0.0,
            "confidence": 0.0,
            "message": message,
            "reason": reason,
            "latency_ms": round(latency_ms, 1)
        }
