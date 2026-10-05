"""
iGuard — FastAPI 高效推論服務主程式 (Main Application)
支援本機 RTX 5090 算力原生加速、全環節 AI 檢驗 API、Supabase 雲端資料庫/儲存同步與 Realtime 即時推送。
"""

import os
import sys
import time
import uuid
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# 加入專案根目錄並載入環境變數
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from dotenv import load_dotenv
_ENV_PATH = BASE_DIR / ".env"
if _ENV_PATH.exists():
    load_dotenv(_ENV_PATH)
else:
    load_dotenv()

# 設定日誌
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("iguard.api")

# 引入 GPU 管理器與各模組
from app.gpu_manager import gpu_manager
from app.guard.quality import check_basic_quality, compute_quality_score
from app.database import (
    is_supabase_configured,
    get_supabase_client,
    insert_inspection,
    get_inspection,
    get_unreviewed_cases,
    insert_work_order,
    get_work_orders,
    update_work_order_status,
    get_vehicles,
    update_vehicle_status,
    upload_image,
    BUCKET_NAME
)
from app.schemas import (
    GuardCheckResponse, QualityResult, CompositionResult, PlateResult,
    ExteriorInspectResponse, DamageItem,
    InteriorInspectResponse, DetectedItem,
    DispatchEvaluateRequest, DispatchEvaluateResponse, WorkOrderInfo,
    VehicleAngleItem, VehicleScoreBreakdown, VehicleEvaluateRequest, VehicleEvaluateResponse
)


# -----------------------------------------------------------------------------
# 輔助函式：影像安全載入與暫存
# -----------------------------------------------------------------------------
def read_upload_to_bgr(upload_bytes: bytes) -> np.ndarray:
    """將上傳之二進位圖檔安全轉為 OpenCV BGR 格式（含 EXIF 自動轉正）"""
    nparr = np.frombuffer(upload_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="無法解碼上傳之影像檔案，請確認是否為有效圖檔 (JPG/PNG/JFIF)")
    return img


def save_temp_image(bgr_img: np.ndarray, filename_prefix: str) -> str:
    """儲存暫存圖檔於 outputs/temp 並回傳檔案絕對路徑"""
    temp_dir = BASE_DIR / "outputs" / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{filename_prefix}_{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}.jpg"
    out_path = str(temp_dir / fname)
    is_ok, buf = cv2.imencode(".jpg", bgr_img)
    if is_ok:
        with open(out_path, "wb") as f:
            f.write(buf)
    return out_path


# -----------------------------------------------------------------------------
# FastAPI Lifespan 管理 (初始化模型與預熱)
# -----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("================================================================================")
    logger.info("🚗 iGuard — AI 視覺智能檢驗服務正在啟動...")
    logger.info(f"⚡ 硬體運算設備: {gpu_manager.device_name} (總顯存: {gpu_manager.total_vram_gb:.1f} GB)")
    
    # 檢查 Supabase 設定狀態
    if is_supabase_configured():
        logger.info("🟢 Supabase 雲端資料庫與儲存庫已就緒 (Full Cloud Mode)")
    else:
        logger.warning("🟡 Supabase 尚未設定金鑰，系統將以本機模擬模式 (Local Mode) 運行。")
        logger.warning("   如需啟用雲端同步，請於 visionguard-v3-5090/.env 設定 SUPABASE_URL 與金鑰。")
    
    # 背景非同步預熱輕量 Guard 模組
    import threading
    threading.Thread(target=gpu_manager.warmup_guard_models, daemon=True).start()
    logger.info("================================================================================")
    
    yield
    
    logger.info("🛑 iGuard 服務正在關閉...")


# 建立 FastAPI 實例
app = FastAPI(
    title="iGuard — AI 視覺智能檢驗與多模態派工 API",
    description="以 RTX 5090 本機原生算力驅動之端到端車身檢驗、微光增強、像素級車損多邊形標註、車內多模態推理與三色分流平台。",
    version="3.0.0",
    lifespan=lifespan
)

# 設定 CORS
allowed_origins_env = os.getenv("CORS_ORIGINS", "*")
origins = [o.strip() for o in allowed_origins_env.split(",") if o.strip()]
if "*" in origins or not origins:
    origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 掛載靜態檔案目錄供本機圖檔存取
outputs_dir = BASE_DIR / "outputs"
outputs_dir.mkdir(parents=True, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=str(outputs_dir)), name="outputs")

# 掛載真實車輛影像資料庫
dataset_dir = BASE_DIR.parent / "iRent_dataset_new"
if dataset_dir.exists():
    app.mount("/dataset", StaticFiles(directory=str(dataset_dir)), name="dataset")

# 掛載 iGuard 前端靜態網頁 (GitHub Pages 雙前端)
frontend_dir = BASE_DIR / "frontend"
if frontend_dir.exists():
    app.mount("/frontend", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")


# =============================================================================
# 系統監控與健康度端點
# =============================================================================
@app.get("/", tags=["System"])
def root_info():
    return {
        "service": "iGuard AI Engine",
        "version": "3.0.0",
        "status": "online",
        "gpu": gpu_manager.device_name,
        "supabase_connected": is_supabase_configured(),
        "frontend_url": "/frontend/",
        "admin_url": "/frontend/admin.html",
        "docs_url": "/docs"
    }


@app.get("/health", tags=["System"])
def health_check():
    """檢視 RTX 5090 硬體狀態、VRAM 顯存配置與模型就緒指標"""
    gpu_stat = gpu_manager.get_status()
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "supabase_configured": is_supabase_configured(),
        "gpu": gpu_stat
    }


@app.get("/api/v1/config/public", tags=["System"])
def get_public_config():
    """
    回傳前端安全連線所需之公開參數（僅提供 Supabase URL 與 Anon Key），
    絕不洩漏 SERVICE_ROLE_KEY 或系統內部密鑰，避免前端程式碼硬編碼敏感資訊。
    """
    url = os.getenv("SUPABASE_URL", "")
    anon_key = os.getenv("SUPABASE_ANON_KEY", "")
    return {
        "supabase_url": url if "your-project-id" not in url else "",
        "supabase_anon_key": anon_key if "your-anon-key" not in anon_key else "",
        "configured": bool(url and anon_key and "your-project-id" not in url)
    }


@app.get("/api/v1/dataset/vehicle-folders", tags=["Dataset"])
def get_dataset_vehicle_folders():
    """
    掃描並回傳真實車輛相片資料夾清單，包含 01 正常無損車輛、02 索賠車損車輛與 03 租賃訂單車輛。
    """
    target_dir = BASE_DIR.parent / "iRent_dataset_new"
    if not target_dir.exists():
        return {"total": 0, "folders": []}
    
    folders_list = []
    # 掃描各子分類目錄
    for cat_dir in sorted(target_dir.iterdir()):
        if not cat_dir.is_dir() or cat_dir.name.startswith("."):
            continue
        cat_name = cat_dir.name
        # 僅掃描包含車輛的資料夾
        if not any(k in cat_name for k in ["正常", "索賠", "租賃", "ORDER"]):
            continue
        
        for v_dir in sorted(cat_dir.iterdir()):
            if not v_dir.is_dir() or v_dir.name.startswith("."):
                continue
            v_name = v_dir.name
            img_files = []
            for img_path in sorted(v_dir.iterdir()):
                if img_path.is_file() and img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".jfif", ".webp"]:
                    import urllib.parse
                    encoded_url = f"/dataset/{urllib.parse.quote(cat_name)}/{urllib.parse.quote(v_name)}/{urllib.parse.quote(img_path.name)}"
                    img_files.append({
                        "name": img_path.name,
                        "url": encoded_url,
                        "size": img_path.stat().st_size
                    })
            if img_files:
                folders_list.append({
                    "id": f"{cat_name}/{v_name}",
                    "vehicle": v_name,
                    "category": cat_name,
                    "count": len(img_files),
                    "images": img_files
                })

    return {"total": len(folders_list), "folders": folders_list}


# =============================================================================
# 環節一：拍照防呆 (Step 1~4)
# =============================================================================
@app.post("/api/v1/guard/check", response_model=GuardCheckResponse, tags=["Inspection Step 1~4"])
async def guard_check(
    file: UploadFile = File(..., description="車身照片檔案"),
    order_number: str = Form(..., description="租賃訂單編號"),
    vehicle_code: str = Form(..., description="車輛代碼，如 RCR-7661"),
    image_type: int = Form(..., ge=1, le=11, description="視角代碼 (1-4:車外, 10-11:車內)"),
    expected_plate: Optional[str] = Form(None, description="預期車牌號碼")
):
    """
    執行環節一拍照防呆守門員：
    1. Laplacian 模糊度與亮度檢驗
    2. Zero-DCE 微光偵測與增強
    3. YOLOv11x-seg 構圖與車身佔比驗證
    4. EasyOCR GPU 車牌字元比對
    """
    t0 = time.time()
    contents = await file.read()
    bgr = read_upload_to_bgr(contents)

    # 1. 基礎品質評估
    q_res = check_basic_quality(bgr)
    blur_score = q_res.get("blur_score", 0.0)
    mean_brightness = q_res.get("mean_brightness", 0.0)
    q_score = compute_quality_score(q_res)
    needs_enhance = q_res.get("needs_enhancement", False)

    # 2. 構圖驗證 (YOLOv11x-seg)
    comp_res = gpu_manager.composition_guard.verify_viewpoint(bgr, image_type)
    comp_ok = comp_res.get("passed", False)
    car_box = comp_res.get("primary_box")
    occ = comp_res.get("occupancy_ratio", 0.0)

    # 3. 車牌辨識與比對
    p_res = gpu_manager.plate_guard.verify_plate(bgr, expected_plate or vehicle_code, image_type, car_bbox=car_box)
    plate_match = p_res.get("matched_order", False)
    plate_detected = p_res.get("plate_detected", False)
    plate_not_in_frame = p_res.get("plate_not_in_frame", False)
    best_cand = p_res.get("best_match", "未檢出")

    # 總體判定邏輯
    overall_passed = q_res.get("passed", False) and comp_ok
    if image_type not in (5, 10, 11) and not plate_not_in_frame:
        if expected_plate and not plate_match:
            overall_passed = False

    latency_ms = int((time.time() - t0) * 1000)

    # 上傳至 Supabase Storage (若已配置)
    image_url = None
    if is_supabase_configured():
        try:
            temp_path = save_temp_image(bgr, f"guard_{vehicle_code}")
            storage_path = f"guard/{order_number}/{vehicle_code}_{image_type}_{int(time.time())}.jpg"
            image_url = await upload_image(temp_path, storage_path)
        except Exception as e:
            logger.warning(f"[Guard] Upload image to Supabase Storage failed: {e}")

    return GuardCheckResponse(
        overall_passed=overall_passed,
        quality=QualityResult(
            passed=q_res.get("passed", False),
            blur_score=blur_score,
            mean_brightness=mean_brightness,
            zero_dce_applied=needs_enhance,
            warnings=q_res.get("warnings", [])
        ),
        composition=CompositionResult(
            passed=comp_ok,
            occupancy_ratio=occ,
            message=comp_res.get("reason") or ("構圖合規" if comp_ok else "主體佔比不合規"),
            reason=comp_res.get("reason")
        ),
        plate=PlateResult(
            plate_detected=plate_detected,
            matched_order=plate_match,
            detected_candidates=p_res.get("candidates", []),
            message=f"讀出車牌: {best_cand}"
        ),
        latency_ms=latency_ms,
        enhanced_image_url=image_url
    )


# =============================================================================
# 環節二(A)：車外前後比對與新車損標註 (Step 5)
# =============================================================================
@app.post("/api/v1/exterior/inspect", response_model=ExteriorInspectResponse, tags=["Inspection Step 5"])
async def exterior_inspect(
    pre_file: UploadFile = File(..., description="借車照片 (Pre-rental)"),
    post_file: UploadFile = File(..., description="還車照片 (Post-rental)"),
    case_id: str = Form(..., description="檢驗案件編號"),
    order_number: str = Form(..., description="租賃訂單編號"),
    vehicle_code: str = Form(..., description="車輛代碼"),
    image_type: int = Form(..., ge=1, le=4, description="視角 (1:左前, 2:右前, 3:左後, 4:右後)")
):
    """
    執行環節二(A) 車外前後比對：
    1. SuperPoint + LightGlue 幾何自適應對齊
    2. SSIM 結構特徵差分定位
    3. Meta SAM 2 Large 像素級多邊形車損分割
    4. 自動歸因受損部位與嚴重度分級
    """
    t0 = time.time()
    pre_bytes = await pre_file.read()
    post_bytes = await post_file.read()

    pre_bgr = read_upload_to_bgr(pre_bytes)
    post_bgr = read_upload_to_bgr(post_bytes)

    # 1. 深度幾何對齊 (align_target="post" 保持還車原圖無畸變)
    align_res = gpu_manager.aligner.align_pair(pre_bgr, post_bgr, align_target="post")
    pre_aligned = align_res["aligned_image"]
    post_img = align_res["target_image"]
    M = align_res.get("affine_matrix")
    inliers = align_res.get("inliers_count", 0)

    # 2. 車身主體遮罩 (YOLOv11x-seg)
    h_pre, w_pre = pre_bgr.shape[:2]
    h_post, w_post = post_img.shape[:2]

    res_post = gpu_manager.composition_guard.model.predict(post_img, imgsz=640, conf=0.15, verbose=False)[0]
    c_boxes_post = [i for i, b in enumerate(res_post.boxes) if int(b.cls[0].item()) in [2, 5, 7]]
    if c_boxes_post and res_post.masks is not None:
        best_i = max(c_boxes_post, key=lambda i: (res_post.boxes[i].xyxy[0][2] - res_post.boxes[i].xyxy[0][0]) * (res_post.boxes[i].xyxy[0][3] - res_post.boxes[i].xyxy[0][1]))
        m_post = cv2.resize(res_post.masks.data[best_i].cpu().numpy(), (w_post, h_post), interpolation=cv2.INTER_LINEAR)
        car_mask_post = (m_post > 0.5).astype(np.uint8) * 255
    else:
        car_mask_post = np.ones((h_post, w_post), dtype=np.uint8) * 255

    res_pre = gpu_manager.composition_guard.model.predict(pre_bgr, imgsz=640, conf=0.15, verbose=False)[0]
    c_boxes_pre = [i for i, b in enumerate(res_pre.boxes) if int(b.cls[0].item()) in [2, 5, 7]]
    if c_boxes_pre and res_pre.masks is not None:
        best_i = max(c_boxes_pre, key=lambda i: (res_pre.boxes[i].xyxy[0][2] - res_pre.boxes[i].xyxy[0][0]) * (res_pre.boxes[i].xyxy[0][3] - res_pre.boxes[i].xyxy[0][1]))
        m_pre = cv2.resize(res_pre.masks.data[best_i].cpu().numpy(), (w_pre, h_pre), interpolation=cv2.INTER_LINEAR)
        car_mask_pre = (m_pre > 0.5).astype(np.uint8) * 255
    else:
        car_mask_pre = np.ones((h_pre, w_pre), dtype=np.uint8) * 255

    if M is not None:
        car_mask_pre_aligned = cv2.warpAffine(car_mask_pre, M, (w_post, h_post), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    else:
        car_mask_pre_aligned = cv2.resize(car_mask_pre, (w_post, h_post), interpolation=cv2.INTER_NEAREST)

    plate_candidates = gpu_manager.plate_guard.extract_plate_candidates(post_img)

    # 3. SSIM 結構差異分析
    diff_res = gpu_manager.aligner.compute_structural_diff(
        pre_image=pre_aligned,
        post_image_aligned=post_img,
        vehicle_mask=car_mask_post,
        vehicle_mask_src_aligned=car_mask_pre_aligned,
        plate_boxes=plate_candidates,
        image_type=image_type,
        diff_threshold=55,
        min_contour_area=295
    )

    # 4. SAM 2 像素級車損多邊形標註
    insp_res = gpu_manager.exterior_engine.analyze_damage(
        pre_image=pre_aligned,
        post_image_aligned=post_img,
        candidate_boxes=diff_res["candidate_boxes"],
        yolo_car_mask=car_mask_post,
        image_type=image_type,
        diff_mask=diff_res["diff_mask"],
        raw_joint_diff=diff_res["raw_joint_diff"],
        ssim_score=diff_res["ssim_score"]
    )

    latency_ms = int((time.time() - t0) * 1000)

    # 5. 繪製診斷圖
    overlay_img = gpu_manager.exterior_engine.render_damage_overlay(post_img, insp_res)
    has_damage = insp_res.get("damage_detected", False)
    severity = insp_res.get("severity", "none")

    damages_list = []
    parts_list = []
    for d in insp_res.get("damages", []):
        part_name = d.get("affected_part", "車身外部")
        parts_list.append(part_name)
        damages_list.append(DamageItem(
            damage_id=d.get("damage_id", str(uuid.uuid4())[:8]),
            affected_part=part_name,
            area_pixels=d.get("area_pixels", 0),
            confidence=d.get("confidence", 0.95),
            location_desc=d.get("location_desc", ""),
            bbox=d.get("bbox", []),
            polygon=d.get("polygon", [])
        ))

    view_names = {1: "左前", 2: "右前", 3: "左後", 4: "右後"}
    v_zh = view_names.get(image_type, "車身外部")
    parts_unique = list(dict.fromkeys(parts_list))
    damage_region = f"{v_zh} ({', '.join(parts_unique)})" if has_damage else "無新車損"

    alert_level = "ALERT_SEVERE" if has_damage and severity == "severe" else ("ALERT_NEW_DAMAGE" if has_damage else "NORMAL")

    # 上傳至 Supabase Storage (若已配置)
    overlay_url = None
    if is_supabase_configured():
        try:
            temp_path = save_temp_image(overlay_img, f"damage_{case_id}")
            storage_path = f"damages/{order_number}/{case_id}_{image_type}.jpg"
            overlay_url = await upload_image(temp_path, storage_path)

            # 寫入 inspections 表
            await insert_inspection({
                "case_id": case_id,
                "order_number": order_number,
                "vehicle_code": vehicle_code,
                "image_type": image_type,
                "ssim_score": float(diff_res["ssim_score"]),
                "damage_detected": has_damage,
                "damage_severity": severity,
                "damage_count": insp_res.get("damage_count", 0),
                "damage_parts": str(parts_unique),
                "risk_level": "red" if severity == "severe" else ("yellow" if has_damage else "green"),
                "risk_reason": damage_region,
                "gpu_latency_ms": latency_ms,
                "mask_overlay_url": overlay_url
            })
        except Exception as e:
            logger.warning(f"[Exterior] Save to Supabase failed: {e}")

    return ExteriorInspectResponse(
        damage_detected=has_damage,
        has_new_damage=has_damage,
        alert_triggered=has_damage,
        alert_level=alert_level,
        damage_region=damage_region,
        damage_locations=parts_unique,
        summary_description=f"在{v_zh}發現 {len(parts_unique)} 處新增損傷" if has_damage else "全車身正常，未發現新車損",
        severity=severity,
        damage_count=insp_res.get("damage_count", 0),
        damages=damages_list,
        ssim_score=diff_res["ssim_score"],
        latency_ms=latency_ms,
        mask_overlay_url=overlay_url
    )


# =============================================================================
# 環節二(B)：車內整潔度多模態推理 (Step 6)
# =============================================================================
@app.post("/api/v1/interior/inspect", response_model=InteriorInspectResponse, tags=["Inspection Step 6"])
async def interior_inspect(
    file: UploadFile = File(..., description="車內照片"),
    case_id: str = Form(..., description="檢驗案件編號"),
    order_number: str = Form(..., description="租賃訂單編號"),
    vehicle_code: str = Form(..., description="車輛代碼"),
    image_type: int = Form(..., ge=10, le=11, description="10:前車內, 11:後車內")
):
    """
    執行環節二(B) 車內整潔度多模態推理：
    調用 Qwen2.5-VL-3B-Instruct 進行物品識別（垃圾/失物/污漬）、整潔打分與派工處置建議。
    """
    t0 = time.time()
    contents = await file.read()
    bgr = read_upload_to_bgr(contents)

    res = gpu_manager.interior_engine.inspect_interior(
        image_input=bgr,
        image_type=image_type,
        case_id=case_id,
        order_number=order_number,
        vehicle_code=vehicle_code
    )

    latency_ms = int((time.time() - t0) * 1000)

    # 繪製診斷圖
    hud_bgr = gpu_manager.interior_engine.render_interior_overlay(bgr, res)

    # 上傳至 Supabase Storage (若已配置)
    hud_url = None
    if is_supabase_configured():
        try:
            temp_path = save_temp_image(hud_bgr, f"interior_{case_id}")
            storage_path = f"interior/{order_number}/{case_id}_{image_type}.jpg"
            hud_url = await upload_image(temp_path, storage_path)

            await insert_inspection({
                "case_id": case_id,
                "order_number": order_number,
                "vehicle_code": vehicle_code,
                "image_type": image_type,
                "cleanliness_level": res.get("cleanliness_level", "clean"),
                "cleanliness_score": res.get("score", 95),
                "vlm_reasoning": res.get("vlm_reasoning", ""),
                "detected_items": res.get("detected_items", []),
                "cleaning_action_required": res.get("cleaning_action_required", False),
                "risk_level": "red" if res.get("cleanliness_level") == "dirty" else ("yellow" if res.get("lost_items") else "green"),
                "gpu_latency_ms": latency_ms,
                "mask_overlay_url": hud_url
            })
        except Exception as e:
            logger.warning(f"[Interior] Save to Supabase failed: {e}")

    items = [
        DetectedItem(
            category=it.get("category", "other"),
            item=it.get("item", "物品"),
            location=it.get("location", "車內"),
            bbox=it.get("bbox", [])
        )
        for it in res.get("detected_items", [])
    ]

    return InteriorInspectResponse(
        cleanliness_level=res.get("cleanliness_level", "clean"),
        score=res.get("score", 95),
        vlm_reasoning=res.get("vlm_reasoning", ""),
        detected_items=items,
        cleaning_action_required=res.get("cleaning_action_required", False),
        lost_items_detected=bool(res.get("lost_items")),
        action_recommendation=res.get("action_recommendation", "車況整潔無虞，無縫放行。"),
        latency_ms=latency_ms,
        hud_overlay_url=hud_url
    )


# =============================================================================
# 環節三：營運三色分流評級與工單派發 (Step 7)
# =============================================================================
@app.post("/api/v1/dispatch/evaluate", response_model=DispatchEvaluateResponse, tags=["Inspection Step 7"])
async def dispatch_evaluate(req: DispatchEvaluateRequest):
    """
    執行環節三 營運預警與三色派工中樞：
    1. 綜合車外車損與車內多模態檢驗結果
    2. 評定 紅 (阻斷預約)、黃 (人工覆核/失物推播)、綠 (自動放行)
    3. 自動生成 VLM 結構化 SOP 智能修復工單
    4. 輸出 1280x720 戰情室派工卡片
    """
    t0 = time.time()
    risk = gpu_manager.dispatcher.evaluate_trip_risk(
        exterior_results=req.exterior_results,
        interior_results=req.interior_results
    )

    work_order_data = None
    if risk["color"] in ["red", "yellow"]:
        ext_first = req.exterior_results[0] if req.exterior_results else None
        int_first = req.interior_results[0] if req.interior_results else None
        work_order_data = gpu_manager.dispatcher.generate_work_order(
            case_id=req.case_id,
            vehicle_code=req.vehicle_code,
            risk_evaluation=risk,
            exterior_result=ext_first,
            interior_result=int_first
        )

    # 繪製戰情室派工卡片
    card_bgr = gpu_manager.dispatcher.render_dispatch_card(
        risk_evaluation=risk,
        work_order=work_order_data,
        case_id=req.case_id,
        order_number=req.order_number,
        vehicle_code=req.vehicle_code
    )

    latency_ms = int((time.time() - t0) * 1000)

    # 同步至 Supabase 資料庫與儲存庫 (若已配置)
    card_url = None
    if is_supabase_configured():
        try:
            temp_path = save_temp_image(card_bgr, f"card_{req.case_id}")
            storage_path = f"dispatch/{req.order_number}/{req.case_id}_dispatch_card.jpg"
            card_url = await upload_image(temp_path, storage_path)

            if work_order_data:
                await insert_work_order({
                    "work_order_id": work_order_data["work_order_id"],
                    "case_id": req.case_id,
                    "vehicle_code": req.vehicle_code,
                    "category": work_order_data["category"],
                    "priority": work_order_data["priority"],
                    "ai_repair_guide": work_order_data["ai_repair_guide"],
                    "estimated_labor_hours": work_order_data["estimated_labor_hours"],
                    "estimated_parts": work_order_data["estimated_parts"],
                    "status": "open",
                    "evidence_images": {"dispatch_card": card_url}
                })

            # 更新車輛狀態 (若為紅燈阻斷，車輛狀態切換為 repair/clean)
            if risk["block_next_booking"]:
                new_status = "repairing" if "車損" in risk["reason"] else "cleaning"
                await update_vehicle_status(req.vehicle_code, new_status)
        except Exception as e:
            logger.warning(f"[Dispatch] Save to Supabase failed: {e}")

    wo_resp = None
    if work_order_data:
        wo_resp = WorkOrderInfo(
            work_order_id=work_order_data["work_order_id"],
            case_id=req.case_id,
            vehicle_code=req.vehicle_code,
            category=work_order_data["category"],
            category_zh=work_order_data["category_zh"],
            priority=work_order_data["priority"],
            ai_repair_guide=work_order_data["ai_repair_guide"],
            affected_part=work_order_data["affected_part"],
            estimated_labor_hours=work_order_data["estimated_labor_hours"],
            estimated_parts=work_order_data["estimated_parts"],
            status="open"
        )

    return DispatchEvaluateResponse(
        case_id=req.case_id,
        order_number=req.order_number,
        vehicle_code=req.vehicle_code,
        risk_level=risk["color"],
        priority=risk["priority"],
        action=risk["action"],
        block_next_booking=risk["block_next_booking"],
        reason=risk["reason"],
        overall_vehicle_score=95 if risk["color"] == "green" else (75 if risk["color"] == "yellow" else 50),
        conclusion=f"【綜合分流：{risk['color'].upper()}】{risk['reason']}",
        work_order=wo_resp,
        latency_ms=latency_ms,
        dispatch_card_url=card_url
    )


@app.post("/api/v1/vehicle/evaluate", response_model=VehicleEvaluateResponse, tags=["Inspection Whole Vehicle"])
async def evaluate_vehicle(req: VehicleEvaluateRequest):
    """
    執行整台車全視角綜合評估 (Whole Vehicle Unit)：
    1. 整合全車所有檢驗角度 (車外視角 1~4、車內視角 10~11 等)
    2. 依外觀完整度、內裝整潔度、拍攝品質綜合核算全車健康評分 (0-100)
    3. 產出全車權威綜合結論 (Conclusion) 與三色營運處置行動
    4. 依需生成 VLM 智能工單 (SOP) 與戰情室派工卡片
    """
    t0 = time.time()
    angles_dicts = [a.model_dump() for a in req.angles]
    eval_res = gpu_manager.dispatcher.evaluate_whole_vehicle(
        vehicle_code=req.vehicle_code,
        order_number=req.order_number,
        angles=angles_dicts,
        case_id=req.case_id
    )

    latency_ms = int((time.time() - t0) * 1000)

    # 繪製戰情室派工卡片
    card_bgr = gpu_manager.dispatcher.render_dispatch_card(
        risk_evaluation={
            "color": eval_res["risk_level"],
            "priority": eval_res["priority"],
            "action": eval_res["action"],
            "reason": eval_res["reason"],
            "block_next_booking": eval_res["block_next_booking"]
        },
        work_order=eval_res["work_order"],
        case_id=eval_res["case_id"],
        order_number=req.order_number,
        vehicle_code=req.vehicle_code
    )

    card_url = None
    if is_supabase_configured():
        try:
            temp_path = save_temp_image(card_bgr, f"vehicle_card_{eval_res['case_id']}")
            storage_path = f"dispatch/{req.order_number}/{eval_res['case_id']}_vehicle_card.jpg"
            card_url = await upload_image(temp_path, storage_path)

            wo_data = eval_res["work_order"]
            if wo_data:
                await insert_work_order({
                    "work_order_id": wo_data["work_order_id"],
                    "case_id": eval_res["case_id"],
                    "vehicle_code": req.vehicle_code,
                    "category": wo_data["category"],
                    "priority": wo_data["priority"],
                    "ai_repair_guide": wo_data["ai_repair_guide"],
                    "estimated_labor_hours": wo_data["estimated_labor_hours"],
                    "estimated_parts": wo_data["estimated_parts"],
                    "status": "open",
                    "evidence_images": {"dispatch_card": card_url}
                })

            if eval_res["block_next_booking"]:
                new_status = "repairing" if eval_res["damaged_angles"] else "cleaning"
                await update_vehicle_status(req.vehicle_code, new_status)
        except Exception as e:
            logger.warning(f"[Vehicle Evaluate] Save to Supabase failed: {e}")

    wo_resp = None
    wo_data = eval_res.get("work_order")
    if wo_data:
        wo_resp = WorkOrderInfo(
            work_order_id=wo_data.get("work_order_id", f"WO-{eval_res['case_id'][-6:]}"),
            case_id=eval_res["case_id"],
            vehicle_code=req.vehicle_code,
            category=wo_data.get("category", "repair"),
            category_zh=wo_data.get("category_zh", "維修整備"),
            priority=wo_data.get("priority", "normal"),
            ai_repair_guide=wo_data.get("ai_repair_guide", "執行標準整備流程。"),
            affected_part=wo_data.get("affected_part", "車身"),
            estimated_labor_hours=float(wo_data.get("estimated_labor_hours", 1.0)),
            estimated_parts=wo_data.get("estimated_parts", ["標準耗材"]),
            status="open"
        )

    return VehicleEvaluateResponse(
        case_id=eval_res["case_id"],
        order_number=req.order_number,
        vehicle_code=req.vehicle_code,
        total_angles_evaluated=eval_res["total_angles_evaluated"],
        overall_vehicle_score=eval_res["overall_vehicle_score"],
        risk_level=eval_res["risk_level"],
        priority=eval_res["priority"],
        action=eval_res["action"],
        block_next_booking=eval_res["block_next_booking"],
        conclusion=eval_res["conclusion"],
        abnormalities=eval_res.get("abnormalities", []),
        abnormal_notes=eval_res.get("abnormal_notes", "全車無異常"),
        score_breakdown=VehicleScoreBreakdown(**eval_res["score_breakdown"]),
        damaged_angles=eval_res["damaged_angles"],
        dirty_angles=eval_res["dirty_angles"],
        work_order=wo_resp,
        latency_ms=latency_ms,
        dispatch_card_url=card_url
    )


# =============================================================================
# 營運管理端查詢 API (Supabase 整合)
# =============================================================================
@app.get("/api/v1/inspections", tags=["Management"])
async def list_inspections(limit: int = Query(50, ge=1, le=200)):
    """查詢近期車輛檢驗歷史案件"""
    if not is_supabase_configured():
        return {"data": [], "message": "Supabase 未設定，此為本機模式"}
    try:
        cases = await get_unreviewed_cases(limit=limit)
        return {"data": cases}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/work-orders", tags=["Management"])
async def list_work_orders(
    status: Optional[str] = Query(None, description="工單狀態: open / in_progress / completed")
):
    """查詢智能維修/清潔派工工單"""
    if not is_supabase_configured():
        return {"data": [], "message": "Supabase 未設定，此為本機模式"}
    try:
        orders = await get_work_orders(status=status)
        return {"data": orders}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.patch("/api/v1/work-orders/{work_order_id}", tags=["Management"])
async def update_work_order(
    work_order_id: str,
    status: str = Query(..., description="open / in_progress / completed / cancelled"),
    assigned_to: Optional[str] = Query(None, description="指派工程師/專人")
):
    """更新工單狀態（例如工程師接單或完成修復）"""
    if not is_supabase_configured():
        return {"data": {"work_order_id": work_order_id, "status": status}, "message": "本機模擬模式"}
    try:
        updated = await update_work_order_status(work_order_id, status, assigned_to)
        return {"data": updated}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/vehicles", tags=["Management"])
async def list_vehicles(status: Optional[str] = Query(None)):
    """查詢車隊即時車況"""
    if not is_supabase_configured():
        return {"data": [], "message": "Supabase 未設定，此為本機模式"}
    try:
        vehicles = await get_vehicles(status=status)
        return {"data": vehicles}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------------------------------------------------------
# 直接執行主程式 (用於 python app/main.py 本地測試)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    logger.info(f"啟動 Uvicorn 伺服器於 http://{host}:{port} ...")
    uvicorn.run("app.main:app", host=host, port=port, reload=True)
