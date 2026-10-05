"""
VisionGuard V3 — Pydantic 資料傳輸模型 (Schemas)
定義 API 請求/回應的資料結構驗證。
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


# ============================================================
# 環節一：拍照防呆 (Smart Capture Guard)
# ============================================================

class GuardCheckRequest(BaseModel):
    """環節一防呆檢查請求"""
    order_number: str = Field(..., description="訂單編號")
    vehicle_code: str = Field(..., description="車輛代碼，如 RCR-7661")
    image_type: int = Field(..., ge=1, le=11, description="ImageType 代碼 (1-4: 車外, 5-9: 回報, 10-11: 車內)")
    expected_plate: Optional[str] = Field(None, description="預期車牌號碼（用於 OCR 比對）")


class QualityResult(BaseModel):
    """基礎品質檢查結果"""
    passed: bool
    blur_score: float
    mean_brightness: float
    zero_dce_applied: bool = False
    warnings: list[str] = []


class CompositionResult(BaseModel):
    """構圖檢查結果"""
    passed: bool
    occupancy_ratio: Optional[float] = None
    message: Optional[str] = "構圖檢查完成"
    reason: Optional[str] = None


class PlateResult(BaseModel):
    """車牌辨識結果"""
    plate_detected: bool
    matched_order: bool
    detected_candidates: list[str] = []
    message: str


class GuardCheckResponse(BaseModel):
    """環節一完整回應"""
    overall_passed: bool
    quality: QualityResult
    composition: Optional[CompositionResult] = None
    plate: Optional[PlateResult] = None
    latency_ms: int
    enhanced_image_url: Optional[str] = None


# ============================================================
# 環節二：車損辨識 (Exterior Inspection)
# ============================================================

class DamageItem(BaseModel):
    """單一車損項目"""
    damage_id: str
    affected_part: str
    area_pixels: int = 0
    confidence: float = 0.0
    location_desc: str = ""
    bbox: list[int] = []
    polygon: list[list[int]] = []


class ExteriorInspectRequest(BaseModel):
    """車外車損辨識請求"""
    case_id: str
    order_number: str
    vehicle_code: str
    image_type: int = Field(..., ge=1, le=4)


class ExteriorInspectResponse(BaseModel):
    """車外車損辨識回應 (新車損判定 + 異常預警 + 大約範圍描述)"""
    damage_detected: bool
    has_new_damage: bool = False
    alert_triggered: bool = False
    alert_level: str = "NORMAL"
    damage_region: str = "無新車損"
    damage_locations: list[str] = []
    summary_description: str = ""
    severity: str = Field(..., description="none / minor / moderate / severe")
    damage_count: int = 0
    damages: list[DamageItem] = []
    ssim_score: Optional[float] = None
    latency_ms: int
    mask_overlay_url: Optional[str] = None


# ============================================================
# 環節二：車內整潔度 (Interior Inspection)
# ============================================================

class DetectedItem(BaseModel):
    """偵測到的車內物品"""
    category: str = Field(..., description="trash / stain / personal_belonging")
    item: str
    location: str
    confidence: float = 0.95
    bbox: list[int] = []


class InteriorInspectRequest(BaseModel):
    """車內整潔度辨識請求"""
    case_id: str
    order_number: str
    vehicle_code: str
    image_type: int = Field(..., ge=10, le=11)


class InteriorInspectResponse(BaseModel):
    """車內整潔度辨識回應"""
    cleanliness_level: str = Field(..., description="clean / fair / dirty")
    score: int = Field(..., ge=0, le=100)
    vlm_reasoning: str = ""
    detected_items: list[DetectedItem] = []
    cleaning_action_required: bool = False
    lost_items_detected: bool = False
    action_recommendation: str = ""
    latency_ms: int = 0
    hud_overlay_url: Optional[str] = None


# ============================================================
# 環節三：風險分流與工單
# ============================================================

class RiskEvaluation(BaseModel):
    """三色風險分流結果"""
    color: str = Field(..., description="green / yellow / red")
    priority: str = Field(..., description="low / medium / high / urgent")
    action: str
    reason: str


class WorkOrderInfo(BaseModel):
    """派工單明細結構"""
    work_order_id: str
    case_id: str
    vehicle_code: str
    category: str = "repair"
    category_zh: str = "維修整備"
    priority: str = "normal"
    ai_repair_guide: str = ""
    affected_part: str = ""
    estimated_labor_hours: float = 0.0
    estimated_parts: list[str] = []
    status: str = "open"


# ============================================================
# 整車全視角綜合評估 (Whole Vehicle Inspection & Evaluation)
# ============================================================

class VehicleAngleItem(BaseModel):
    """單一視角檢驗數據 (含取車基準與還車新車損比對)"""
    image_type: int = Field(..., ge=1, le=11, description="1-4: 車外, 10-11: 車內")
    view_name: Optional[str] = None
    guard_passed: bool = True
    blur_score: Optional[float] = None
    mean_brightness: Optional[float] = None
    damage_detected: bool = False
    has_new_damage: bool = False  # 是否為本次租車期間產生之新增車損
    is_pre_existing: bool = False # 是否為借車時已有之既有舊痕 (免責不扣分)
    damage_severity: str = "none"  # none / minor / moderate / severe
    damage_region: Optional[str] = None
    pre_photo_name: Optional[str] = None
    post_photo_name: Optional[str] = None
    cleanliness_level: Optional[str] = "clean"  # clean / fair / dirty
    cleanliness_score: Optional[int] = 100
    detected_items: list[dict] = []
    latency_ms: int = 0


class VehicleScoreBreakdown(BaseModel):
    """全車各維度評分明細 (外觀 45% + 車內座艙整潔 55%)"""
    exterior_score: int = Field(..., ge=0, le=100, description="車身外觀完整度評分 (比重 45%)")
    interior_score: int = Field(..., ge=0, le=100, description="車內座艙清潔度評分 (比重 55%)")
    exterior_weight: float = 0.45
    interior_weight: float = 0.55
    capture_quality_score: Optional[int] = Field(None, description="防呆合規 (車牌入鏡為還車強制門檻，不計入分數比重)")
    overall_vehicle_score: int = Field(..., ge=0, le=100, description="全車綜合健康度總分 (>=90 綠色合格, <70 紅色阻斷)")


class VehicleEvaluateRequest(BaseModel):
    """整車全視角綜合評估請求"""
    order_number: str
    vehicle_code: str
    case_id: Optional[str] = None
    angles: list[VehicleAngleItem] = []


class VehicleEvaluateResponse(BaseModel):
    """整車全視角綜合評估回應"""
    case_id: str
    order_number: str
    vehicle_code: str
    total_angles_evaluated: int
    overall_vehicle_score: int
    risk_level: str  # green / yellow / red
    priority: str    # low / medium / high / urgent
    action: str      # 自動歸檔 / 人工覆核車損 / 遺留物通知 / 維修派工 / 清潔派工
    block_next_booking: bool
    conclusion: str
    abnormalities: list[str] = []
    abnormal_notes: Optional[str] = "全車無異常"
    score_breakdown: VehicleScoreBreakdown
    damaged_angles: list[int] = []
    new_damaged_angles: list[int] = []
    pre_existing_angles: list[int] = []
    dirty_angles: list[int] = []
    work_order: Optional[WorkOrderInfo] = None
    latency_ms: int
    dispatch_card_url: Optional[str] = None


class DispatchEvaluateRequest(BaseModel):
    """三色分流評級請求"""
    case_id: str
    order_number: str
    vehicle_code: str
    exterior_results: list[dict] = []
    interior_results: list[dict] = []


class DispatchEvaluateResponse(BaseModel):
    """三色分流評級與工單產出回應"""
    case_id: str
    order_number: str
    vehicle_code: str
    risk_level: str
    priority: str
    action: str
    block_next_booking: bool
    reason: str
    overall_vehicle_score: Optional[int] = 100
    conclusion: Optional[str] = None
    work_order: Optional[WorkOrderInfo] = None
    latency_ms: int
    dispatch_card_url: Optional[str] = None


class WorkOrderCreate(BaseModel):
    """工單建立請求"""
    case_id: str
    vehicle_code: str
    category: str = Field(..., description="cleaning / repair")
    priority: str = Field(default="normal", description="normal / high / urgent")
    assigned_to: Optional[str] = None


class WorkOrderResponse(BaseModel):
    """工單回應"""
    work_order_id: str
    case_id: str
    vehicle_code: str
    category: str
    priority: str
    ai_repair_guide: Optional[str] = None
    estimated_labor_hours: Optional[float] = None
    estimated_parts: Optional[list[str]] = None
    status: str
    created_at: datetime


class WorkOrderUpdate(BaseModel):
    """工單狀態更新"""
    status: Optional[str] = None
    assigned_to: Optional[str] = None


# ============================================================
# 端到端完整流程
# ============================================================

class FullFlowRequest(BaseModel):
    """端到端完整流程請求"""
    order_number: str
    vehicle_code: str
    image_type: int
    expected_plate: Optional[str] = None
    trip_phase: str = Field(default="return", description="pickup / return")


class FullFlowResponse(BaseModel):
    """端到端完整流程回應"""
    case_id: str
    guard: GuardCheckResponse
    inspection: Optional[ExteriorInspectResponse | InteriorInspectResponse] = None
    risk: Optional[RiskEvaluation] = None
    work_order: Optional[WorkOrderResponse] = None
    total_latency_ms: int


# ============================================================
# 通用查詢
# ============================================================

class CaseListQuery(BaseModel):
    """案件列表查詢參數"""
    risk_level: Optional[str] = None
    vehicle_code: Optional[str] = None
    limit: int = Field(default=50, le=200)


class VehicleTimeline(BaseModel):
    """車輛時間軸條目"""
    case_id: str
    image_type: int
    risk_level: str
    damage_severity: Optional[str] = None
    cleanliness_level: Optional[str] = None
    created_at: datetime
