"""
VisionGuard V3 — Supabase 資料庫連線模組
提供與 Supabase PostgreSQL、Storage、Realtime 的統一連線介面。
"""

import os
from pathlib import Path
from functools import lru_cache
from supabase import create_client, Client
from dotenv import load_dotenv

# 載入環境變數 (支援當前目錄與專案目錄)
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
if _ENV_PATH.exists():
    load_dotenv(_ENV_PATH)
else:
    load_dotenv()


def is_supabase_configured() -> bool:
    """檢查 Supabase 連線資訊是否已在環境變數中設定"""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    return bool(url and key and "your-project-id" not in url)


def _clean_supabase_url(url: str | None) -> str:
    """清理 URL 避免包含結尾斜線或 /rest/v1 後綴"""
    if not url:
        return ""
    cleaned = url.strip().rstrip("/")
    if cleaned.endswith("/rest/v1"):
        cleaned = cleaned[:-8].rstrip("/")
    return cleaned


@lru_cache(maxsize=1)
def get_supabase_client() -> Client:
    """
    取得 Supabase 客戶端單例。
    使用 service_role_key 以繞過 RLS（後端專用）。
    """
    url = _clean_supabase_url(os.getenv("SUPABASE_URL"))
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        raise ValueError(
            "請在 .env 中設定 SUPABASE_URL 和 SUPABASE_SERVICE_ROLE_KEY\n"
            "參考 .env.example 設定。"
        )
    return create_client(url, key)


async def insert_inspection(data: dict) -> dict:
    """新增一筆檢驗案件"""
    client = get_supabase_client()
    result = client.table("inspections").insert(data).execute()
    return result.data[0] if result.data else {}


async def get_inspection(case_id: str) -> dict | None:
    """依據 case_id 取得單一檢驗案件紀錄"""
    client = get_supabase_client()
    result = client.table("inspections").select("*").eq("case_id", case_id).limit(1).execute()
    return result.data[0] if result.data else None


async def get_inspections_by_case(case_id: str) -> list[dict]:
    """取得同一案件的全部角度檢驗紀錄。"""
    client = get_supabase_client()
    base_result = (
        client.table("inspections")
        .select("*")
        .eq("case_id", case_id)
        .execute()
    )
    angle_result = (
        client.table("inspections")
        .select("*")
        .like("case_id", f"{case_id}-ANGLE-%")
        .execute()
    )
    records = (base_result.data or []) + (angle_result.data or [])
    return sorted(records, key=lambda item: item.get("image_type") or 0)


async def get_unreviewed_cases(limit: int = 50) -> list[dict]:
    """取得待覆核案件（黃色 + 紅色）"""
    client = get_supabase_client()
    result = (
        client.table("inspections")
        .select("*")
        .in_("risk_level", ["yellow", "red"])
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return result.data or []


async def get_recent_inspections(limit: int = 50) -> list[dict]:
    """取得近期所有檢驗案件，供後端管理 API 使用。"""
    client = get_supabase_client()
    result = (
        client.table("inspections")
        .select("*")
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return result.data or []


# ============================================================
# 工單 CRUD
# ============================================================

async def insert_work_order(data: dict) -> dict:
    """新增一筆工單"""
    client = get_supabase_client()
    result = client.table("work_orders").insert(data).execute()
    return result.data[0] if result.data else {}


async def get_work_orders(status: str | None = None, limit: int = 50) -> list[dict]:
    """取得工單列表"""
    client = get_supabase_client()
    query = client.table("work_orders").select("*").order("created_at", desc=True).limit(limit)
    if status:
        query = query.eq("status", status)
    result = query.execute()
    return result.data or []


async def update_work_order(work_order_id: str, data: dict) -> dict:
    """更新工單狀態"""
    client = get_supabase_client()
    result = (
        client.table("work_orders")
        .update(data)
        .eq("work_order_id", work_order_id)
        .execute()
    )
    return result.data[0] if result.data else {}


async def update_work_order_status(work_order_id: str, status: str, assigned_to: str | None = None) -> dict:
    """更新工單狀態及指派人員"""
    update_data = {"status": status}
    if assigned_to:
        update_data["assigned_to"] = assigned_to
    if status == "completed":
        import datetime
        update_data["completed_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    return await update_work_order(work_order_id, update_data)


# ============================================================
# 車輛 CRUD
# ============================================================

async def get_vehicles(status: str | None = None, limit: int = 50) -> list[dict]:
    """取得車輛列表"""
    client = get_supabase_client()
    query = client.table("vehicles").select("*").order("updated_at", desc=True).limit(limit)
    if status:
        query = query.eq("status", status)
    result = query.execute()
    return result.data or []


async def update_vehicle_status(vehicle_code: str, status: str) -> dict:
    """更新車輛狀態"""
    client = get_supabase_client()
    result = (
        client.table("vehicles")
        .update({"status": status})
        .eq("vehicle_code", vehicle_code)
        .execute()
    )
    return result.data[0] if result.data else {}


# ============================================================
# Supabase Storage 圖片上傳
# ============================================================

BUCKET_NAME = "inspection-images"


async def upload_image(file_path: str, storage_path: str) -> str:
    """
    上傳圖片到 Supabase Storage 並回傳公開 URL。

    Args:
        file_path: 本地檔案路徑
        storage_path: Storage 中的路徑（如 "inspections/case123/pre.jpg"）

    Returns:
        圖片的公開 URL
    """
    client = get_supabase_client()
    with open(file_path, "rb") as f:
        client.storage.from_(BUCKET_NAME).upload(
            storage_path,
            f,
            file_options={"content-type": "image/jpeg"}
        )
    # 取得公開 URL
    result = client.storage.from_(BUCKET_NAME).get_public_url(storage_path)
    return result
