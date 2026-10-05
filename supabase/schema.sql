-- ============================================================
-- iGuard — Supabase PostgreSQL Schema
-- 在 Supabase Dashboard > SQL Editor 中執行此腳本
-- ============================================================

-- 啟用 UUID 擴展
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================
-- 1. 檢驗案件表 (inspections)
-- ============================================================
CREATE TABLE inspections (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    case_id TEXT NOT NULL UNIQUE,
    order_number TEXT NOT NULL,
    vehicle_code TEXT NOT NULL,
    image_type INTEGER NOT NULL,

    -- 環節一：拍照防呆結果
    quality_passed BOOLEAN NOT NULL DEFAULT FALSE,
    blur_score REAL,
    mean_brightness REAL,
    zero_dce_applied BOOLEAN DEFAULT FALSE,
    plate_verified BOOLEAN,
    plate_detected_text TEXT,
    composition_passed BOOLEAN,
    car_occupancy_ratio REAL,

    -- 環節二：車損辨識結果（車外）
    ssim_score REAL,
    damage_detected BOOLEAN DEFAULT FALSE,
    damage_severity TEXT CHECK (damage_severity IN ('none', 'minor', 'moderate', 'severe')),
    damage_count INTEGER DEFAULT 0,
    damage_parts TEXT,                    -- JSON 字串，如 ["右前保險桿", "左後葉子板"]
    sam2_mask_polygons JSONB,             -- SAM2 多邊形遮罩座標

    -- 環節二：車內整潔度結果
    cleanliness_level TEXT CHECK (cleanliness_level IN ('clean', 'fair', 'dirty')),
    cleanliness_score INTEGER,
    vlm_reasoning TEXT,                   -- Qwen2.5-VL 自然語言推理
    detected_items JSONB,                 -- 偵測到的物品列表
    cleaning_action_required BOOLEAN DEFAULT FALSE,

    -- 環節三：分流結果
    risk_level TEXT NOT NULL CHECK (risk_level IN ('green', 'yellow', 'red')),
    risk_reason TEXT,
    risk_action TEXT,

    -- 效能指標
    gpu_latency_ms INTEGER,

    -- 圖片路徑（Supabase Storage）
    pre_image_url TEXT,
    post_image_url TEXT,
    mask_overlay_url TEXT,

    -- 時間戳
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 索引加速查詢
CREATE INDEX idx_inspections_vehicle ON inspections(vehicle_code);
CREATE INDEX idx_inspections_risk ON inspections(risk_level);
CREATE INDEX idx_inspections_created ON inspections(created_at DESC);
CREATE INDEX idx_inspections_order ON inspections(order_number);

-- ============================================================
-- 2. 智能派工工單表 (work_orders)
-- ============================================================
CREATE TABLE work_orders (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    work_order_id TEXT NOT NULL UNIQUE,
    case_id TEXT NOT NULL REFERENCES inspections(case_id),
    vehicle_code TEXT NOT NULL,
    category TEXT NOT NULL CHECK (category IN ('cleaning', 'repair')),
    priority TEXT NOT NULL CHECK (priority IN ('normal', 'high', 'urgent')),

    -- VLM 自動生成內容
    ai_repair_guide TEXT,
    estimated_labor_hours REAL,
    estimated_parts JSONB,                -- ["前保險桿固定扣", "專用色號原廠漆料"]

    -- 工單狀態
    status TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'in_progress', 'completed', 'cancelled')),
    assigned_to TEXT,

    -- 證據圖片
    evidence_images JSONB,                -- {"pre": "url", "post": "url", "mask": "url"}

    -- 時間戳
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);

CREATE INDEX idx_work_orders_status ON work_orders(status);
CREATE INDEX idx_work_orders_vehicle ON work_orders(vehicle_code);
CREATE INDEX idx_work_orders_priority ON work_orders(priority);

-- ============================================================
-- 3. 車輛狀態表 (vehicles)
-- ============================================================
CREATE TABLE vehicles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    vehicle_code TEXT NOT NULL UNIQUE,
    plate_number TEXT,
    model TEXT,
    status TEXT NOT NULL DEFAULT 'available' CHECK (status IN ('available', 'rented', 'inspecting', 'cleaning', 'repairing')),
    total_trips INTEGER DEFAULT 0,
    last_inspection_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_vehicles_status ON vehicles(status);
CREATE INDEX idx_vehicles_code ON vehicles(vehicle_code);

-- ============================================================
-- 4. 使用者拍照品質積分 (capture_scores)
-- ============================================================
CREATE TABLE capture_scores (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id TEXT NOT NULL,
    total_uploads INTEGER DEFAULT 0,
    quality_pass_rate REAL DEFAULT 1.0,
    avg_blur_score REAL,
    avg_composition_score REAL,
    trust_level TEXT DEFAULT 'standard' CHECK (trust_level IN ('bronze', 'silver', 'gold', 'diamond', 'standard')),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE UNIQUE INDEX idx_capture_scores_user ON capture_scores(user_id);

-- ============================================================
-- 5. 自動更新 updated_at 觸發器
-- ============================================================
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_inspections_updated_at
    BEFORE UPDATE ON inspections
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_work_orders_updated_at
    BEFORE UPDATE ON work_orders
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_vehicles_updated_at
    BEFORE UPDATE ON vehicles
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_capture_scores_updated_at
    BEFORE UPDATE ON capture_scores
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================
-- 6. 啟用 Realtime（即時推播功能）
-- ============================================================
ALTER PUBLICATION supabase_realtime ADD TABLE inspections;
ALTER PUBLICATION supabase_realtime ADD TABLE work_orders;

-- ============================================================
-- 7. Row Level Security (RLS) — 基礎策略
-- ============================================================
ALTER TABLE inspections ENABLE ROW LEVEL SECURITY;
ALTER TABLE work_orders ENABLE ROW LEVEL SECURITY;
ALTER TABLE vehicles ENABLE ROW LEVEL SECURITY;
ALTER TABLE capture_scores ENABLE ROW LEVEL SECURITY;

-- 不建立公開或一般 authenticated 使用者政策。
-- FastAPI 後端使用 service_role_key，會在受控環境中繞過 RLS；
-- 前端不得直接讀寫這些營運資料表。
