/**
 * iGuard — Backend API Client Module
 * Communicates with FastAPI backend on localhost:8000, with intelligent fallback simulation.
 */

const DEFAULT_API_BASE = "https://moses-kingston-stockholm-graduate.trycloudflare.com";

class IGuardAPI {
  constructor() {
    let queryApi = null;
    try {
      if (typeof window !== "undefined" && window.location && window.location.search) {
        const urlParams = new URLSearchParams(window.location.search);
        queryApi = urlParams.get("api");
      }
    } catch (e) {}

    if (queryApi) {
      this.apiBase = queryApi.trim().replace(/\/+$/, "");
      localStorage.setItem("iguard_api_base", this.apiBase);
    } else {
      this.apiBase = localStorage.getItem("iguard_api_base") || DEFAULT_API_BASE;
    }
    this.isBackendOnline = false;
    this.checkHealth();
  }

  setApiBase(url) {
    this.apiBase = (url || "").trim().replace(/\/+$/, "");
    localStorage.setItem("iguard_api_base", this.apiBase);
    return this.checkHealth();
  }

  async quickFetch(url, timeoutMs = 2800, headers = { "Accept": "application/json" }) {
    if (!url) return null;
    try {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), timeoutMs);
      const res = await fetch(url, {
        method: "GET",
        headers: headers,
        signal: controller.signal
      });
      clearTimeout(timer);
      if (res.ok) {
        return await res.json();
      }
    } catch (e) {}
    return null;
  }

  async discoverApiFromSupabase() {
    try {
      let discoveredUrl = null;

      // 1. Direct REST fetch to Supabase (instant, zero SDK dependency)
      const restRes = await this.quickFetch(
        "https://uzpmwyeirkgdweuuptbr.supabase.co/rest/v1/vehicles?vehicle_code=eq.SYSTEM_API_URL&select=plate_number",
        2500,
        {
          "apikey": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InV6cG13eWVpcmtnZHdldXVwdGJyIiwicm9sZSI6ImFub24iLCJpYXQiOjE3OTA3NTkyODgsImV4cCI6MjEwNjMzNTI4OH0.VQbB3jgYyAslq2bP1YNPYM5BDj2mlGNHvUQ8zf5Cv0g",
          "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InV6cG13eWVpcmtnZHdldXVwdGJyIiwicm9sZSI6ImFub24iLCJpYXQiOjE3OTA3NTkyODgsImV4cCI6MjEwNjMzNTI4OH0.VQbB3jgYyAslq2bP1YNPYM5BDj2mlGNHvUQ8zf5Cv0g",
          "Accept": "application/json"
        }
      );

      if (Array.isArray(restRes) && restRes.length > 0 && restRes[0].plate_number) {
        discoveredUrl = restRes[0].plate_number.trim().replace(/\/+$/, "");
      } else if (window.iguardSupabase && window.iguardSupabase.client) {
        const res = await window.iguardSupabase.client
          .table("vehicles")
          .select("plate_number")
          .eq("vehicle_code", "SYSTEM_API_URL")
          .limit(1);
        if (res.data && res.data.length > 0 && res.data[0].plate_number) {
          discoveredUrl = res.data[0].plate_number.trim().replace(/\/+$/, "");
        }
      }

      if (discoveredUrl && discoveredUrl.startsWith("http")) {
        console.log("[iGuard Discovery] Found 5090 API in Supabase:", discoveredUrl);
        const healthData = await this.quickFetch(`${discoveredUrl}/health`, 3000);
        if (healthData) {
          this.apiBase = discoveredUrl;
          localStorage.setItem("iguard_api_base", discoveredUrl);
          this.isBackendOnline = true;
          console.log("[iGuard Discovery] Successfully connected to 5090:", discoveredUrl);
          return { online: true, data: healthData };
        } else {
          console.warn("[iGuard Discovery] 5090 API from Supabase did not respond to /health:", discoveredUrl);
        }
      }
    } catch (err) {
      console.warn("[iGuard Discovery] Supabase lookup error:", err);
    }
    return null;
  }

  async checkHealth() {
    const isRemoteBrowser = typeof window !== "undefined" && window.location && window.location.hostname !== "localhost" && window.location.hostname !== "127.0.0.1";
    const localApiBase = "http://127.0.0.1:8000";

    // When the frontend itself is served locally, never wait for a stale
    // Cloudflare URL before checking the local FastAPI process.
    if (!isRemoteBrowser) {
      const localData = await this.quickFetch(`${localApiBase}/health`, 1200);
      if (localData) {
        this.apiBase = localApiBase;
        localStorage.setItem("iguard_api_base", localApiBase);
        this.isBackendOnline = true;
        return { online: true, data: localData };
      }
    }

    // 1. 若在遠端其他電腦/手機上，優先向 Supabase 查詢 5090 當前活動穿透網址
    if (isRemoteBrowser) {
      const discovery = await this.discoverApiFromSupabase();
      if (discovery && discovery.online) {
        return discovery;
      }
    }

    // 2. 嘗試目前網址 (2.5 秒超時，避免卡死)
    const curData = await this.quickFetch(`${this.apiBase}/health`, 2500);
    if (curData) {
      this.isBackendOnline = true;
      return { online: true, data: curData };
    }

    // 3. 若在本機 5090 環境，嘗試 127.0.0.1:8000 直連 (1.2 秒超時)
    if (!isRemoteBrowser && this.apiBase !== localApiBase) {
      const localData = await this.quickFetch(`${localApiBase}/health`, 1200);
      if (localData) {
        this.apiBase = localApiBase;
        localStorage.setItem("iguard_api_base", localApiBase);
        this.isBackendOnline = true;
        return { online: true, data: localData };
      }
    }

    // 4. 若上述皆未成功，非本機環境下再次嘗試 Supabase 尋標
    if (!isRemoteBrowser) {
      const discovery = await this.discoverApiFromSupabase();
      if (discovery && discovery.online) {
        return discovery;
      }
    }

    // 5. 嘗試預設網址
    if (this.apiBase !== DEFAULT_API_BASE && DEFAULT_API_BASE) {
      const defData = await this.quickFetch(`${DEFAULT_API_BASE}/health`, 2500);
      if (defData) {
        this.apiBase = DEFAULT_API_BASE;
        localStorage.setItem("iguard_api_base", DEFAULT_API_BASE);
        this.isBackendOnline = true;
        return { online: true, data: defData };
      }
    }

    this.isBackendOnline = false;
    return {
      online: false,
      data: {
        status: "simulated",
        gpu: {
          device_name: "NVIDIA GeForce RTX 5090 (Simulation Mode)",
          total_vram_gb: 32.0,
          free_vram_gb: 28.4
        }
      }
    };
  }

  async checkGuard(file, orderNumber, vehicleCode, imageType, expectedPlate = "") {
    if (this.isBackendOnline) {
      try {
        const formData = new FormData();
        formData.append("file", file);
        formData.append("order_number", orderNumber);
        formData.append("vehicle_code", vehicleCode);
        formData.append("image_type", imageType);
        if (expectedPlate) formData.append("expected_plate", expectedPlate);

        const res = await fetch(`${this.apiBase}/api/v1/guard/check`, {
          method: "POST",
          body: formData
        });
        if (res.ok) return await res.json();
      } catch (err) {
        console.warn("API checkGuard failed, using simulation:", err);
      }
    }

    await new Promise((r) => setTimeout(r, 450));
    const isInterior = imageType >= 10;
    return {
      passed: true,
      case_id: `CASE-${Date.now().toString().slice(-6)}`,
      order_number: orderNumber,
      vehicle_code: vehicleCode,
      image_type: Number(imageType),
      quality: {
        blur_score: 285.4,
        mean_brightness: 118.2,
        is_clear: true,
        is_lit: true,
        quality_score: 95
      },
      zero_dce: {
        applied: false,
        reason: "足夠亮度，無需微光增強"
      },
      composition: {
        passed: true,
        car_detected: !isInterior,
        occupancy_ratio: isInterior ? 0.0 : 0.68,
        center_offset: [0.02, -0.01],
        bbox: isInterior ? null : [85, 120, 1140, 680]
      },
      plate: {
        verified: true,
        plate_found: true,
        plate_text: expectedPlate || "RCR-7661",
        expected_plate: expectedPlate,
        match: true,
        confidence: 0.985
      },
      feedback_messages: [
        "構圖比例極佳 (68.0%)",
        "清晰度良好 (Laplacian: 285.4)",
        "車牌比對一致: " + (expectedPlate || "RCR-7661")
      ]
    };
  }

  async inspectExterior(preFile, postFile, orderNumber, vehicleCode, imageType, caseId = "") {
    if (this.isBackendOnline) {
      try {
        const formData = new FormData();
        formData.append("pre_file", preFile);
        formData.append("post_file", postFile);
        formData.append("case_id", caseId || `CASE-${Date.now().toString().slice(-6)}`);
        formData.append("order_number", orderNumber);
        formData.append("vehicle_code", vehicleCode);
        formData.append("image_type", imageType);

        const res = await fetch(`${this.apiBase}/api/v1/exterior/inspect`, {
          method: "POST",
          body: formData
        });
        if (res.ok) return await res.json();
      } catch (err) {
        console.warn("API inspectExterior failed, using simulation:", err);
      }
    }

    await new Promise((r) => setTimeout(r, 720));
    return {
      case_id: `CASE-${Date.now().toString().slice(-6)}`,
      order_number: orderNumber,
      vehicle_code: vehicleCode,
      image_type: Number(imageType),
      alignment: {
        method: "SuperPoint + LightGlue",
        inliers_count: 342,
        alignment_score: 0.94
      },
      structural_diff: {
        ssim_score: 0.812,
        anomaly_regions_count: 1
      },
      damages_detected: [
        {
          id: 1,
          part_name: "右前保險桿下緣 (Front Bumper)",
          severity: "moderate",
          area_pixels: 4820,
          confidence: 0.92,
          bbox: [240, 310, 420, 480],
          polygon_points: [[240, 310], [410, 325], [420, 470], [250, 460]]
        }
      ],
      overall_severity: "moderate",
      diagnostics_image_url: null,
      recommendation: "檢出 1 處新增擦傷 (面積: 4820px)，建議轉派工單"
    };
  }

  async inspectInterior(file, orderNumber, vehicleCode, imageType, caseId = "") {
    if (this.isBackendOnline) {
      try {
        const formData = new FormData();
        formData.append("file", file);
        formData.append("case_id", caseId || `CASE-${Date.now().toString().slice(-6)}`);
        formData.append("order_number", orderNumber);
        formData.append("vehicle_code", vehicleCode);
        formData.append("image_type", imageType);

        const res = await fetch(`${this.apiBase}/api/v1/interior/inspect`, {
          method: "POST",
          body: formData
        });
        if (res.ok) return await res.json();
      } catch (err) {
        console.warn("API inspectInterior failed, using simulation:", err);
      }
    }

    await new Promise((r) => setTimeout(r, 850));
    return {
      case_id: `CASE-${Date.now().toString().slice(-6)}`,
      order_number: orderNumber,
      vehicle_code: vehicleCode,
      image_type: Number(imageType),
      cleanliness_level: "fair",
      cleanliness_score: 72,
      detected_items: [
        {
          category: "trash",
          name: "手搖飲料紙杯與吸管套",
          location: "中央扶手杯架",
          confidence: 0.94
        }
      ],
      vlm_reasoning: "後座腳踏墊有輕微泥沙痕跡，中央扶手杯架遺留空飲料紙杯，需輕度吸塵與清潔。",
      dispatch_action: "cleaning",
      suggested_sop: "指示整備人員攜帶無線車用吸塵器與抹布進行 5 分鐘快洗整潔。"
    };
  }

  async evaluateDispatch(payload) {
    if (this.isBackendOnline) {
      try {
        const res = await fetch(`${this.apiBase}/api/v1/dispatch/evaluate`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
        if (res.ok) return await res.json();
      } catch (err) {
        console.warn("API evaluateDispatch failed, using simulation:", err);
      }
    }

    await new Promise((r) => setTimeout(r, 200));
    const tier = payload.damage_detected ? "red" : (payload.cleanliness_level === "dirty" ? "red" : (payload.cleanliness_level === "fair" ? "yellow" : "green"));
    return {
      case_id: payload.case_id || `CASE-${Date.now().toString().slice(-6)}`,
      order_number: payload.order_number || "ORD-2026-8891",
      vehicle_code: payload.vehicle_code || "RCR-7661",
      triage_tier: tier,
      tier_display: tier === "red" ? "紅色警報 (立即派工整備)" : (tier === "yellow" ? "黃色待審 (人工快速複核)" : "綠色放行 (全自動直通上架)"),
      work_order_required: tier === "red",
      work_order: tier === "red" ? {
        work_order_id: `WO-${Date.now().toString().slice(-6)}`,
        category: payload.damage_detected ? "repair" : "cleaning",
        priority: "urgent",
        ai_repair_guide: "檢測到右前保險桿損傷，建議使用 2000 號水砂紙研磨後進行補漆作業。",
        estimated_labor_hours: 1.5,
        estimated_parts: ["專用色號原廠漆料", "保險桿固定扣"]
      } : null,
      auto_released: tier === "green"
    };
  }

  async evaluateVehicle(payload) {
    if (!this.isBackendOnline) {
      await this.checkHealth();
    }
    if (this.isBackendOnline) {
      try {
        const res = await fetch(`${this.apiBase}/api/v1/vehicle/evaluate`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
        if (res.ok) return await res.json();
      } catch (err) {
        console.warn("API evaluateVehicle failed, using simulation:", err);
      }
    }

    // High-Fidelity Simulation Fallback (Strictly adhering to 45% Ext / 55% Int / Thresholds >=90/70)
    await new Promise((r) => setTimeout(r, 600));

    const vehicleCode = payload.vehicle_code || "RCR-7661";
    const orderNumber = payload.order_number || "ORD-AUTO-8891";
    const angles = payload.angles || [];
    const caseId = payload.case_id || `CASE-${Date.now().toString().slice(-6)}`;

    const exteriorAngles = angles.filter(a => [1, 2, 3, 4, 5, 6, 7, 8, 9].includes(Number(a.image_type)));
    const interiorAngles = angles.filter(a => [10, 11].includes(Number(a.image_type)));

    const damagedAngles = [];
    const newDamagedAngles = [];
    const preExistingAngles = [];
    const dirtyAngles = [];
    const abnormalities = [];

    const viewNames = {
      1: "左前方", 2: "右前方", 3: "左後方", 4: "右後方",
      5: "正前方", 6: "正後方", 7: "車身左側", 8: "車身右側", 9: "車頂",
      10: "前車內", 11: "後車內"
    };

    // 1. Exterior Score (45% weight) — 僅針對「本次新增車損」扣分！排除借車前既有舊痕！
    let exteriorScore = 100;
    exteriorAngles.forEach(a => {
      const itype = Number(a.image_type);
      const sev = (a.damage_severity || "none").toLowerCase();
      const hasDmg = a.damage_detected || sev !== "none";
      const isNewDmg = a.has_new_damage || (hasDmg && !a.is_pre_existing);
      const isPre = a.is_pre_existing;

      if (isNewDmg && sev !== "none") {
        newDamagedAngles.push(itype);
        damagedAngles.push(itype);
        if (sev === "severe") {
          exteriorScore = Math.min(exteriorScore, 30);
          abnormalities.push(`【外觀】檢出本次新增車損：${a.damage_region || viewNames[itype]} (嚴重車損，非借車前舊痕)`);
        } else if (sev === "moderate") {
          exteriorScore = Math.min(exteriorScore, 65);
          abnormalities.push(`【外觀】檢出本次新增車損：${a.damage_region || viewNames[itype]} (中度擦傷，非借車前舊痕)`);
        } else {
          exteriorScore = Math.min(exteriorScore, 82);
          abnormalities.push(`【外觀】檢出本次新增車損：${a.damage_region || viewNames[itype]} (輕微細痕，非借車前舊痕)`);
        }
      } else if (isPre) {
        preExistingAngles.push(itype);
        abnormalities.push(`【外觀】${a.damage_region || viewNames[itype]} 經取還車對齊比對確認為借車前既有舊痕，判定免責不扣分`);
      }
    });

    // 2. Interior Score (55% weight)
    let interiorScore = 100;
    if (interiorAngles.length > 0) {
      const scores = interiorAngles.map(a => Number(a.cleanliness_score ?? (a.cleanliness_level === "dirty" ? 50 : (a.cleanliness_level === "fair" ? 75 : 100))));
      interiorScore = Math.round(scores.reduce((sum, v) => sum + v, 0) / scores.length);
      interiorAngles.forEach(a => {
        const itype = Number(a.image_type);
        const lvl = (a.cleanliness_level || "clean").toLowerCase();
        if (lvl === "dirty" || (a.cleanliness_score && a.cleanliness_score < 65)) {
          dirtyAngles.push(itype);
          abnormalities.push(`【內裝】${viewNames[itype]}座艙髒亂 (評分 ${a.cleanliness_score || 50}分)`);
        }
        if (a.detected_items && a.detected_items.length > 0) {
          a.detected_items.forEach(it => {
            if (it.category === "trash" && lvl !== "clean") {
              abnormalities.push(`【內裝】檢出${it.item || "垃圾髒污"}(${it.location || "車內"})`);
            } else if (it.category === "personal_belonging") {
              abnormalities.push(`【內裝】疑似遺留私人物品：${it.item || "物品"}(${it.location || "車內"})`);
            }
          });
        }
      });
    }

    // 3. Composite Calculation
    const rawComposite = Math.round(0.45 * exteriorScore + 0.55 * interiorScore);

    // Determine risk level & calibrated score
    let riskLevel = "green";
    let priority = "low";
    let action = "自動歸檔";
    let blockNextBooking = false;
    let overallVehicleScore = rawComposite;

    if (newDamagedAngles.length > 0 && exteriorScore <= 65) {
      riskLevel = "red";
      priority = "urgent";
      action = "維修派工";
      blockNextBooking = true;
      overallVehicleScore = Math.min(rawComposite, 68);
    } else if (dirtyAngles.length > 0 && interiorScore < 65) {
      riskLevel = "red";
      priority = "urgent";
      action = "清潔派工";
      blockNextBooking = true;
      overallVehicleScore = Math.min(rawComposite, 68);
    } else if (newDamagedAngles.length > 0 || dirtyAngles.length > 0 || rawComposite < 90) {
      riskLevel = "yellow";
      priority = "medium";
      action = newDamagedAngles.length > 0 ? "人工覆核車損" : "人工覆核清潔";
      blockNextBooking = false;
      overallVehicleScore = Math.min(89, Math.max(70, rawComposite));
    } else {
      riskLevel = "green";
      priority = "low";
      action = "自動歸檔";
      blockNextBooking = false;
      overallVehicleScore = Math.max(90, Math.min(100, rawComposite));
    }

    let abnormalNotes = "全車無異常（取車 vs 還車前後比對無新增車損）";
    if (abnormalities.length > 0) {
      abnormalNotes = `異常說明：${abnormalities.join("；")}`;
    } else if (preExistingAngles.length > 0) {
      abnormalNotes = "全車無新增車損（已排除借車前既有舊痕，判定免責）";
    }

    let conclusion = "";
    if (riskLevel === "green") {
      conclusion = `【全車綜合判定：綠色合格 (Green Pass) | 全車總評分: ${overallVehicleScore}/100 分】\n全車無異常。系統比對取車與還車照片後，確認外觀沒有新增損傷，車內也保持整潔且沒有遺留物品。車況良好，系統已自動完成檢查並開放下一位租客預約。`;
    } else if (riskLevel === "yellow") {
      conclusion = `【全車綜合判定：黃色待審 (Yellow Review) | 全車總評分: ${overallVehicleScore}/100 分】\n異常項目：${abnormalities.join("；")}。車況基本安全但需進一步核實，系統已標註相關特徵，由營運後台 30 秒內快速確認。`;
    } else {
      conclusion = `【全車綜合判定：紅色阻斷 (Red Alert) | 全車總評分: ${overallVehicleScore}/100 分】\n異常項目：${abnormalities.join("；")}。檢出本次租車產生之新增車損。`;
    }

    let workOrder = null;
    if (riskLevel === "red" || riskLevel === "yellow") {
      const isRepair = damagedAngles.length > 0;
      workOrder = {
        work_order_id: `WO-${Date.now().toString().slice(-6)}`,
        case_id: caseId,
        vehicle_code: vehicleCode,
        category: isRepair ? "repair" : "cleaning",
        category_zh: isRepair ? "鈑噴維修" : "內裝清潔",
        priority: riskLevel === "red" ? "urgent" : "normal",
        ai_repair_guide: isRepair
          ? "【局部快修指引】建議使用粗蠟與 2000 號砂紙研磨平整後，採用原廠色號色漆進行局部點漆補修。"
          : "【內裝深度清潔作業指南】全面清查前後座椅夾縫與腳踏墊，進行微型吸塵與除臭擦拭作業。",
        affected_part: isRepair ? "右前保險桿" : "座艙置杯架與腳踏墊",
        estimated_labor_hours: isRepair ? 1.5 : 0.8,
        estimated_parts: isRepair ? ["專用色號原廠漆料", "保險桿固定扣"] : ["專用環保清潔劑", "抗菌擦拭布"],
        status: "open"
      };
    }

    return {
      case_id: caseId,
      order_number: orderNumber,
      vehicle_code: vehicleCode,
      total_angles_evaluated: angles.length,
      overall_vehicle_score: overallVehicleScore,
      risk_level: riskLevel,
      priority: priority,
      action: action,
      block_next_booking: blockNextBooking,
      conclusion: conclusion,
      abnormalities: abnormalities,
      abnormal_notes: abnormalNotes,
      score_breakdown: {
        exterior_score: exteriorScore,
        interior_score: interiorScore,
        exterior_weight: 0.45,
        interior_weight: 0.55,
        capture_quality_score: 100,
        overall_vehicle_score: overallVehicleScore
      },
      damaged_angles: damagedAngles,
      new_damaged_angles: newDamagedAngles,
      pre_existing_angles: preExistingAngles,
      dirty_angles: dirtyAngles,
      work_order: workOrder,
      latency_ms: 280,
      dispatch_card_url: null
    };
  }

  // ==========================================
  // Real Dataset Vehicle Folders Scanner & Loader
  // ==========================================
  async getVehicleFolders() {
    if (this.isBackendOnline) {
      try {
        const res = await fetch(`${this.apiBase}/api/v1/dataset/vehicle-folders`);
        if (res.ok) {
          const data = await res.json();
          if (data && data.folders && data.folders.length > 0) {
            return data.folders;
          }
        }
      } catch (err) {
        console.warn("Failed to fetch vehicle folders from backend, using local catalog:", err);
      }
    }

    // Static / Local Fallback Catalog with Real File Paths
    const basePath = "../../iRent_dataset_new";
    return [
      {
        id: "03_ORDER-001",
        vehicle: "ORDER-001",
        category: "03_租賃訂單車輛_Hims500 (500台)",
        count: 10,
        images: [
          { name: "取車_01.png", url: `${basePath}/03_租賃訂單車輛_Hims500 (500台)/ORDER-001/取車_01.png` },
          { name: "取車_02.png", url: `${basePath}/03_租賃訂單車輛_Hims500 (500台)/ORDER-001/取車_02.png` },
          { name: "取車_03.png", url: `${basePath}/03_租賃訂單車輛_Hims500 (500台)/ORDER-001/取車_03.png` },
          { name: "取車_04.png", url: `${basePath}/03_租賃訂單車輛_Hims500 (500台)/ORDER-001/取車_04.png` },
          { name: "還車_01.png", url: `${basePath}/03_租賃訂單車輛_Hims500 (500台)/ORDER-001/還車_01.png` },
          { name: "還車_02.png", url: `${basePath}/03_租賃訂單車輛_Hims500 (500台)/ORDER-001/還車_02.png` },
          { name: "還車_03.png", url: `${basePath}/03_租賃訂單車輛_Hims500 (500台)/ORDER-001/還車_03.png` },
          { name: "還車_04.png", url: `${basePath}/03_租賃訂單車輛_Hims500 (500台)/ORDER-001/還車_04.png` },
          { name: "還車_10.png", url: `${basePath}/03_租賃訂單車輛_Hims500 (500台)/ORDER-001/還車_10.png` },
          { name: "還車_11.png", url: `${basePath}/03_租賃訂單車輛_Hims500 (500台)/ORDER-001/還車_11.png` }
        ]
      },
      {
        id: "03_ORDER-002",
        vehicle: "ORDER-002",
        category: "03_租賃訂單車輛_Hims500 (500台)",
        count: 10,
        images: [
          { name: "取車_01.png", url: `${basePath}/03_租賃訂單車輛_Hims500 (500台)/ORDER-002/取車_01.png` },
          { name: "取車_02.png", url: `${basePath}/03_租賃訂單車輛_Hims500 (500台)/ORDER-002/取車_02.png` },
          { name: "取車_03.png", url: `${basePath}/03_租賃訂單車輛_Hims500 (500台)/ORDER-002/取車_03.png` },
          { name: "取車_04.png", url: `${basePath}/03_租賃訂單車輛_Hims500 (500台)/ORDER-002/取車_04.png` },
          { name: "還車_01.png", url: `${basePath}/03_租賃訂單車輛_Hims500 (500台)/ORDER-002/還車_01.png` },
          { name: "還車_02.png", url: `${basePath}/03_租賃訂單車輛_Hims500 (500台)/ORDER-002/還車_02.png` },
          { name: "還車_03.png", url: `${basePath}/03_租賃訂單車輛_Hims500 (500台)/ORDER-002/還車_03.png` },
          { name: "還車_04.png", url: `${basePath}/03_租賃訂單車輛_Hims500 (500台)/ORDER-002/還車_04.png` },
          { name: "還車_10.png", url: `${basePath}/03_租賃訂單車輛_Hims500 (500台)/ORDER-002/還車_10.png` },
          { name: "還車_11.png", url: `${basePath}/03_租賃訂單車輛_Hims500 (500台)/ORDER-002/還車_11.png` }
        ]
      },
      {
        id: "02_RCR-7661",
        vehicle: "RCR-7661",
        category: "02_索賠車損車輛 (28台)",
        count: 3,
        images: [
          { name: "RCR-7661 左後 借.jpg", url: `${basePath}/02_索賠車損車輛 (28台)/RCR-7661/RCR-7661 左後 借.jpg` },
          { name: "RCR-7661 左後 還.jpg", url: `${basePath}/02_索賠車損車輛 (28台)/RCR-7661/RCR-7661 左後 還.jpg` },
          { name: "RCR-7661 左後 還2.jpg", url: `${basePath}/02_索賠車損車輛 (28台)/RCR-7661/RCR-7661 左後 還2.jpg` }
        ]
      },
      {
        id: "01_RCG2235",
        vehicle: "RCG2235",
        category: "01_正常無損車輛 (30台)",
        count: 8,
        images: [
          { name: "RCG2235左前 借.jfif", url: `${basePath}/01_正常無損車輛 (30台)/RCG2235/RCG2235左前 借.jfif` },
          { name: "RCG2235左前 還.jfif", url: `${basePath}/01_正常無損車輛 (30台)/RCG2235/RCG2235左前 還.jfif` },
          { name: "RCG2235右前 借.jfif", url: `${basePath}/01_正常無損車輛 (30台)/RCG2235/RCG2235右前 借.jfif` },
          { name: "RCG2235右前 還.jfif", url: `${basePath}/01_正常無損車輛 (30台)/RCG2235/RCG2235右前 還.jfif` },
          { name: "RCG2235左後 借.jfif", url: `${basePath}/01_正常無損車輛 (30台)/RCG2235/RCG2235左後 借.jfif` },
          { name: "RCG2235左後 還.jfif", url: `${basePath}/01_正常無損車輛 (30台)/RCG2235/RCG2235左後 還.jfif` },
          { name: "RCG2235右後 借.jfif", url: `${basePath}/01_正常無損車輛 (30台)/RCG2235/RCG2235右後 借.jfif` },
          { name: "RCG2235右後 還.jfif", url: `${basePath}/01_正常無損車輛 (30台)/RCG2235/RCG2235右後 還.jfif` }
        ]
      },
      {
        id: "01_RCX5100",
        vehicle: "RCX5100",
        category: "01_正常無損車輛 (30台)",
        count: 8,
        images: [
          { name: "RCX5100左前 借.jfif", url: `${basePath}/01_正常無損車輛 (30台)/RCX5100/RCX5100左前 借.jfif` },
          { name: "RCX5100左前 還.jfif", url: `${basePath}/01_正常無損車輛 (30台)/RCX5100/RCX5100左前 還.jfif` },
          { name: "RCX5100右前 借.jfif", url: `${basePath}/01_正常無損車輛 (30台)/RCX5100/RCX5100右前 借.jfif` },
          { name: "RCX5100右前 還.jfif", url: `${basePath}/01_正常無損車輛 (30台)/RCX5100/RCX5100右前 還.jfif` },
          { name: "RCX5100左後 借.jfif", url: `${basePath}/01_正常無損車輛 (30台)/RCX5100/RCX5100左後 借.jfif` },
          { name: "RCX5100左後 還.jfif", url: `${basePath}/01_正常無損車輛 (30台)/RCX5100/RCX5100左後 還.jfif` },
          { name: "RCX5100右後 借.jfif", url: `${basePath}/01_正常無損車輛 (30台)/RCX5100/RCX5100右後 借.jfif` },
          { name: "RCX5100右後 還.jfif", url: `${basePath}/01_正常無損車輛 (30台)/RCX5100/RCX5100右後 還.jfif` }
        ]
      }
    ];
  }

  // Generates high-tech vector visual SVG data URL for each angle
  createAngleSvgDataUrl(type, plate, hasDamage = false, hasTrash = false) {
    const isInterior = type >= 10;
    const viewTitle = {
      1: "視角 #1 — 左前方 45°",
      2: "視角 #2 — 右前方 45°",
      3: "視角 #3 — 左後方 45°",
      4: "視角 #4 — 右後方 45°",
      10: "視角 #10 — 前座艙 (駕駛艙/中控)",
      11: "視角 #11 — 後座艙 (乘客座椅/地毯)"
    }[type] || `視角 #${type}`;

    let visualElements = "";
    if (isInterior) {
      visualElements = `
        <rect x="60" y="70" width="360" height="180" rx="16" fill="#1e293b" stroke="#334155" stroke-width="2"/>
        <path d="M 80 180 Q 150 140 240 140 Q 330 140 400 180" stroke="#00f0ff" stroke-width="2" fill="none" opacity="0.4"/>
        <circle cx="160" cy="130" r="36" fill="#0f172a" stroke="#475569" stroke-width="2"/>
        <rect x="230" y="110" width="60" height="40" rx="6" fill="#0f172a" stroke="#00f0ff" stroke-width="1.5"/>
        <rect x="235" y="165" width="50" height="70" rx="8" fill="#0f172a" stroke="#334155" stroke-width="1.5"/>
        ${hasTrash ? `
          <g>
            <rect x="245" y="175" width="22" height="32" rx="3" fill="#f59e0b" opacity="0.9"/>
            <line x1="256" y1="170" x2="265" y2="160" stroke="#f59e0b" stroke-width="3"/>
            <rect x="235" y="165" width="42" height="52" fill="none" stroke="#ef4444" stroke-width="2" stroke-dasharray="4"/>
            <text x="282" y="185" fill="#ef4444" font-size="10" font-family="monospace">垃圾遺留</text>
          </g>
        ` : `
          <circle cx="260" cy="190" r="10" fill="#10b981" opacity="0.3"/>
          <text x="240" y="225" fill="#10b981" font-size="9" font-family="monospace">杯架整潔</text>
        `}
      `;
    } else {
      visualElements = `
        <ellipse cx="240" cy="180" rx="170" ry="24" fill="#0c121e" opacity="0.6"/>
        <path d="M 100 170 Q 140 100 230 95 Q 340 95 380 170 Z" fill="#1e293b" stroke="#38bdf8" stroke-width="2"/>
        <path d="M 140 130 Q 180 105 240 105 Q 300 105 340 130 Z" fill="#0f172a" stroke="#00f0ff" stroke-width="1.5" opacity="0.8"/>
        <circle cx="150" cy="170" r="22" fill="#090d16" stroke="#475569" stroke-width="4"/>
        <circle cx="330" cy="170" r="22" fill="#090d16" stroke="#475569" stroke-width="4"/>
        <!-- License Plate -->
        <g transform="translate(205, 155)">
          <rect width="70" height="22" rx="3" fill="#ffffff" stroke="#1e293b" stroke-width="1.5"/>
          <text x="35" y="15" fill="#0f172a" font-size="11" font-weight="bold" font-family="monospace" text-anchor="middle">${plate}</text>
        </g>
        ${hasDamage ? `
          <g>
            <path d="M 330 145 Q 355 135 375 148 Q 365 160 340 155 Z" fill="#ef4444" fill-opacity="0.3" stroke="#ef4444" stroke-width="2.5" stroke-dasharray="3"/>
            <circle cx="370" cy="145" r="4" fill="#ef4444"/>
            <text x="310" y="132" fill="#ef4444" font-size="11" font-weight="bold" font-family="monospace">擦傷 15cm</text>
          </g>
        ` : `
          <text x="240" y="80" fill="#10b981" font-size="10" font-family="monospace" text-anchor="middle">外觀完好無損</text>
        `}
      `;
    }

    const svg = `
      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 480 270" width="100%" height="100%">
        <defs>
          <linearGradient id="bgGrad" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="#070a12"/>
            <stop offset="100%" stop-color="#111827"/>
          </linearGradient>
        </defs>
        <rect width="480" height="270" fill="url(#bgGrad)"/>
        <!-- Grid lines -->
        <line x1="0" y1="90" x2="480" y2="90" stroke="#1e293b" stroke-width="0.7" opacity="0.6"/>
        <line x1="0" y1="180" x2="480" y2="180" stroke="#1e293b" stroke-width="0.7" opacity="0.6"/>
        <line x1="160" y1="0" x2="160" y2="270" stroke="#1e293b" stroke-width="0.7" opacity="0.6"/>
        <line x1="320" y1="0" x2="320" y2="270" stroke="#1e293b" stroke-width="0.7" opacity="0.6"/>
        
        <!-- Header Info -->
        <text x="16" y="28" fill="#38bdf8" font-size="12" font-weight="bold" font-family="sans-serif">${viewTitle}</text>
        <text x="464" y="28" fill="#94a3b8" font-size="11" font-family="monospace" text-anchor="end">${plate}</text>
        
        ${visualElements}
        
        <!-- Bottom Corner HUD Tags -->
        <rect x="16" y="240" width="80" height="18" rx="4" fill="#0f172a" opacity="0.8"/>
        <text x="56" y="253" fill="#64748b" font-size="9" font-family="monospace" text-anchor="middle">1080P HD</text>
        
        <rect x="384" y="240" width="80" height="18" rx="4" fill="#0f172a" opacity="0.8"/>
        <text x="424" y="253" fill="${hasDamage || hasTrash ? '#ef4444' : '#10b981'}" font-size="9" font-family="monospace" text-anchor="middle">${hasDamage ? '外觀異常' : (hasTrash ? '座艙異常' : '合規標記')}</text>
      </svg>
    `;

    return "data:image/svg+xml;charset=utf-8," + encodeURIComponent(svg.trim());
  }

  // ==========================================
  // Management Endpoints
  // ==========================================
  async getVehicles() {
    if (this.isBackendOnline) {
      try {
        const res = await fetch(`${this.apiBase}/api/v1/vehicles`);
        if (res.ok) return await res.json();
      } catch (err) { }
    }
    return {
      data: [
        { vehicle_code: "RCR-7661", plate_number: "RCR-7661", model: "Toyota Yaris Cross", status: "available", total_trips: 142 },
        { vehicle_code: "RCX-5100", plate_number: "RCX-5100", model: "Toyota Corolla Cross", status: "rented", total_trips: 98 },
        { vehicle_code: "BAP-1234", plate_number: "BAP-1234", model: "Toyota Prius C", status: "cleaning", total_trips: 215 },
        { vehicle_code: "RDF-8822", plate_number: "RDF-8822", model: "Nissan Kicks", status: "repairing", total_trips: 84 },
        { vehicle_code: "RDE-3311", plate_number: "RDE-3311", model: "Honda Fit e:HEV", status: "available", total_trips: 63 }
      ]
    };
  }

  async getWorkOrders() {
    if (this.isBackendOnline) {
      try {
        const res = await fetch(`${this.apiBase}/api/v1/work-orders`);
        if (res.ok) return await res.json();
      } catch (err) { }
    }
    return {
      data: [
        {
          work_order_id: "WO-992140",
          case_id: "CASE-01",
          vehicle_code: "RCX-5100",
          category: "repair",
          priority: "urgent",
          status: "open",
          ai_repair_guide: "右前保險桿擦傷 (深度 > 2mm)，建議局部烤漆處理。",
          estimated_labor_hours: 2.0,
          created_at: new Date(Date.now() - 3600000).toISOString()
        },
        {
          work_order_id: "WO-992138",
          case_id: "CASE-02",
          vehicle_code: "BAP-1234",
          category: "cleaning",
          priority: "high",
          status: "in_progress",
          ai_repair_guide: "後座乘客腳踏墊泥沙覆蓋，執行深層吸塵清潔。",
          estimated_labor_hours: 0.5,
          created_at: new Date(Date.now() - 7200000).toISOString()
        }
      ]
    };
  }
}

// Global Singleton
window.iguardAPI = new IGuardAPI();
