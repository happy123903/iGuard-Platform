const DEFAULT_SUPABASE_URL = "https://uzpmwyeirkgdweuuptbr.supabase.co";
const DEFAULT_SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InV6cG13eWVpcmtnZHdldXVwdGJyIiwicm9sZSI6ImFub24iLCJpYXQiOjE3OTA3NTkyODgsImV4cCI6MjEwNjMzNTI4OH0.VQbB3jgYyAslq2bP1YNPYM5BDj2mlGNHvUQ8zf5Cv0g";

class IGuardSupabase {
  constructor() {
    this.url = localStorage.getItem("iguard_supabase_url") || DEFAULT_SUPABASE_URL;
    this.anonKey = localStorage.getItem("iguard_supabase_anon_key") || DEFAULT_SUPABASE_ANON_KEY;
    this.client = null;
    this.realtimeChannels = [];
    this.init();

    // If not configured in localStorage, dynamically fetch public config from backend
    if (!this.url || !this.anonKey) {
      this.fetchPublicConfig();
    }
  }

  async fetchPublicConfig() {
    try {
      const apiBase = localStorage.getItem("iguard_api_base") || "http://localhost:8000";
      const res = await fetch(`${apiBase}/api/v1/config/public`);
      if (res.ok) {
        const data = await res.json();
        if (data.supabase_url && data.supabase_anon_key) {
          this.setCredentials(data.supabase_url, data.supabase_anon_key);
        }
      }
    } catch (err) {
      // Backend offline or in simulation mode
    }
  }

  init() {
    if (window.supabase && this.url && this.anonKey) {
      try {
        this.client = window.supabase.createClient(this.url, this.anonKey);
        console.log("[iGuard Supabase] Connected to:", this.url);
      } catch (err) {
        console.warn("[iGuard Supabase] Initialization error:", err);
      }
    }
  }

  isReady() {
    return Boolean(this.client);
  }

  setCredentials(url, key) {
    this.url = (url || "").trim().replace(/\/+$/, "").replace(/\/rest\/v1\/?$/, "");
    this.anonKey = (key || "").trim();
    if (this.url) localStorage.setItem("iguard_supabase_url", this.url);
    if (this.anonKey) localStorage.setItem("iguard_supabase_anon_key", this.anonKey);
    this.init();
  }

  subscribeToInspections(onInsert, onUpdate) {
    if (!this.client) return null;
    const channel = this.client
      .channel("realtime-inspections")
      .on(
        "postgres_changes",
        { event: "INSERT", schema: "public", table: "inspections" },
        (payload) => onInsert && onInsert(payload.new)
      )
      .on(
        "postgres_changes",
        { event: "UPDATE", schema: "public", table: "inspections" },
        (payload) => onUpdate && onUpdate(payload.new)
      );

    this.realtimeChannels.push(channel);
    return channel;
  }

  subscribeToWorkOrders(onInsert, onUpdate) {
    if (!this.client) return null;
    const channel = this.client
      .channel("realtime-work-orders")
      .on(
        "postgres_changes",
        { event: "INSERT", schema: "public", table: "work_orders" },
        (payload) => onInsert && onInsert(payload.new)
      )
      .on(
        "postgres_changes",
        { event: "UPDATE", schema: "public", table: "work_orders" },
        (payload) => onUpdate && onUpdate(payload.new)
      );

    this.realtimeChannels.push(channel);
    return channel;
  }

  async fetchRecentInspections(limit = 30) {
    if (!this.client) return [];
    try {
      const { data, error } = await this.client
        .table("inspections")
        .select("*")
        .order("created_at", { ascending: false })
        .limit(limit);
      if (error) throw error;
      return data || [];
    } catch (err) {
      console.warn("Fetch inspections error:", err);
      return [];
    }
  }

  async fetchRecentWorkOrders(limit = 30) {
    if (!this.client) return [];
    try {
      const { data, error } = await this.client
        .table("work_orders")
        .select("*")
        .order("created_at", { ascending: false })
        .limit(limit);
      if (error) throw error;
      return data || [];
    } catch (err) {
      console.warn("Fetch work orders error:", err);
      return [];
    }
  }

  async fetchVehicles(limit = 50) {
    if (!this.client) return [];
    try {
      const { data, error } = await this.client
        .table("vehicles")
        .select("*")
        .order("updated_at", { ascending: false })
        .limit(limit);
      if (error) throw error;
      return data || [];
    } catch (err) {
      console.warn("Fetch vehicles error:", err);
      return [];
    }
  }

  async updateWorkOrderStatus(workOrderId, status) {
    if (!this.client) return null;
    try {
      const { data, error } = await this.client
        .table("work_orders")
        .update({ status: status, updated_at: new Date().toISOString() })
        .eq("work_order_id", workOrderId)
        .select();
      if (error) throw error;
      return data && data[0];
    } catch (err) {
      console.error("Update work order status failed:", err);
      throw err;
    }
  }

  async updateVehicleStatus(vehicleCode, status) {
    if (!this.client) return null;
    try {
      const { data, error } = await this.client
        .table("vehicles")
        .update({ status: status, updated_at: new Date().toISOString() })
        .eq("vehicle_code", vehicleCode)
        .select();
      if (error) throw error;
      return data && data[0];
    } catch (err) {
      console.error("Update vehicle status failed:", err);
      throw err;
    }
  }

  async uploadImage(blobOrFile, fileName) {
    if (!this.client) return null;
    try {
      const bucket = "inspection-images";
      const filePath = `uploads/${Date.now()}_${fileName}`;
      const { data, error } = await this.client.storage
        .from(bucket)
        .upload(filePath, blobOrFile, {
          contentType: "image/jpeg",
          upsert: true
        });

      if (error) throw error;

      const { data: urlData } = this.client.storage
        .from(bucket)
        .getPublicUrl(filePath);

      return urlData.publicUrl;
    } catch (err) {
      console.error("Supabase Storage upload error:", err);
      return null;
    }
  }
}

// Global Singleton
window.iguardSupabase = new IGuardSupabase();
