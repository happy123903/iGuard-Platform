using System;
using System.IO;
using System.Diagnostics;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;

namespace IGuardLauncher
{
    class Program
    {
        static Process uvicornProc = null;
        static Process cfProc = null;
        static bool isExiting = false;

        static void Main(string[] args)
        {
            try
            {
                Console.OutputEncoding = Encoding.UTF8;
            }
            catch {}
            Console.Title = "iGuard 智慧車況檢驗系統 — AI 算力與穿透監控中心";

            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine(@"
===============================================================================
   🚗 iGuard — 智慧車況檢驗與營運分流守護者
   ⚡ AI 核心算力 : 本地 NVIDIA GeForce RTX 5090 (32GB VRAM)
   🌐 跨網穿透   : Cloudflare Tunnel (HTTPS)
   ☁️ 雲端服務   : Supabase (PostgreSQL + Storage + Realtime)
===============================================================================
");
            Console.ResetColor();

            // 1. 定位專案根目錄
            string baseDir = AppDomain.CurrentDomain.BaseDirectory.TrimEnd('\\');
            if (!File.Exists(Path.Combine(baseDir, "app", "main.py")))
            {
                if (File.Exists(Path.Combine(Directory.GetCurrentDirectory(), "app", "main.py")))
                {
                    baseDir = Directory.GetCurrentDirectory();
                }
                else
                {
                    baseDir = @"c:\Users\user\Desktop\炫晟\iRent\iGuard";
                }
            }

            Console.WriteLine("[系統] 專案根目錄: " + baseDir);

            // 2. 定位 Python 虛擬環境
            string pythonExe = Path.Combine(baseDir, "..", "iRent_env", "Scripts", "python.exe");
            if (!File.Exists(pythonExe))
            {
                pythonExe = Path.Combine(baseDir, "iRent_env", "Scripts", "python.exe");
            }
            if (!File.Exists(pythonExe))
            {
                pythonExe = @"c:\Users\user\Desktop\炫晟\iRent\iRent_env\Scripts\python.exe";
            }
            if (!File.Exists(pythonExe))
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine("[錯誤] 找不到 Python 虛擬環境: " + pythonExe);
                Console.ResetColor();
                Console.WriteLine("請按任意鍵退出...");
                Console.ReadKey();
                return;
            }
            Console.WriteLine("[系統] Python 核心: " + pythonExe);

            // 3. 定位 cloudflared
            string cfExe = @"C:\Program Files (x86)\cloudflared\cloudflared.exe";
            if (!File.Exists(cfExe))
            {
                cfExe = @"C:\Program Files\cloudflared\cloudflared.exe";
            }
            if (!File.Exists(cfExe))
            {
                cfExe = "cloudflared.exe";
            }
            Console.WriteLine("[系統] Cloudflare 核心: " + cfExe);

            // 註冊關閉事件處理
            Console.CancelKeyPress += delegate(object sender, ConsoleCancelEventArgs e) {
                e.Cancel = true;
                ShutdownProcesses();
                Environment.Exit(0);
            };
            AppDomain.CurrentDomain.ProcessExit += delegate(object sender, EventArgs e) {
                ShutdownProcesses();
            };

            // 4. 啟動 FastAPI 服務
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine("\n[步驟 1/2] 正在啟動 FastAPI AI 服務 (監聽 127.0.0.1:8000)...");
            Console.ResetColor();

            ProcessStartInfo uvicornInfo = new ProcessStartInfo();
            uvicornInfo.FileName = pythonExe;
            uvicornInfo.Arguments = "-m uvicorn app.main:app --host 0.0.0.0 --port 8000";
            uvicornInfo.WorkingDirectory = baseDir;
            uvicornInfo.UseShellExecute = false;
            uvicornInfo.RedirectStandardOutput = true;
            uvicornInfo.RedirectStandardError = true;
            uvicornInfo.StandardOutputEncoding = Encoding.UTF8;
            uvicornInfo.StandardErrorEncoding = Encoding.UTF8;

            try
            {
                uvicornProc = new Process();
                uvicornProc.StartInfo = uvicornInfo;
                uvicornProc.OutputDataReceived += delegate(object sender, DataReceivedEventArgs e) {
                    if (!string.IsNullOrEmpty(e.Data))
                    {
                        Console.WriteLine("[FastAPI] " + e.Data);
                    }
                };
                uvicornProc.ErrorDataReceived += delegate(object sender, DataReceivedEventArgs e) {
                    if (!string.IsNullOrEmpty(e.Data))
                    {
                        Console.WriteLine("[FastAPI] " + e.Data);
                    }
                };
                uvicornProc.Start();
                uvicornProc.BeginOutputReadLine();
                uvicornProc.BeginErrorReadLine();
            }
            catch (Exception ex)
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine("[錯誤] 無法啟動 FastAPI 服務: " + ex.Message);
                Console.ResetColor();
                Console.WriteLine("請按任意鍵退出...");
                Console.ReadKey();
                return;
            }

            Console.WriteLine("[系統] 等待 AI 預熱與 GPU 就緒 (約 4 秒)...");
            Thread.Sleep(4000);

            // 5. 啟動 Cloudflare Quick Tunnel
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine("\n[步驟 2/2] 正在建立 Cloudflare 對外 HTTPS 穿透通道 (-> 127.0.0.1:8000)...");
            Console.ResetColor();

            ProcessStartInfo cfInfo = new ProcessStartInfo();
            cfInfo.FileName = cfExe;
            cfInfo.Arguments = "tunnel --url http://127.0.0.1:8000";
            cfInfo.WorkingDirectory = baseDir;
            cfInfo.UseShellExecute = false;
            cfInfo.RedirectStandardOutput = true;
            cfInfo.RedirectStandardError = true;
            cfInfo.StandardOutputEncoding = Encoding.UTF8;
            cfInfo.StandardErrorEncoding = Encoding.UTF8;

            bool urlCaptured = false;
            string apiJsPath = Path.Combine(baseDir, "frontend", "js", "api.js");

            Action<string> parseLine = delegate(string line) {
                if (string.IsNullOrEmpty(line)) return;

                if (!urlCaptured && line.Contains(".trycloudflare.com"))
                {
                    Match m = Regex.Match(line, @"https://[a-zA-Z0-9-]+\.trycloudflare\.com");
                    if (m.Success)
                    {
                        urlCaptured = true;
                        string tunnelUrl = m.Value;
                        string webUrl = "https://happy123903.github.io/iGuard-Platform/?api=" + tunnelUrl;

                        Console.WriteLine("\n");
                        Console.ForegroundColor = ConsoleColor.Green;
                        Console.WriteLine("===============================================================================");
                        Console.WriteLine("  🚀【Cloudflare 公網穿透通道已成功建立！】");
                        Console.WriteLine("  👉 您的 API 公網網址: " + tunnelUrl);
                        Console.WriteLine("  👉 您的 GitHub Pages 專屬直連網址: ");
                        Console.WriteLine("     " + webUrl);
                        Console.WriteLine("===============================================================================");
                        Console.ResetColor();

                        try
                        {
                            Process.Start(webUrl);
                            Console.ForegroundColor = ConsoleColor.Cyan;
                            Console.WriteLine("  🌐 [自動啟動] 已自動為您在瀏覽器開啟前端頁面 (自帶 5090 連線金鑰)！");
                            Console.ResetColor();
                        }
                        catch {}

                        // 自動更新 frontend/js/api.js 中的 DEFAULT_API_BASE
                        try
                        {
                            if (File.Exists(apiJsPath))
                            {
                                string content = File.ReadAllText(apiJsPath, Encoding.UTF8);
                                string updated = Regex.Replace(
                                    content,
                                    @"const DEFAULT_API_BASE = "".*?"";",
                                    "const DEFAULT_API_BASE = \"" + tunnelUrl + "\";"
                                );
                                File.WriteAllText(apiJsPath, updated, Encoding.UTF8);
                                Console.ForegroundColor = ConsoleColor.Magenta;
                                Console.WriteLine("  ✨ [自動同步] 已將前端 frontend/js/api.js 的 API 網址自動更新為此網址！");
                                Console.ResetColor();
                            }
                        }
                        catch (Exception updateEx)
                        {
                            Console.WriteLine("  [提示] 自動寫入 api.js 略過: " + updateEx.Message);
                        }

                        Console.WriteLine("\n[系統] 服務已全部就緒！此視窗請保持開啟以持續監控 AI 請求與顯存日誌。");
                        Console.WriteLine("[系統] 按 Ctrl + C 或直接關閉此視窗即可停止所有服務。\n");
                    }
                }

                if (line.Contains("INF") || line.Contains("ERR") || line.Contains("WRN"))
                {
                    Console.WriteLine("[Cloudflare] " + line);
                }
            };

            try
            {
                cfProc = new Process();
                cfProc.StartInfo = cfInfo;
                cfProc.OutputDataReceived += delegate(object sender, DataReceivedEventArgs e) {
                    parseLine(e.Data);
                };
                cfProc.ErrorDataReceived += delegate(object sender, DataReceivedEventArgs e) {
                    parseLine(e.Data);
                };
                cfProc.Start();
                cfProc.BeginOutputReadLine();
                cfProc.BeginErrorReadLine();
            }
            catch (Exception ex)
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine("[錯誤] 無法啟動 Cloudflared: " + ex.Message);
                Console.ResetColor();
            }

            // 保持主線程運行
            while (!isExiting)
            {
                Thread.Sleep(500);
                if (uvicornProc != null && uvicornProc.HasExited)
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.WriteLine("\n[警告] FastAPI 服務已終止！");
                    Console.ResetColor();
                    break;
                }
            }

            ShutdownProcesses();
        }

        static void ShutdownProcesses()
        {
            if (isExiting) return;
            isExiting = true;
            Console.WriteLine("\n[系統] 正在關閉 iGuard 服務與穿透通道...");
            try
            {
                if (cfProc != null && !cfProc.HasExited)
                {
                    cfProc.Kill();
                }
            }
            catch {}
            try
            {
                if (uvicornProc != null && !uvicornProc.HasExited)
                {
                    uvicornProc.Kill();
                }
            }
            catch {}
            Console.WriteLine("[系統] 服務已完全停止。");
        }
    }
}
