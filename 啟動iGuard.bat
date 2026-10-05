@echo off
chcp 65001 >nul
title iGuard 智慧車況檢驗系統 [RTX 5090 + Cloudflare Tunnel]

echo ===============================================================================
echo   🚗 iGuard — 智慧車況檢驗與營運分流守護者
echo   ⚡ AI 核心算力 : 本地 NVIDIA GeForce RTX 5090 (32GB VRAM)
echo   🌐 跨網穿透   : Cloudflare Tunnel (HTTPS)
echo ===============================================================================
echo.

cd /d "%~dp0"

echo [1/2] 正在背景啟動 FastAPI AI 服務...
start /b ..\iRent_env\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000

echo 等待 AI 模型預熱與顯存配置 (4 秒)...
timeout /t 4 /nobreak >nul

echo.
echo [2/2] 正在建立 Cloudflare 對外 HTTPS 穿透通道...
"C:\Program Files (x86)\cloudflared\cloudflared.exe" tunnel --url http://127.0.0.1:8000

pause
