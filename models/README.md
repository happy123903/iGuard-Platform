# iGuard AI 模型權重目錄 (Model Weights Directory)

此目錄存放專案所需之 AI 深度學習模型權重。
依據 `.gitignore` 規範，大型二進位權重檔（*.pt, *.pth, *.bin, *.safetensors 等）不會上傳至 GitHub。

## 權重放置清單

請將各模型權重放置於本目錄下：

| 模型名稱 | 檔案 / 資料夾名稱 | 格式 | 說明 |
|---|---|---|---|
| **YOLOv11x-seg** | `yolo11x-seg.pt` | PyTorch (.pt, ~125MB) | 車身視角辨識與各部件（車門、保桿、玻璃、輪胎）實例分割 |
| **SAM 2 Large** | `sam2_hiera_large.pt` | PyTorch (.pt, ~898MB) | Meta SAM 2 像素級精準車損遮罩提取 |
| **Zero-DCE** | `zero_dce.pth` | PyTorch (.pth, ~320KB) | 地下室/夜間暗光影像零參考曲線增強 |
| **Qwen2.5-VL** | `Qwen2.5-VL-3B-Instruct/` | Hugging Face / ModelScope 資料夾 (~7.16GB) | 車內整潔度多模態辨識、垃圾檢測與失物招領推理 |

## 快速下載指引

### 1. YOLOv11x-seg
系統在首次啟動時，若未檢測到 `yolo11x-seg.pt`，將自動透過 Ultralytics 下載官方預訓練權重。

### 2. SAM 2 Large
可透過 Meta 官方倉庫下載：
```bash
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt -P models/
```

### 3. Zero-DCE
下載官方 Epoch99.pth 權重並命名為 `zero_dce.pth` 置於 `models/` 目錄。

### 4. Qwen2.5-VL-3B-Instruct
可透過 Hugging Face 或 ModelScope 下載：
```python
from modelscope import snapshot_download
model_dir = snapshot_download('Qwen/Qwen2.5-VL-3B-Instruct', local_dir='models/Qwen2.5-VL-3B-Instruct')
```
