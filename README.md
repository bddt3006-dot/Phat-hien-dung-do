# Hệ Thống Giám Sát Dừng Đỗ Xe - Parking Violation Detection System

## Cấu trúc thư mục

```
Module dừng đỗ xe/
├── Data UA Detrac/              # Dữ liệu gốc (không chỉnh sửa)
├── data/                        # Dữ liệu đã convert sang YOLO format
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── val/
│   │   ├── images/
│   │   └── labels/
│   └── test/
│       ├── images/
│       └── labels/
├── scripts/                     # Scripts tiền xử lý dữ liệu
├── src/                         # Source code chính
│   ├── detection/               # YOLO inference
│   ├── tracking/                # ByteTrack integration
│   ├── rule_engine/             # Logic dừng đỗ
│   └── dashboard/               # Web UI
├── models/                      # Trọng số mô hình (best.pt)
├── configs/                     # File cấu hình
├── outputs/                     # Kết quả: ảnh vi phạm, logs
├── data.yaml                    # Config cho YOLO training
└── requirements.txt
```

## Cài đặt

```bash
pip install -r requirements.txt
```

## Sử dụng

### 1. Convert dữ liệu
```bash
python scripts/convert_detrac_to_yolo.py
```

### 2. Train YOLO
```bash
yolo detect train data=data.yaml model=yolo11m.pt epochs=50 imgsz=960
```

### 3. Chạy hệ thống bằng CLI
```powershell
# Chạy với mô hình đã train (output_runs/best.pt):
& "C:\Users\ADMIN\anaconda3\envs\ai_env3\python.exe" src/main.py

# ĐƯỜNG LUI AN TOÀN: Chạy ngay mô hình gốc Baseline (yolo11m.pt):
& "C:\Users\ADMIN\anaconda3\envs\ai_env3\python.exe" src/main.py --baseline
```

### 4. Mở Dashboard Trực Quan, Bộ Chọn Model & Tool Vẽ ROI

**Cách 1: Chạy file launcher (Khuyên dùng - 1 click):**
- Click đúp vào file `run_dashboard.bat` trong thư mục dự án.

**Cách 2: Chạy trực tiếp từ PowerShell:**
```powershell
& "C:\Users\ADMIN\anaconda3\envs\ai_env3\python.exe" -m streamlit run src/dashboard.py
```
*(Hoặc kích hoạt môi trường conda trước: `conda activate ai_env3` rồi gõ `streamlit run src/dashboard.py`)*

> **Đường lui 1-Click trên Web:** Trên Sidebar Dashboard có mục **"Mô Hình Nhận Diện (Weights)"** với nút **"↩️ Khôi Phục Gốc"**. Bấm nút này sẽ lập tức hoàn tác về `yolo11m.pt` trong 1 giây!

### 5. So Sánh Hiệu Năng Mô Hình (Benchmark & Side-by-Side)
```powershell
& "C:\Users\ADMIN\anaconda3\envs\ai_env3\python.exe" scripts/compare_models.py
```
Lệnh trên sẽ so sánh trực quan định lượng giữa `output_runs/best.pt` và `yolo11m.pt` (FPS, độ tin cậy, số lượng xe) và xuất ảnh so sánh song song vào `evidence/model_comparison.jpg`.

