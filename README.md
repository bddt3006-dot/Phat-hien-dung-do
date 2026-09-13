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
& "C:\Users\ADMIN\anaconda3\envs\ai_env3\python.exe" src/main.py
```

### 4. Mở Dashboard Trực Quan & Tool Vẽ ROI

**Cách 1: Chạy file launcher (Khuyên dùng - 1 click):**
- Click đúp vào file `run_dashboard.bat` trong thư mục dự án.

**Cách 2: Chạy trực tiếp từ PowerShell:**
```powershell
& "C:\Users\ADMIN\anaconda3\envs\ai_env3\python.exe" -m streamlit run src/dashboard.py
```
*(Hoặc kích hoạt môi trường conda trước: `conda activate ai_env3` rồi gõ `streamlit run src/dashboard.py`)*
