# Hệ Thống Giám Sát Dừng Đỗ Xe - Parking Violation Detection System

## Cấu trúc thư mục

```
Module dừng đỗ xe/
├── data/                        # Video và dữ liệu mẫu (sample.mp4)
├── config/                      # File cấu hình hệ thống & ByteTrack
│   ├── settings.yaml
│   └── bytetrack_custom.yaml
├── scripts/                     # Scripts hỗ trợ và benchmark
│   ├── compare_models.py        # So sánh hiệu năng mô hình Custom vs Baseline
│   ├── convert_detrac_to_yolo.py
│   ├── make_sample_video.py
│   ├── test_occlusion.py
│   └── test_pipeline.py
├── src/                         # Source code chính
│   ├── main.py                  # Chạy CLI nhận diện video
│   ├── dashboard.py             # Web UI Dashboard Streamlit
│   ├── model_utils.py           # Quản lý weights & Smart Class Resolver
│   └── rule_engine.py           # Logic phát hiện dừng đỗ vi phạm
├── output_runs/                 # Checkpoint và kết quả huấn luyện (best.pt)
├── models/                      # Trọng số dự phòng
├── evidence/                    # Bằng chứng vi phạm (ảnh + CSV)
├── run_dashboard.bat            # Khởi động Web Dashboard (1-click)
├── run_main.bat                 # Khởi động CLI (1-click)
├── data.yaml                    # Config cho YOLO training
└── requirements.txt
```

## Cài đặt

```bash
pip install -r requirements.txt
```

## Hướng dẫn sử dụng

### 1. Chạy hệ thống bằng CLI

**Cách 1: Dùng file launcher:**
- Click đúp `run_main.bat`

**Cách 2: Chạy lệnh dòng lệnh:**
```powershell
# Chạy với mô hình đã train (output_runs/best.pt):
python src/main.py

# ĐƯỜNG LUI AN TOÀN: Chạy ngay mô hình gốc Baseline (yolo11m.pt):
python src/main.py --baseline

# Tùy chọn video khác và chỉ định thiết bị:
python src/main.py --video data/sample.mp4 --device cpu
```

### 2. Mở Dashboard Trực Quan, Bộ Chọn Model & Tool Vẽ ROI

**Cách 1: Chạy file launcher (Khuyên dùng - 1 click):**
- Click đúp vào file `run_dashboard.bat` trong thư mục dự án.

**Cách 2: Chạy trực tiếp từ terminal:**
```powershell
python -m streamlit run src/dashboard.py
```
*(Nếu dùng môi trường ảo: `.\.venv\Scripts\python.exe -m streamlit run src/dashboard.py`)*

> **Đường lui 1-Click trên Web:** Trên Sidebar Dashboard có mục **"Mô Hình Nhận Diện (Weights)"** với nút **"↩️ Khôi Phục Gốc"**. Bấm nút này sẽ lập tức hoàn tác về `yolo11m.pt` trong 1 giây!

### 3. So Sánh Hiệu Năng Mô Hình (Benchmark & Side-by-Side)

```powershell
python scripts/compare_models.py
```
Lệnh trên sẽ so sánh trực quan định lượng giữa `output_runs/best.pt` và `yolo11m.pt` (FPS, độ trễ ms, độ tin cậy, số lượng xe) và xuất ảnh so sánh song song vào `evidence/model_comparison.jpg`.

### 4. Huấn Luyện Chuẩn Hóa YOLO11n (Kaggle & Colab)

Hệ thống cung cấp sẵn 2 notebook và script huấn luyện đã được tinh chỉnh chống Overfitting và chống mất nhận diện:
- **Kaggle**: Tải file [`train_kaggle.ipynb`](train_kaggle.ipynb) lên Kaggle (Bật GPU T4 x 2 hoặc P100, bật **Internet: On**).
- **Google Colab**: Tải file [`train_colab.ipynb`](train_colab.ipynb) lên Google Colab (chọn GPU T4).
- **Script dòng lệnh**: [`scripts/train_yolo11n.py`](scripts/train_yolo11n.py)

> **Cơ chế tối ưu:** Khóa 10 tầng Backbone (`freeze=10`), điều chỉnh tốc độ học `lr0=0.001`, Cosine Annealing `cos_lr=True`, và tự động ngắt `patience=7` khi phát hiện overfitting.
