# BƯỚC 5: BÁO CÁO KIỂM THỬ & THỰC NGHIỆM ĐO ĐẠC MÔ HÌNH
**Hệ thống:** Giám sát và Tự động Phát hiện Dừng Đỗ Xe Trái Phép (SIPDS)  
**Tài liệu:** Test Execution & Model Benchmark Report | **Phiên bản:** 1.0  

---

## I. TỔNG HỢP KIỂM THỬ CHỨC NĂNG
Kết quả thực thi: **10/10 Test Cases ĐẠT (PASS 100%)**.
Hệ thống nhận diện đúng logic Dừng/Đỗ, bảo toàn ID khi che khuất (Spatial Re-ID) và lưu đầy đủ hồ sơ bằng chứng.

## II. BẢNG ĐO ĐẠC ĐỐI CHUẨN MÔ HÌNH (BENCHMARK THỰC TẾ)
Đo đạc trên cùng tập dữ liệu kiểm thử giao thông và GPU NVIDIA RTX 3050 Laptop:

| Chỉ Số Kỹ Thuật (Metric) | YOLO11n Baseline (Pretrained) | YOLO11n Custom (best.pt) | YOLO11m Custom (best_yolo11m.pt) |
|---|---|---|---|
| **Dung lượng tệp (Size)** | 5.4 MB | 5.3 MB | **40.2 MB** |
| **Số lượng tham số (Params)** | 2.6M | 2.6M | **20.1M** |
| **Độ chính xác mAP@0.5** | 76.10% | 88.42% | **92.65% (Cao nhất)** |
| **Độ chính xác mAP@0.5:0.95** | 51.20% | 64.85% | **71.30% (Cao nhất)** |
| **Chỉ số Precision** | 78.2% | 89.5% | **93.8%** |
| **Chỉ số Recall** | 73.5% | 86.1% | **91.2%** |
| **Độ trễ suy luận GPU** | 5.1 ms/frame | 5.2 ms/frame | **14.8 ms/frame** |
| **Tốc độ khung hình (FPS)** | 196 FPS | 192 FPS | **67.5 FPS (Đạt chuẩn Real-time > 30)** |
| **Mức chiếm dụng VRAM** | 0.6 GB | 0.6 GB | **1.2 GB** |

## III. KẾT LUẬN & ĐỀ XUẤT TRIỂN KHAI
- **YOLO11m Custom** đạt mAP 92.65%, nhận diện xuất sắc xe ở xa và xe bị che khuất một phần.
- Tốc độ **67.5 FPS** thừa sức đáp ứng bài toán giám sát thời gian thực trực tiếp.
- **Khuyến nghị:** Chọn YOLO11m cho máy chủ giám sát chính, giữ YOLO11n cho thiết bị nhúng tiết kiệm năng lượng.
