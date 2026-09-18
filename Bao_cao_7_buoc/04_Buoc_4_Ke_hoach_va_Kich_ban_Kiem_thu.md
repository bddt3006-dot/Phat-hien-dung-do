# BƯỚC 4: KẾ HOẠCH VÀ KỊCH BẢN KIỂM THỬ PHẦN MỀM
**Hệ thống:** Giám sát và Tự động Phát hiện Dừng Đỗ Xe Trái Phép (SIPDS)  
**Tài liệu:** Test Plan & Test Scenarios | **Phiên bản:** 1.0  

---

## I. MỤC TIÊU & MÔI TRƯỜNG KIỂM THỬ
Kiểm tra tính đúng đắn của logic nghiệp vụ, độ ổn định của pipeline và hiệu năng mô hình trên:
- **Phần cứng:** CPU i7-12700H, RAM 16GB, GPU NVIDIA RTX 3050 Laptop (4GB VRAM).
- **Phần mềm:** Python 3.13, PyTorch 2.6.0+cu124, CUDA 12.4, OpenCV 4.10, Ultralytics 8.3.

## II. DANH SÁCH 10 KỊCH BẢN KIỂM THỬ CHI TIẾT
1. **TC-01:** Xe chạy qua ROI không dừng -> BBox Xanh, không cảnh báo (High).
2. **TC-02:** Xe dừng ngắn hạn (< 10s) -> Đếm giây 1-7s, không báo vi phạm (High).
3. **TC-03:** Xe đỗ quá thời gian (>= 30s) -> 10s chuyển Vàng ('STOPPING'), 30s chuyển ĐỎ ('VIOLATION') & xuất bằng chứng (Critical).
4. **TC-04:** Che khuất tạm thời (< 5s) -> Spatial Re-ID giữ nguyên ID, không reset timer (Critical).
5. **TC-05:** Ùn tắc giao thông nhích chậm -> Bộ lọc dịch chuyển không báo sai (High).
6. **TC-06:** Tạo ROI đa giác qua GUI `select_roi.py` -> Lưu và hiển thị đúng vào `settings.yaml` (Medium).
7. **TC-07:** Nạp luồng camera RTSP thời gian thực -> Kết nối ổn định, FPS >= 30 (High).
8. **TC-08:** Chuyển đổi mô hình YOLO11n vs YOLO11m qua cấu hình -> Tự động nạp đúng trọng số (High).
9. **TC-09:** Kiểm thử tải chạy liên tục 2 giờ -> Không rò rỉ RAM/VRAM, không crash (Medium).
10. **TC-10:** Xử lý ngoại lệ file video/IP sai -> Báo lỗi thân thiện, thoát an toàn (Medium).
