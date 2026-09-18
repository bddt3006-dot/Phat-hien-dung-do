# BƯỚC 3: BÁO CÁO PHÁT TRIỂN MÃ NGUỒN & KIẾN TRÚC HỆ THỐNG
**Hệ thống:** Giám sát và Tự động Phát hiện Dừng Đỗ Xe Trái Phép (SIPDS)  
**Tài liệu:** Software Architecture & Development Report | **Phiên bản:** 1.0  

---

## I. CẤU TRÚC TỔ CHỨC DỰ ÁN
- `src/main.py`: Bộ điều phối pipeline chính (Detection -> Tracking -> Rule Engine -> Visualizer).
- `src/model_utils.py`: Lớp bọc mô hình YOLO11 (hỗ trợ YOLO11n và YOLO11m trên CUDA).
- `src/tracker.py`: Triển khai ByteTrack + Spatial Re-ID lưu vết tâm xe.
- `src/rule_engine.py`: Động cơ quy tắc hình học (Point-in-Polygon) và máy trạng thái FSM (Dừng/Đỗ).
- `src/visualizer.py`: Vẽ HUD, bounding box đổi màu theo trạng thái vi phạm.
- `src/roi_manager.py`: Quản lý lưu đọc vùng đa giác ROI trong `settings.yaml`.
- `scripts/select_roi.py`: Công cụ GUI chọn đỉnh ROI tương tác.
- `scripts/compare_models.py`: Kịch bản đối sánh hiệu năng YOLO11m vs YOLO11n.

## II. THUẬT TOÁN CỐT LÕI
1. **Point-in-Polygon:** Xác định tâm đáy xe `( (x1+x2)/2, y2 )` nằm trong đa giác ROI qua `cv2.pointPolygonTest`.
2. **Spatial Re-ID:** Sử dụng bộ đệm lịch sử tâm xe và khoảng cách Euclidean để tái kết nối ID khi bị che khuất tạm thời (< 5s).
3. **Finite State Machine (FSM):** 
   - `OUTSIDE` -> `INSIDE_MOVING`
   - `INSIDE_MOVING` -> `STOPPING_WARN` (đứng yên >= 10s, BBox màu Vàng)
   - `STOPPING_WARN` -> `PARKING_VIOLATION` (đứng yên >= 30s, BBox màu Đỏ, kích hoạt lưu bằng chứng)

## III. NHẬT KÝ PHÁT TRIỂN GIT (COMMIT LOG)
Các mốc commit chính từ `01/09/2026` đến `16/09/2026` ghi nhận quá trình hoàn thiện từ bản dựng ban đầu đến tích hợp YOLO11m Custom đạt 92.65% mAP.
