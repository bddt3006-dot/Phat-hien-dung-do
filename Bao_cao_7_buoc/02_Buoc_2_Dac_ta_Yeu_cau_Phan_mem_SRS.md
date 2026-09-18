# BƯỚC 2: TÀI LIỆU ĐẶC TẢ YÊU CẦU PHẦN MỀM (SRS)
**Hệ thống:** Giám sát và Tự động Phát hiện Dừng Đỗ Xe Trái Phép (SIPDS)  
**Tài liệu:** Software Requirements Specification | **Phiên bản:** 1.0  

---

## 1. MỤC ĐÍCH TÀI LIỆU
Đặc tả chi tiết toàn bộ các yêu cầu chức năng (FR), yêu cầu phi chức năng (NFR), giao diện và kiến trúc của Hệ thống SIPDS phục vụ làm căn cứ lập trình, nghiệm thu.

## 2. BỐI CẢNH NGHIỆP VỤ
Hệ thống giải quyết bài toán giám sát tự động 24/7 trên camera giao thông:
- Nhận diện phương tiện chính xác.
- Đo đếm thời gian đỗ thực tế trong vùng cấm.
- Phân biệt dừng tạm thời (< T_stop) vs đỗ sai quy định (>= T_park).
- Lưu trữ bộ hồ sơ bằng chứng đầy đủ ảnh toàn cảnh, ảnh phóng to và metadata.

## 3. BẢNG ĐẶC TẢ YÊU CẦU CHỨC NĂNG (FUNCTIONAL REQUIREMENTS)
| Mã YC | Tên Chức Năng | Mô Tả Nghiệp Vụ Chi Tiết | Ưu Tiên |
|---|---|---|---|
| **FR-01** | Cấu hình Vùng cấm (ROI) | Cho phép người dùng vẽ vùng cấm đỗ đa giác bất kỳ qua GUI (`select_roi.py`), lưu vào YAML. | High |
| **FR-02** | Nhận diện Phương tiện | YOLO11 phát hiện xe (ô tô, xe máy, xe buýt, xe tải); trích xuất BBox và Score. | High |
| **FR-03** | Theo dõi Đa mục tiêu | ByteTrack gán ID duy nhất và theo dõi quỹ đạo phương tiện liên khung hình. | High |
| **FR-04** | Bảo toàn ID (Spatial Re-ID) | Phục hồi ID và bộ đếm thời gian khi phương tiện bị che khuất tạm thời (< 5s). | High |
| **FR-05** | Động cơ Quy tắc (Rule Engine) | Point-in-Polygon kiểm tra xe trong ROI; đếm thời gian phân định Dừng vs Đỗ. | High |
| **FR-06** | Tạo Hồ sơ Bằng chứng | Tự động lưu ảnh full vi phạm (vẽ ROI + BBox), ảnh crop xe và JSON metadata. | High |
| **FR-07** | Hiển thị HUD Giám sát | Vẽ HUD trực quan: BBox đổi màu (Xanh/Vàng/Đỏ), bộ đếm thời gian, số lượng vi phạm. | Medium |
| **FR-08** | Hỗ trợ Đa nguồn dữ liệu | Hỗ trợ Video MP4, luồng RTSP camera giao thông và Webcam. | High |

## 4. YÊU CẦU PHI CHỨC NĂNG (NON-FUNCTIONAL REQUIREMENTS)
- **NFR-01 (Hiệu năng):** Xử lý >= 30 FPS trên GPU NVIDIA RTX 3050; độ trễ < 35ms/frame.
- **NFR-02 (Độ chính xác):** mAP@0.5 >= 90%; tỷ lệ báo động giả < 3%.
- **NFR-03 (Độ ổn định):** Hoạt động liên tục 24/7; tự reconnect khi mất mạng RTSP sau 3s; không leak bộ nhớ.
- **NFR-04 (Linh hoạt):** Cấu hình tập trung qua YAML, mã nguồn module hóa chuẩn PEP8.
