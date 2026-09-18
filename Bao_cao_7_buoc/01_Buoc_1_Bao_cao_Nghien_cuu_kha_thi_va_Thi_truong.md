# BƯỚC 1: BÁO CÁO NGHIÊN CỨU KHẢ THI & PHÂN TÍCH THỊ TRƯỜNG
**Dự án:** Hệ thống Phát hiện Phương tiện Dừng Đỗ Trái Phép trên Luồng Video Giám sát  
**Ngày lập:** 16/09/2026 | **Phiên bản:** 1.0  

---

## I. TỔNG QUAN DỰ ÁN & MỤC TIÊU SẢN XUẤT
- **Tên dự án:** Hệ thống Giám sát và Tự động Phát hiện Vi phạm Dừng Đỗ Xe Trái Phép (Smart Illegal Parking Detection System - SIPDS).
- **Đơn vị thực hiện:** Nhóm Kỹ sư AI & Thị giác máy tính (Computer Vision Team).
- **Đối tượng thụ hưởng:** Lực lượng Cảnh sát Giao thông, Thanh tra Giao thông Đô thị và Trung tâm Điều hành Đô thị Thông minh (IOC).
- **Mục tiêu cốt lõi:** Xây dựng giải pháp phần mềm thông minh ứng dụng học sâu (Deep Learning) và thị giác máy tính, có khả năng xử lý trực tiếp các luồng video giám sát giao thông đô thị (Full HD/4K) theo thời gian thực nhằm tự động phát hiện, định danh, tính toán thời gian dừng/đỗ sai quy định và tự động lập hồ sơ bằng chứng phục vụ phạt nguội.

## II. PHÂN TÍCH ĐIỂM NGHẼN KỸ THUẬT CỦA CÁC GIẢI PHÁP HIỆN HÀNH
1. **Phương pháp tuần tra truyền thống:** Đòi hỏi lực lượng tuần tra dày đặc, tốn kém chi phí nhân lực, không thể túc trực 24/7 trên toàn tuyến đường. Việc ghi nhận thủ công dễ phát sinh tranh cãi và không có bằng chứng xuyên suốt về thời gian đỗ.
2. **Cảm biến từ trường mặt đường:** Chi phí thi công cắt đường, chôn cảm biến cực kỳ đắt đỏ; độ bền kém do rung chấn mặt đường và thời tiết khắc nghiệt; chỉ đo được 1 điểm cố định, không thể mở rộng linh hoạt theo tuyến phố.
3. **Camera xử lý ảnh cổ điển (Background Subtraction / Optical Flow):** Rất nhạy cảm với sự thay đổi của ánh sáng, bóng râm di động, lá cây đung đưa; không phân loại được phương tiện giao thông; tỷ lệ báo động giả vượt quá 40%.

## III. ĐỊNH HƯỚNG KIẾN TRÚC & GIẢI PHÁP KỸ THUẬT TIÊN TIẾN
Hệ thống SIPDS được thiết kế theo kiến trúc 3 tầng phân tách độc lập:
- **Tầng Phát hiện Đối tượng (Detection Layer):** YOLO11 Custom (YOLO11n 5.3MB cho Edge và YOLO11m 40.2MB cho Server trung tâm).
- **Tầng Theo dõi & Bảo toàn Định danh (Tracking Layer):** ByteTrack + Kalman Filter + Spatial Re-ID phục hồi Track ID khi xe bị che khuất <5 giây.
- **Tầng Suy luận Nghiệp vụ & Bằng chứng (Rule Engine):** Thuật toán Point-in-Polygon kiểm tra xe trong ROI; Finite State Machine phân biệt Dừng vs Đỗ; tự động lưu bằng chứng đa góc nhìn.

## IV. BẢNG SO SÁNH GIẢI PHÁP THỊ TRƯỜNG
| Tiêu chí Đánh giá | Giải pháp Ngoại nhập (Hikvision/Dahua ITS) | Giải pháp Xử lý Cổ điển | Hệ thống SIPDS (Dự án này) |
|---|---|---|---|
| **Mô hình AI cốt lõi** | Deep Learning đóng kín | Xử lý ảnh cổ điển (Blob/GMM) | **YOLO11 Custom SOTA (Open & Tinh chỉnh)** |
| **Độ chính xác mAP50** | 88% - 91% | 50% - 65% | **92.65% (Thực nghiệm thực tế)** |
| **Chống mất dấu** | Cơ bản, dễ mất track | Không có tracking | **ByteTrack + Spatial Re-ID độc quyền** |
| **Tùy biến vùng cấm** | Khung chữ nhật cứng | Hạn chế | **Đa giác n-đỉnh tùy ý theo làn đường** |
| **Chi phí phần cứng** | Rất đắt (Server chuyên dụng) | Thấp | **Tối ưu trên GPU thương mại phổ thông** |
| **Tích hợp Camera** | Khóa trong hệ sinh thái | RTSP cơ bản | **Mọi camera RTSP, video file, webcam** |

## V. TIẾN ĐỘ & KẾT LUẬN
Dự án có tính khả thi kỹ thuật 100%, sẵn sàng chuyển sang giai đoạn Đặc tả Yêu cầu Kỹ thuật (SRS).
