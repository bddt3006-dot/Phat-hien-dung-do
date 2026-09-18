# BƯỚC 6: BÁO CÁO ĐÁNH GIÁ CÁC PHIÊN BẢN (ALPHA, BETA, RELEASE)
**Hệ thống:** Giám sát và Tự động Phát hiện Dừng Đỗ Xe Trái Phép (SIPDS)  
**Tài liệu:** Release & Version Evaluation Report | **Phiên bản:** 1.0  

---

## I. MỤC ĐÍCH
Ghi nhận quá trình trưởng thành của phần mềm qua 3 cột mốc: Alpha -> Beta -> Release.

## II. LỊCH TRÌNH TIẾN HÓA CÁC PHIÊN BẢN
1. **Bản Alpha (01/09/2026):** Kiểm chứng mô hình YOLO11n cơ bản. Hạn chế: Chưa có ByteTrack, ID bị nhảy liên tục, chưa có bộ đếm thời gian đỗ.
2. **Bản Beta (10/09/2026):** Tích hợp ByteTrack, thuật toán Spatial Re-ID chống mất track, hoàn thiện Rule Engine đếm thời gian dừng/đỗ, công cụ chọn ROI qua GUI.
3. **Bản Release (16/09/2026) - BẢN PHÁT HÀNH CHÍNH THỨC:**
   - Tích hợp mô hình YOLO11m Custom đạt **92.65% mAP@0.5**.
   - Hiệu năng xử lý đạt **67.5 FPS** trên GPU RTX 3050.
   - Tự động lưu bằng chứng phạt nguội 3 thành phần: Ảnh toàn cảnh + Ảnh crop xe + File JSON metadata.
   - Cấu hình tập trung hoàn toàn qua `config/settings.yaml`.

## III. KẾT QUẢ NGHIỆM THU PHÁT HÀNH
Toàn bộ 7/7 tiêu chí kỹ thuật đều **ĐẠT YÊU CẦU PHÁT HÀNH CHÍNH THỨC**.
