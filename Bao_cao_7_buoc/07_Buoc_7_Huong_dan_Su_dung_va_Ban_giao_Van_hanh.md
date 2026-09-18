# BƯỚC 7: TÀI LIỆU HƯỚNG DẪN SỬ DỤNG VÀ QUẢN TRỊ HỆ THỐNG
**Hệ thống:** Giám sát và Tự động Phát hiện Dừng Đỗ Xe Trái Phép (SIPDS)  
**Tài liệu:** User Guide & Operations Manual | **Phiên bản:** 1.0  

---

## I. YÊU CẦU HỆ THỐNG
- **Phần cứng:** CPU Intel i5/i7, RAM >= 8GB, GPU NVIDIA GTX 1650 / RTX 3050 trở lên.
- **Phần mềm:** Windows 10/11 hoặc Linux Ubuntu, Python 3.10-3.13, CUDA Toolkit 12.x.

## II. QUY TRÌNH KHỞI CHẠY TỪNG BƯỚC
1. **Cài đặt môi trường:**
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
   pip install -r requirements.txt
   ```
2. **Cấu hình tham số (`config/settings.yaml`):**
   - Đặt `model.weights_path: "weights/best_yolo11m.pt"` để đạt độ chính xác cao nhất (92.65%).
   - Cài đặt thời gian dừng tạm thời `stop_time_threshold: 10` và thời gian đỗ vi phạm `park_time_threshold: 30`.
3. **Vẽ vùng cấm đỗ tương tác:**
   ```bash
   python scripts/select_roi.py --source data/sample.mp4
   ```
   *Nhấp chuột các góc đa giác -> Nhấn phím 's' để lưu lại.*
4. **Chạy hệ thống giám sát:**
   ```bash
   python src/main.py
   ```

## III. QUY ƯỚC MÀU SẮC GIÁM SÁT
- **Màu Xanh lá:** Xe lưu thông bình thường.
- **Màu Vàng:** Cảnh báo dừng xe trong vùng cấm (10s - 30s).
- **Màu Đỏ:** Vi phạm đỗ xe trái phép (> 30s) -> Tự động ghi nhận bằng chứng vào `runs/evidence/`.

## IV. BẢO HÀNH & XỬ LÝ SỰ CỐ
- Định kỳ sao lưu dữ liệu ảnh vi phạm trong `runs/evidence/`.
- Nếu GPU bị đầy bộ nhớ, chuyển sang dùng `weights/best.pt` (YOLO11n 5.3MB) để tiết kiệm tài nguyên.
