"""
Công Cụ So Sánh Trực Quan & Định Lượng Giữa Mô Hình Custom (output_runs) và Mô Hình Gốc Baseline (yolo11m)
---------------------------------------------------------------------------------------------------------
Chạy script này để đánh giá chất lượng phát hiện (số lượng xe, độ tin cậy, độ trễ/FPS) 
và xuất ảnh so sánh song song (side-by-side) vào thư mục evidence/model_comparison.jpg.
"""

import os
import sys
import time
import cv2
import numpy as np

# Thêm đường dẫn project
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, 'src'))

from model_utils import load_yolo_model, get_target_classes, get_model_info

def draw_detections(frame, results, model_name, target_classes, color=(0, 255, 0), latency_ms=0.0):
    """Vẽ bounding boxes và thông số lên frame."""
    annotated = frame.copy()
    boxes = results[0].boxes
    count = 0
    confs = []

    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().numpy()
        conf_arr = boxes.conf.cpu().numpy()
        cls_arr = boxes.cls.cpu().numpy().astype(int)

        for b, conf, cls_id in zip(xyxy, conf_arr, cls_arr):
            x1, y1, x2, y2 = map(int, b)
            confs.append(conf)
            count += 1
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            lbl = f"ID:{cls_id} {conf:.2f}"
            cv2.rectangle(annotated, (x1, max(0, y1 - 20)), (x1 + 80, max(0, y1)), color, -1)
            cv2.putText(annotated, lbl, (x1 + 2, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

    # Header thống kê trên frame
    avg_c = np.mean(confs) if confs else 0.0
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (annotated.shape[1], 45), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

    title = f"[{model_name}] Xe: {count} | Avg Conf: {avg_c:.2f} | Latency: {latency_ms:.1f}ms ({1000/max(1e-3, latency_ms):.1f} FPS)"
    cv2.putText(annotated, title, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    return annotated, count, avg_c

def main():
    print("=" * 70)
    print("   CÔNG CỤ SO SÁNH HIỆU NĂNG MÔ HÌNH NHẬN DIỆN XE")
    print("=" * 70)

    video_path = os.path.join(ROOT_DIR, 'data', 'sample.mp4')
    if not os.path.exists(video_path):
        print(f"[LỖI] Không tìm thấy video test: {video_path}")
        return

    # 1. Tải 2 mô hình
    print("\n[1/3] Đang tải các mô hình...")
    custom_weights = 'output_runs/best.pt'
    baseline_weights = 'yolo11m.pt'

    model_custom, path_c, _ = load_yolo_model(custom_weights, root_dir=ROOT_DIR)
    model_baseline, path_b, _ = load_yolo_model(baseline_weights, root_dir=ROOT_DIR)

    info_c = get_model_info(model_custom, path_c)
    info_b = get_model_info(model_baseline, path_b)

    cls_c = get_target_classes(model_custom)
    cls_b = get_target_classes(model_baseline)

    print(f"  - Model A (Custom):   {info_c['filename']} ({info_c['size_mb']} MB) | Classes: {cls_c}")
    print(f"  - Model B (Baseline): {info_b['filename']} ({info_b['size_mb']} MB) | Classes: {cls_b}")

    # 2. Chạy so sánh trên các frame video
    print("\n[2/3] Đang phân tích và đo lường trên 60 frames mẫu...")
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    frames_tested = 0
    max_test_frames = min(60, total_frames)

    times_c, counts_c, confs_c = [], [], []
    times_b, counts_b, confs_b = [], [], []

    saved_comparison_frame = None

    while frames_tested < max_test_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Inference Model Custom
        t0 = time.time()
        res_c = model_custom(frame, conf=0.3, classes=cls_c, imgsz=960, verbose=False)
        dt_c = (time.time() - t0) * 1000.0

        # Inference Model Baseline
        t0 = time.time()
        res_b = model_baseline(frame, conf=0.3, classes=cls_b, imgsz=960, verbose=False)
        dt_b = (time.time() - t0) * 1000.0

        times_c.append(dt_c)
        times_b.append(dt_b)

        # Trích xuất số lượng và conf
        if res_c[0].boxes is not None:
            counts_c.append(len(res_c[0].boxes))
            if len(res_c[0].boxes) > 0:
                confs_c.extend(res_c[0].boxes.conf.cpu().numpy().tolist())
        if res_b[0].boxes is not None:
            counts_b.append(len(res_b[0].boxes))
            if len(res_b[0].boxes) > 0:
                confs_b.extend(res_b[0].boxes.conf.cpu().numpy().tolist())

        # Lấy frame thứ 15 để ghép ảnh so sánh trực quan
        if frames_tested == 15 or (saved_comparison_frame is None and frames_tested == max_test_frames - 1):
            ann_c, _, _ = draw_detections(frame, res_c, "CUSTOM MODEL (output_runs)", cls_c, color=(0, 230, 118), latency_ms=dt_c)
            ann_b, _, _ = draw_detections(frame, res_b, "BASELINE MODEL (yolo11m)", cls_b, color=(30, 136, 229), latency_ms=dt_b)
            # Ghép song song (Side-by-Side)
            saved_comparison_frame = np.hstack([ann_c, ann_b])

        frames_tested += 1

    cap.release()

    # 3. Lưu ảnh so sánh
    evidence_dir = os.path.join(ROOT_DIR, 'evidence')
    os.makedirs(evidence_dir, exist_ok=True)
    if saved_comparison_frame is not None:
        comp_img_path = os.path.join(evidence_dir, 'model_comparison.jpg')
        cv2.imwrite(comp_img_path, saved_comparison_frame)
        print(f"\n[3/3] Đã lưu ảnh so sánh trực quan song song vào:\n      -> {comp_img_path}")

    # Bảng tổng kết kết quả
    avg_t_c = np.mean(times_c) if times_c else 0.0
    avg_t_b = np.mean(times_b) if times_b else 0.0
    avg_cnt_c = np.mean(counts_c) if counts_c else 0.0
    avg_cnt_b = np.mean(counts_b) if counts_b else 0.0
    avg_cnf_c = np.mean(confs_c) if confs_c else 0.0
    avg_cnf_b = np.mean(confs_b) if confs_b else 0.0

    size_c_str = f"{info_c['size_mb']} MB"
    size_b_str = f"{info_b['size_mb']} MB"
    lat_c_str = f"{avg_t_c:.1f} ms"
    lat_b_str = f"{avg_t_b:.1f} ms"
    fps_c_str = f"{1000/max(1e-3, avg_t_c):.1f} FPS"
    fps_b_str = f"{1000/max(1e-3, avg_t_b):.1f} FPS"
    cnt_c_str = f"{avg_cnt_c:.1f} xe"
    cnt_b_str = f"{avg_cnt_b:.1f} xe"
    cnf_c_str = f"{avg_cnf_c:.2f}"
    cnf_b_str = f"{avg_cnf_b:.2f}"

    print("\n" + "=" * 70)
    print("                    BẢNG TỔNG KẾT SO SÁNH")
    print("=" * 70)
    print(f"{'Tiêu chí đánh giá':<30} | {'Custom (output_runs)':<18} | {'Baseline (yolo11m)':<18}")
    print("-" * 70)
    print(f"{'Tên file trọng số':<30} | {info_c['filename']:<18} | {info_b['filename']:<18}")
    print(f"{'Dung lượng mô hình':<30} | {size_c_str:<18} | {size_b_str:<18}")
    print(f"{'Độ trễ trung bình (ms/frame)':<30} | {lat_c_str:<18} | {lat_b_str:<18}")
    print(f"{'Tốc độ ước tính (FPS)':<30} | {fps_c_str:<18} | {fps_b_str:<18}")
    print(f"{'Số xe TB phát hiện / frame':<30} | {cnt_c_str:<18} | {cnt_b_str:<18}")
    print(f"{'Độ tin cậy TB (Confidence)':<30} | {cnf_c_str:<18} | {cnf_b_str:<18}")
    print("=" * 70)

    # Đưa ra nhận xét khách quan
    print("\n[NHẬN XÉT & HƯỚNG DẪN LỰA CHỌN]:")
    if avg_t_c < avg_t_b:
        speedup = avg_t_b / max(1e-3, avg_t_c)
        print(f"⚡ Tốc độ: Mô hình Custom nhẹ hơn và nhanh gấp {speedup:.1f} lần so với Baseline!")
    if avg_cnt_c >= avg_cnt_b:
        print(f"🎯 Độ nhạy: Mô hình Custom bắt được lượng xe tương đương hoặc nhiều hơn!")
    else:
        print(f"ℹ️ Độ bao quát: Mô hình Baseline bắt được nhiều phương tiện ở xa hơn do tập huấn luyện COCO đa dạng hơn.")
    print("👉 Bạn có thể chuyển đổi linh hoạt bất cứ lúc nào trên giao diện Web Dashboard hoặc dùng lệnh:")
    print("   - Chạy mô hình Custom:    python src/main.py")
    print("   - Chạy mô hình Baseline:  python src/main.py --baseline")
    print("=" * 70)

if __name__ == '__main__':
    main()
