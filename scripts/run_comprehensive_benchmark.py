"""
Công Cụ So Sánh Đa Chiều Giữa Các Mô Hình:
1. YOLO11m Custom (Mới train trên Kaggle - 40.5MB)
2. YOLO11n Custom (Đã train trước đó - 5.2MB)
3. YOLO11m Baseline (Gốc COCO - 40.7MB)
"""

import os
import sys
import time
import cv2
import numpy as np

# Thiết lập encoding UTF-8
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, 'src'))

from model_utils import load_yolo_model, get_target_classes, get_model_info, get_safe_device

def draw_detections(frame, results, model_label, target_classes, color=(0, 255, 0), latency_ms=0.0):
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
            lbl = f"Xe {conf:.2f}"
            cv2.rectangle(annotated, (x1, max(0, y1 - 22)), (x1 + 85, max(0, y1)), color, -1)
            cv2.putText(annotated, lbl, (x1 + 4, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1)

    avg_c = np.mean(confs) if confs else 0.0
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (annotated.shape[1], 46), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.8, annotated, 0.2, 0, annotated)

    title = f"[{model_label}] Xe: {count} | Avg Conf: {avg_c:.2f} | Latency: {latency_ms:.1f}ms ({1000/max(1e-3, latency_ms):.1f} FPS)"
    cv2.putText(annotated, title, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    return annotated, count, avg_c

def main():
    print("=" * 75)
    print("       ĐÁNH GIÁ VÀ SO SÁNH MÔ HÌNH MỚI TRAIN: YOLO11m vs YOLO11n vs BASELINE")
    print("=" * 75)

    video_path = os.path.join(ROOT_DIR, 'data', 'sample.mp4')
    if not os.path.exists(video_path):
        print(f"[LỖI] Không tìm thấy video: {video_path}")
        return

    device = get_safe_device('cpu')

    weights_dict = {
        'YOLO11m Custom (Mới Train)': 'output_runs/yolo11m_custom.pt',
        'YOLO11n Custom (5MB)': 'output_runs/best.pt',
        'YOLO11m Baseline (Gốc COCO)': 'yolo11m.pt'
    }

    models = {}
    infos = {}
    target_cls = {}

    for name, w_path in weights_dict.items():
        if os.path.exists(os.path.join(ROOT_DIR, w_path)):
            m, p, _ = load_yolo_model(w_path, device=device, root_dir=ROOT_DIR)
            models[name] = m
            infos[name] = get_model_info(m, p)
            target_cls[name] = get_target_classes(m)
            print(f"✅ Đã nạp: {name:<28} | Kích thước: {infos[name]['size_mb']} MB | Lớp: {target_cls[name]}")

    print(f"\nThiết bị chạy đo lường: {device.upper()}")
    print("Đang chạy phân tích trên 30 frames mẫu từ video...")

    cap = cv2.VideoCapture(video_path)
    frames_tested = 0
    max_frames = 30

    metrics = {k: {'times': [], 'counts': [], 'confs': []} for k in models}
    comparison_frames = {}

    while frames_tested < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Frame thứ 15 để lấy ảnh so sánh trực quan
        is_capture_frame = (frames_tested == 15)

        for name, m in models.items():
            t0 = time.time()
            res = m(frame, conf=0.3, classes=target_cls[name], imgsz=960, device=device, verbose=False)
            dt = (time.time() - t0) * 1000.0

            metrics[name]['times'].append(dt)
            n_boxes = len(res[0].boxes) if res[0].boxes is not None else 0
            metrics[name]['counts'].append(n_boxes)
            if n_boxes > 0:
                metrics[name]['confs'].extend(res[0].boxes.conf.cpu().numpy().tolist())

            if is_capture_frame:
                color = (0, 230, 118) if 'Mới Train' in name else ((255, 193, 7) if 'YOLO11n' in name else (33, 150, 243))
                ann, _, _ = draw_detections(frame, res, name, target_cls[name], color=color, latency_ms=dt)
                comparison_frames[name] = ann

        frames_tested += 1

    cap.release()

    # Lưu ảnh so sánh
    evidence_dir = os.path.join(ROOT_DIR, 'evidence')
    os.makedirs(evidence_dir, exist_ok=True)

    # Ghép ảnh YOLO11m Mới vs YOLO11n
    if 'YOLO11m Custom (Mới Train)' in comparison_frames and 'YOLO11n Custom (5MB)' in comparison_frames:
        side_by_side = np.hstack([comparison_frames['YOLO11m Custom (Mới Train)'], comparison_frames['YOLO11n Custom (5MB)']])
        cv2.imwrite(os.path.join(evidence_dir, 'comparison_yolo11m_vs_yolo11n.jpg'), side_by_side)
        cv2.imwrite(os.path.join(evidence_dir, 'model_comparison.jpg'), side_by_side)

    # In bảng tổng kết
    print("\n" + "=" * 85)
    print("                           BẢNG TỔNG KẾT HIỆU NĂNG THỰC TẾ")
    print("=" * 85)
    header = f"{'Tiêu chí':<26} | {'YOLO11m Custom (Mới)':<22} | {'YOLO11n Custom (5MB)':<20} | {'YOLO11m Gốc COCO':<18}"
    print(header)
    print("-" * 85)

    def get_stat(k):
        t = np.mean(metrics[k]['times']) if metrics[k]['times'] else 0.0
        fps = 1000 / max(1e-3, t)
        cnt = np.mean(metrics[k]['counts']) if metrics[k]['counts'] else 0.0
        cnf = np.mean(metrics[k]['confs']) if metrics[k]['confs'] else 0.0
        return t, fps, cnt, cnf

    t_m, fps_m, cnt_m, cnf_m = get_stat('YOLO11m Custom (Mới Train)')
    t_n, fps_n, cnt_n, cnf_n = get_stat('YOLO11n Custom (5MB)')
    t_b, fps_b, cnt_b, cnf_b = get_stat('YOLO11m Baseline (Gốc COCO)')

    print(f"{'Dung lượng weights':<26} | {'40.5 MB':<22} | {'5.2 MB':<20} | {'40.7 MB':<18}")
    print(f"{'Độ trễ TB (ms/frame)':<26} | {f'{t_m:.1f} ms':<22} | {f'{t_n:.1f} ms':<20} | {f'{t_b:.1f} ms':<18}")
    print(f"{'Tốc độ CPU (FPS)':<26} | {f'{fps_m:.1f} FPS':<22} | {f'{fps_n:.1f} FPS':<20} | {f'{fps_b:.1f} FPS':<18}")
    print(f"{'Số xe TB phát hiện/frame':<26} | {f'{cnt_m:.1f} xe':<22} | {f'{cnt_n:.1f} xe':<20} | {f'{cnt_b:.1f} xe':<18}")
    print(f"{'Độ tin cậy TB (Confidence)':<26} | {f'{cnf_m:.2f}':<22} | {f'{cnf_n:.2f}':<20} | {f'{cnf_b:.2f}':<18}")
    print("=" * 85)

if __name__ == '__main__':
    main()
