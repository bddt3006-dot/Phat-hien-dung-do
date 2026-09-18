"""
Script Đánh Giá (Benchmark) Detection & Tracking Trên Dữ Liệu CityFlow
Dành cho việc kiểm thử trực tiếp trên máy local với các model đã train.
----------------------------------------------------------------------
Cách dùng:
1. Tải file cityflow_test_pack.zip từ Kaggle về và giải nén vào thư mục: data/cityflow/
   Cấu trúc mong đợi:
   data/cityflow/
     ├── c006/
     │    ├── video.mp4
     │    ├── gt.txt
     │    └── roi.jpg
     └── c007/ (tùy chọn)
2. Chạy: python scripts/benchmark_cityflow.py
"""

import os
import sys
import time
import glob
import cv2
import numpy as np
import pandas as pd

# Thêm đường dẫn project
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, 'src'))

from model_utils import load_yolo_model, get_target_classes, get_safe_device

def compute_iou(box1, box2):
    """Tính IoU giữa 2 bounding box [x1, y1, x2, y2]."""
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    area1 = max(1e-5, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    area2 = max(1e-5, (box2[2] - box2[0]) * (box2[3] - box2[1]))
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0

def load_gt_by_frame(gt_path):
    """Đọc file gt.txt chuẩn MOT Challenge thành dict {frame_idx: [(id, [x1, y1, x2, y2])]}."""
    cols = ['frame', 'track_id', 'left', 'top', 'width', 'height', 'conf', 'x', 'y', 'z']
    df = pd.read_csv(gt_path, header=None, names=cols)
    gt_dict = {}
    for f_idx, grp in df.groupby('frame'):
        boxes = []
        for _, row in grp.iterrows():
            x1 = float(row['left'])
            y1 = float(row['top'])
            x2 = x1 + float(row['width'])
            y2 = y1 + float(row['height'])
            boxes.append((int(row['track_id']), [x1, y1, x2, y2]))
        gt_dict[int(f_idx)] = boxes
    return gt_dict

def evaluate_model_on_sequence(model_path, model_name, video_path, gt_dict, device='cpu', max_eval_frames=500):
    print(f"\n--- Đang đánh giá mô hình: [{model_name}] ---")
    model, _, _ = load_yolo_model(model_path, device=device, root_dir=ROOT_DIR)
    target_classes = get_target_classes(model, None)

    cap = cv2.VideoCapture(video_path)
    fps_video = cap.get(cv2.CAP_PROP_FPS) or 25.0
    
    total_gt = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_id_switches = 0
    iou_scores = []
    latencies = []
    id_mapping = {}

    frame_idx = 0
    while frame_idx < max_eval_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        t0 = time.perf_counter()
        results = model.track(
            frame,
            tracker="config/bytetrack_custom.yaml" if os.path.exists(os.path.join(ROOT_DIR, "config/bytetrack_custom.yaml")) else "bytetrack.yaml",
            conf=0.25,
            classes=target_classes,
            device=device,
            persist=True,
            verbose=False
        )
        t_ms = (time.perf_counter() - t0) * 1000
        latencies.append(t_ms)

        pred_boxes = []
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            for b, tid in zip(boxes_xyxy, ids):
                pred_boxes.append((tid, b.tolist()))

        current_gt = gt_dict.get(frame_idx, [])
        total_gt += len(current_gt)

        matched_gt = set()
        for p_idx, (p_id, p_box) in enumerate(pred_boxes):
            best_iou = 0.0
            best_g_idx = -1
            for g_idx, (g_id, g_box) in enumerate(current_gt):
                if g_idx in matched_gt:
                    continue
                iou = compute_iou(p_box, g_box)
                if iou > best_iou:
                    best_iou = iou
                    best_g_idx = g_idx

            if best_iou >= 0.5 and best_g_idx != -1:
                matched_gt.add(best_g_idx)
                total_tp += 1
                iou_scores.append(best_iou)

                g_id = current_gt[best_g_idx][0]
                if g_id in id_mapping:
                    if id_mapping[g_id] != p_id:
                        total_id_switches += 1
                        id_mapping[g_id] = p_id
                else:
                    id_mapping[g_id] = p_id
            else:
                total_fp += 1

        total_fn += (len(current_gt) - len(matched_gt))

    cap.release()

    precision = total_tp / max(1, (total_tp + total_fp))
    recall = total_tp / max(1, (total_tp + total_fn))
    f1 = 2 * precision * recall / max(1e-6, (precision + recall))
    mean_iou = np.mean(iou_scores) if iou_scores else 0.0
    avg_latency = np.mean(latencies) if latencies else 0.0
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0
    mota = 1.0 - (total_fn + total_fp + total_id_switches) / max(1, total_gt)

    return {
        'model_name': model_name,
        'frames_evaluated': frame_idx,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mean_iou': mean_iou,
        'id_switches': total_id_switches,
        'mota': mota,
        'latency_ms': avg_latency,
        'fps': fps
    }

def main():
    print("=" * 75)
    print("      BENCHMARK ĐÁNH GIÁ MÔ HÌNH TRÊN DỮ LIỆU THỰC TẾ CITYFLOW")
    print("=" * 75)

    # Tìm các thư mục camera đã tải về trong data/cityflow/
    cityflow_dir = os.path.join(ROOT_DIR, 'data', 'cityflow')
    if not os.path.exists(cityflow_dir):
        # Thử tìm trực tiếp trong data
        cityflow_dir = os.path.join(ROOT_DIR, 'data')

    cams = glob.glob(os.path.join(cityflow_dir, "*", "video.mp4")) + glob.glob(os.path.join(cityflow_dir, "video.mp4"))
    if not cams:
        # Tìm bất kỳ file .mp4 hoặc .avi trong data/cityflow
        cams = glob.glob(os.path.join(cityflow_dir, "**", "*.mp4"), recursive=True) + glob.glob(os.path.join(cityflow_dir, "**", "*.avi"), recursive=True)

    if not cams:
        print(f"❌ Chưa tìm thấy dữ liệu CityFlow trong thư mục: {cityflow_dir}")
        print("💡 Vui lòng chạy script trên Kaggle để tải cityflow_test_pack.zip về,")
        print("   sau đó giải nén vào thư mục 'data/cityflow/' rồi chạy lại script này.")
        return

    video_path = cams[0]
    cam_folder = os.path.dirname(video_path)
    gt_path = os.path.join(cam_folder, "gt.txt")
    if not os.path.exists(gt_path):
        gt_candidates = glob.glob(os.path.join(cam_folder, "**", "gt.txt"), recursive=True)
        if gt_candidates:
            gt_path = gt_candidates[0]

    print(f"📹 Video: {video_path}")
    print(f"📄 Ground Truth: {gt_path if os.path.exists(gt_path) else 'Không tìm thấy gt.txt'}")

    if not os.path.exists(gt_path):
        print("❌ Không tìm thấy file gt.txt để so khớp nhãn ground truth!")
        return

    gt_dict = load_gt_by_frame(gt_path)
    device = get_safe_device('cpu')
    print(f"⚙️ Thiết bị inference: {device.upper()}")

    # Danh sách các mô hình cần so sánh
    models_to_test = []
    if os.path.exists(os.path.join(ROOT_DIR, 'output_runs', 'yolo11m_custom.pt')):
        models_to_test.append(('output_runs/yolo11m_custom.pt', 'YOLO11m Fine-tuned'))
    if os.path.exists(os.path.join(ROOT_DIR, 'output_runs', 'best.pt')):
        models_to_test.append(('output_runs/best.pt', 'YOLO11n Fine-tuned'))

    if not models_to_test:
        models_to_test.append(('yolo11n.pt', 'YOLO11n Pretrained'))

    results = []
    for m_path, m_name in models_to_test:
        res = evaluate_model_on_sequence(m_path, m_name, video_path, gt_dict, device=device, max_eval_frames=400)
        results.append(res)

    print("\n" + "=" * 85)
    print("                    BẢNG KẾT QUẢ SO SÁNH TRÊN CITYFLOW")
    print("=" * 85)
    print(f"{'Mô hình':<22} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Mean IoU':<10} | {'ID Switches':<12} | {'MOTA':<10} | {'FPS':<8}")
    print("-" * 105)
    for r in results:
        print(f"{r['model_name']:<22} | {r['precision']*100:>8.2f}% | {r['recall']*100:>8.2f}% | {r['f1']*100:>8.2f}% | {r['mean_iou']*100:>8.2f}% | {r['id_switches']:>12} | {r['mota']*100:>8.2f}% | {r['fps']:>8.1f}")
    print("=" * 85)

if __name__ == '__main__':
    main()
