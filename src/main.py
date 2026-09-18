import cv2
import yaml
import numpy as np
import os
import csv
import sys
import argparse
from datetime import datetime

# Thêm src vào sys.path
sys.path.insert(0, os.path.dirname(__file__))
from ultralytics import YOLO
from rule_engine import ParkingRuleEngine
from model_utils import load_yolo_model, get_target_classes, get_model_info, get_safe_device, DEFAULT_BASELINE_WEIGHTS

def parse_args():
    parser = argparse.ArgumentParser(description="Chương trình Giám Sát Dừng Đỗ Xe Thông Minh")
    parser.add_argument('--weights', type=str, default=None,
                        help="Đường dẫn tới file trọng số model (VD: output_runs/best.pt hoặc yolo11m.pt)")
    parser.add_argument('--baseline', action='store_true',
                        help="Bật ngay mô hình gốc (yolo11m.pt) để làm đường lui so sánh mà không cần sửa cấu hình")
    parser.add_argument('--video', type=str, default=None,
                        help="Đường dẫn file video đầu vào (ghi đè cấu hình trong settings.yaml)")
    parser.add_argument('--device', type=str, default=None,
                        help="Thiết bị chạy (VD: cuda:0 hoặc cpu)")
    return parser.parse_args()

def main():
    args = parse_args()

    # 1. Tải cấu hình
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    config_path = os.path.join(root_dir, 'config', 'settings.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 2. Khởi tạo Model (Hỗ trợ đường lui Baseline)
    baseline_w = config['model'].get('baseline_weights', DEFAULT_BASELINE_WEIGHTS)
    if args.baseline:
        selected_weights = baseline_w
        print(f"[ĐƯỜNG LUI] Kích hoạt mô hình gốc Baseline: {selected_weights}")
    elif args.weights:
        selected_weights = args.weights
        print(f"[THAM SỐ] Chỉ định mô hình qua dòng lệnh: {selected_weights}")
    else:
        selected_weights = config['model'].get('weights', 'output_runs/best.pt')

    configured_device = args.device or config['model'].get('device', 'cpu')
    device = get_safe_device(configured_device)

    model, loaded_path, is_fallback = load_yolo_model(
        weights_path=selected_weights,
        device=device,
        fallback_path=baseline_w,
        root_dir=root_dir
    )

    # Tự động phân giải các lớp cần phát hiện (Smart Class Resolver)
    target_classes = get_target_classes(model, config['model'].get('classes', None))
    model_info = get_model_info(model, loaded_path)

    print("=" * 60)
    print(f"  MÔ HÌNH NHẬN DIỆN: {model_info['filename']}")
    print(f"  LOẠI MÔ HÌNH:      {model_info['type']}")
    print(f"  LỚP PHÁT HIỆN:     {target_classes} (Tổng {model_info['num_classes']} lớp)")
    print(f"  THIẾT BỊ CHẠY:     {device.upper()}")
    if is_fallback:
        print("  [LƯU Ý] Đang chạy ở chế độ FALLBACK (mô hình gốc)")
    print("=" * 60)

    # Thư mục lưu bằng chứng
    evidence_dir = os.path.join(root_dir, 'evidence')
    os.makedirs(evidence_dir, exist_ok=True)
    evidence_csv = os.path.join(evidence_dir, 'violation_log.csv')
    if not os.path.exists(evidence_csv):
        with open(evidence_csv, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(['Timestamp', 'Track_ID', 'Dwell_Time_Seconds', 'Image_Path'])
    
    # 3. Khởi tạo Rule Engine
    rule_engine = ParkingRuleEngine(config)
    
    # 4. Mở Video
    raw_video = args.video or config['video']['source']
    video_source = os.path.join(root_dir, raw_video) if not os.path.isabs(raw_video) else raw_video
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"[LỖI] Không thể mở video: {video_source}")
        return
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30.0
        
    frame_count = 0
    
    print(f"Bắt đầu xử lý video '{os.path.basename(video_source)}'... Bấm 'q' để thoát.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Đã hết video!")
            break
            
        current_time = frame_count / fps
        frame_count += 1
        
        # Inference bằng YOLO kèm Tracking
        # Sử dụng target_classes đã được tự động chuẩn hóa
        results = model.track(
            frame,
            tracker=config['tracking']['tracker'],
            conf=config['model']['conf_threshold'],
            imgsz=config['model']['imgsz'],
            classes=target_classes,
            device=device,
            persist=True,
            verbose=False
        )
        
        # Trích xuất thông tin tracking
        tracks = []
        boxes_info = []
        
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            
            for box, track_id, cls in zip(boxes, ids, classes):
                x1, y1, x2, y2 = box
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                tracks.append((track_id, cx, cy, x1, y1, x2, y2))
                boxes_info.append((track_id, x1, y1, x2, y2))
                
        # Cập nhật Rule Engine
        violations = rule_engine.update(tracks, current_time)
        violation_ids = [v['track_id'] for v in violations]
        
        # Lưu ảnh bằng chứng khi phát hiện vi phạm mới
        for v in violations:
            if v.get('is_new'):
                tid = v['track_id']
                dur = v['duration']
                t_str = datetime.now().strftime('%Y%m%d_%H%M%S')
                img_name = f"violation_{tid}_{t_str}.jpg"
                img_p = os.path.join(evidence_dir, img_name)
                
                # Tạo ảnh bằng chứng
                ev_frame = frame.copy()
                cv2.putText(ev_frame, f"VIOLATION ID: {tid} - DWELL: {dur:.1f}s", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                cv2.imwrite(img_p, ev_frame)
                
                with open(evidence_csv, 'a', newline='', encoding='utf-8') as f:
                    csv.writer(f).writerow([t_str, tid, round(dur, 1), img_name])
                print(f"[CANH BAO] Phat hien xe ID {tid} do qua {dur:.1f}s! Da luu anh: {img_name}")
        
        # Vẽ ROI
        cv2.polylines(frame, [rule_engine.roi_polygon], True, (255, 0, 0), 2)
        cv2.putText(frame, "Vung Cam Dung Do", (rule_engine.roi_polygon[0][0], rule_engine.roi_polygon[0][1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Vẽ thông tin lên Frame
        for (track_id, x1, y1, x2, y2) in boxes_info:
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            rec = rule_engine.history.get(track_id)
            
            # Đổi màu nếu vi phạm hoặc dừng đỗ
            if track_id in violation_ids:
                color = (0, 0, 255) # Đỏ vi phạm
                dur = rec.get('dwell_time', 0.0) if rec else 0.0
                status_txt = f"VIOLATION ID {track_id} | {dur:.1f}s"
            elif rec and rec.get('is_stopped') and rec.get('dwell_time', 0) > 0:
                color = (0, 215, 255) # Vàng cam khi đang đỗ
                dur = rec['dwell_time']
                status_txt = f"ID {track_id} [DUNG DO] | {dur:.1f}s"
            elif rec and rec.get('in_roi'):
                color = (0, 255, 0) # Xanh lá đang chạy trong ROI
                status_txt = f"ID {track_id} [CHAY]"
            else:
                color = (200, 200, 200)
                status_txt = f"ID {track_id}"
                
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            (w, h), _ = cv2.getTextSize(status_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(frame, (x1, y1 - 25), (x1 + w, y1), color, -1)
            cv2.putText(frame, status_txt, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0) if color == (0, 215, 255) else (255, 255, 255), 2)

        # Hiển thị thông số tổng quát kèm tên model đang chạy
        cv2.putText(frame, f"Model: {model_info['filename']} | Time: {current_time:.1f}s | Violations: {len(violations)}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 165, 255), 2)

        # Hiển thị kết quả (scale nhỏ lại cho vừa màn hình nếu ảnh to)
        display_frame = cv2.resize(frame, (1280, 720))
        cv2.imshow("Parking Violation Monitor", display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
