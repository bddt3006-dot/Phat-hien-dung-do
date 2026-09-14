import cv2
import yaml
import os
import sys
import csv
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from ultralytics import YOLO
from src.rule_engine import ParkingRuleEngine
from src.model_utils import load_yolo_model, get_target_classes, get_model_info

def run_test():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    with open(os.path.join(root_dir, 'config', 'settings.yaml'), 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    weights_to_test = config['model'].get('weights', 'output_runs/best.pt')
    model, loaded_path, is_fallback = load_yolo_model(weights_to_test, root_dir=root_dir)
    target_classes = get_target_classes(model, config['model'].get('classes', None))
    m_info = get_model_info(model, loaded_path)

    engine = ParkingRuleEngine(config)
    video_p = os.path.join(root_dir, config['video']['source'])
    cap = cv2.VideoCapture(video_p)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    evidence_dir = os.path.join(root_dir, 'evidence')
    os.makedirs(evidence_dir, exist_ok=True)
    evidence_csv = os.path.join(evidence_dir, 'violation_log.csv')

    print(f"Running pipeline test with model: {m_info['filename']} ({m_info['type']})")
    print(f"Target classes: {target_classes} | Image size: {config['model']['imgsz']}")
    frame_count = 0
    total_violations_found = 0

    while frame_count < 200:
        ret, frame = cap.read()
        if not ret:
            break

        cur_time = frame_count / fps
        frame_count += 1

        results = model.track(
            frame,
            tracker=config['tracking']['tracker'],
            conf=config['model']['conf_threshold'],
            imgsz=config['model']['imgsz'],
            classes=target_classes,
            device=config['model'].get('device', 'cuda:0'),
            persist=True,
            verbose=False
        )

        tracks = []
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            for b, tid in zip(boxes, ids):
                cx = (b[0] + b[2]) / 2
                cy = (b[1] + b[3]) / 2
                tracks.append((tid, cx, cy, b[0], b[1], b[2], b[3]))

        violations = engine.update(tracks, cur_time)
        for v in violations:
            if v.get('is_new'):
                total_violations_found += 1
                tid = v['track_id']
                dur = v['duration']
                t_str = datetime.now().strftime('%Y%m%d_%H%M%S')
                img_name = f"test_violation_{tid}_{t_str}.jpg"
                img_path = os.path.join(evidence_dir, img_name)

                ev_frame = frame.copy()
                cv2.putText(ev_frame, f"VIOLATION ID: {tid} - DWELL: {dur:.1f}s", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                cv2.imwrite(img_path, ev_frame)

                with open(evidence_csv, 'a', newline='', encoding='utf-8') as f:
                    csv.writer(f).writerow([t_str, tid, round(dur, 1), img_name])

                print(f"-> [NEW VIOLATION] Frame {frame_count} | Vehicle ID: {tid} | Dwell Time: {dur:.1f}s | Saved: {img_name}")

    cap.release()
    print(f"\n[DONE] Processed {frame_count} frames. Captured {total_violations_found} violations.")

if __name__ == '__main__':
    run_test()
