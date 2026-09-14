import cv2
import yaml
import os
import sys
import csv
import time
import numpy as np
from datetime import datetime
from PIL import Image
import streamlit as st

# Thư viện tương tác vẽ ROI trên trình duyệt
try:
    from streamlit_image_coordinates import streamlit_image_coordinates
except ImportError:
    streamlit_image_coordinates = None

try:
    from streamlit_drawable_canvas import st_canvas
except ImportError:
    st_canvas = None

# Thêm đường dẫn src vào sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from ultralytics import YOLO
from rule_engine import ParkingRuleEngine
from model_utils import load_yolo_model, get_target_classes, get_model_info, discover_available_weights, DEFAULT_BASELINE_WEIGHTS

# ---- Cấu hình Trang Streamlit ----
st.set_page_config(
    page_title="Hệ Thống Giám Sát Dừng Đỗ Xe Thông Minh",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS cho giao diện hiện đại, chuyên nghiệp
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1E88E5, #00E676);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #9E9E9E;
        margin-bottom: 1.5rem;
    }
    .metric-box {
        background-color: #1E1E1E;
        border-radius: 8px;
        padding: 12px;
        border-left: 4px solid #1E88E5;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #FFFFFF;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #B0BEC5;
    }
    .status-badge-running {
        background-color: #2E7D32;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .status-badge-paused {
        background-color: #F57F17;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .status-badge-stopped {
        background-color: #C62828;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .roi-badge {
        display: inline-block;
        background: #263238;
        color: #00E676;
        border: 1px solid #37474F;
        padding: 4px 10px;
        border-radius: 6px;
        margin: 3px 4px;
        font-family: monospace;
        font-size: 0.88rem;
        font-weight: 600;
    }
    .roi-tip {
        background-color: #1A237E22;
        border-left: 4px solid #1E88E5;
        padding: 10px 14px;
        border-radius: 6px;
        margin-bottom: 12px;
        font-size: 0.92rem;
    }
</style>
""", unsafe_allow_html=True)

# ---- Đường Dẫn File & Thư Mục ----
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CONFIG_PATH = os.path.join(ROOT_DIR, 'config', 'settings.yaml')
DATA_DIR = os.path.join(ROOT_DIR, 'data')
EVIDENCE_DIR = os.path.join(ROOT_DIR, 'evidence')
os.makedirs(EVIDENCE_DIR, exist_ok=True)
EVIDENCE_LOG_PATH = os.path.join(EVIDENCE_DIR, 'violation_log.csv')

# Khởi tạo CSV nếu chưa tồn tại
if not os.path.exists(EVIDENCE_LOG_PATH):
    with open(EVIDENCE_LOG_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Timestamp', 'Track_ID', 'Dwell_Time_Seconds', 'Image_Path'])

# ---- Quản Lý Cấu Hình (Load & Save) ----
def load_config():
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def save_config(cfg):
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)

config = load_config()

# ---- Tải Model (Cache) ----
@st.cache_resource
def get_cached_model(weights_path, device):
    m, loaded_p, is_fb = load_yolo_model(
        weights_path=weights_path,
        device=device,
        fallback_path=config['model'].get('baseline_weights', DEFAULT_BASELINE_WEIGHTS),
        root_dir=ROOT_DIR
    )
    return m, loaded_p, is_fb

# ---- Khởi Tạo Session State ----
if 'is_playing' not in st.session_state:
    st.session_state.is_playing = False
if 'current_frame' not in st.session_state:
    st.session_state.current_frame = 0
if 'rule_engine' not in st.session_state:
    st.session_state.rule_engine = ParkingRuleEngine(config)
if 'last_processed_time' not in st.session_state:
    st.session_state.last_processed_time = 0.0
if 'new_violations_alert' not in st.session_state:
    st.session_state.new_violations_alert = []
if 'roi_points' not in st.session_state:
    st.session_state.roi_points = [list(map(int, p)) for p in config['rules'].get('roi', [])]
if 'last_click_key' not in st.session_state:
    st.session_state.last_click_key = None
if 'roi_selected_frame_idx' not in st.session_state:
    st.session_state.roi_selected_frame_idx = 0
if 'selected_weights' not in st.session_state:
    st.session_state.selected_weights = config['model'].get('weights', 'output_runs/best.pt')

# Quét tất cả video có sẵn trong thư mục data/
available_videos = []
if os.path.exists(DATA_DIR):
    for f in os.listdir(DATA_DIR):
        if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            available_videos.append(os.path.join('data', f).replace('\\', '/'))

if not available_videos:
    available_videos = ["data/sample.mp4"]

# ---- Sidebar Điều Khiển Nhanh ----
with st.sidebar:
    st.image("https://img.icons8.com/color/96/traffic-jam.png", width=64)
    st.markdown("### 🎛️ Bảng Điều Khiển")
    
    # 1. Chọn nguồn Video
    selected_video = st.selectbox(
        "📹 Chọn Video Giám Sát",
        options=available_videos,
        index=0 if config['video']['source'] not in available_videos else available_videos.index(config['video']['source'])
    )
    
    # Upload video mới
    uploaded_file = st.file_uploader("Hoặc tải video mới lên:", type=['mp4', 'avi', 'mov'])
    if uploaded_file is not None:
        save_uploaded_path = os.path.join(DATA_DIR, uploaded_file.name)
        with open(save_uploaded_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Đã lưu: {uploaded_file.name}")
        if save_uploaded_path.replace('\\', '/') not in available_videos:
            st.rerun()

    st.markdown("---")
    # 2. Lựa Chọn & Quản Lý Mô Hình Nhận Diện (Weights)
    st.markdown("### 🤖 Mô Hình Nhận Diện (Weights)")
    
    available_models = discover_available_weights(ROOT_DIR)
    model_paths = [m[1] for m in available_models]
    model_labels = {m[1]: m[0] for m in available_models}
    
    curr_w = st.session_state.selected_weights
    if curr_w not in model_paths and model_paths:
        model_paths.insert(0, curr_w)
        model_labels[curr_w] = f"⚙️ Mô hình tùy chọn ({os.path.basename(curr_w)})"
        
    def_idx = model_paths.index(curr_w) if curr_w in model_paths else 0
    
    chosen_w = st.selectbox(
        "Trọng số YOLO đang chạy:",
        options=model_paths,
        format_func=lambda x: model_labels.get(x, x),
        index=def_idx,
        help="Chuyển đổi ngay giữa mô hình custom vừa huấn luyện và mô hình gốc baseline"
    )
    
    if chosen_w != st.session_state.selected_weights:
        st.session_state.selected_weights = chosen_w
        st.rerun()

    # Nạp thông tin mô hình để xem trước thông số
    preview_model, preview_path, is_fb = get_cached_model(chosen_w, config['model'].get('device', 'cuda:0'))
    m_info = get_model_info(preview_model, preview_path)
    preview_cls = get_target_classes(preview_model, config['model'].get('classes', None))
    
    st.markdown(f"""
    <div style="background-color: #212121; padding: 10px; border-radius: 6px; border-left: 3px solid #00E676; margin-bottom: 10px; font-size: 0.85rem;">
        <b>Tên file:</b> <code>{m_info['filename']}</code> ({m_info['size_mb']} MB)<br>
        <b>Lớp nhận diện:</b> <code>{preview_cls}</code> ({m_info['num_classes']} classes)<br>
        <span style="color: #9E9E9E;">{m_info['description']}</span>
    </div>
    """, unsafe_allow_html=True)
    
    col_save_w, col_rollback_w = st.columns([1.2, 1.2])
    with col_save_w:
        if st.button("💾 Đặt Mặc Định", use_container_width=True, help="Lưu mô hình này vào file settings.yaml"):
            config['model']['weights'] = chosen_w
            config['model']['classes'] = preview_cls
            save_config(config)
            st.toast("✅ Đã lưu mô hình làm mặc định!", icon="💾")
            
    with col_rollback_w:
        baseline_w = config['model'].get('baseline_weights', DEFAULT_BASELINE_WEIGHTS)
        if st.button("↩️ Khôi Phục Gốc", use_container_width=True, help=f"ĐƯỜNG LUI: Lập tức quay về mô hình gốc {baseline_w}"):
            config['model']['weights'] = baseline_w
            config['model']['classes'] = [2, 3, 5, 7]
            save_config(config)
            st.session_state.selected_weights = baseline_w
            st.toast("🛡️ Đã khôi phục mô hình gốc Baseline!", icon="↩️")
            st.rerun()

    st.markdown("---")
    st.markdown("### ⏱️ Quy Tắc Dừng Đỗ")
    new_time_thresh = st.slider(
        "Ngưỡng thời gian đỗ tối đa (giây)", 
        min_value=2.0, max_value=60.0, 
        value=float(config['rules'].get('time_threshold', 5.0)),
        step=0.5
    )
    new_dist_thresh = st.slider(
        "Dung sai rung lắc (pixel)", 
        min_value=10.0, max_value=80.0, 
        value=float(config['rules'].get('movement_tolerance', 30.0)),
        step=5.0
    )
    new_conf_thresh = st.slider(
        "Ngưỡng tin cậy phát hiện (Conf)",
        min_value=0.1, max_value=0.9,
        value=float(config['model'].get('conf_threshold', 0.3)),
        step=0.05
    )

    if st.button("💾 Lưu Cài Đặt Vào File", use_container_width=True):
        config['rules']['time_threshold'] = float(new_time_thresh)
        config['rules']['movement_tolerance'] = float(new_dist_thresh)
        config['model']['conf_threshold'] = float(new_conf_thresh)
        config['video']['source'] = selected_video
        save_config(config)
        st.session_state.rule_engine = ParkingRuleEngine(config)
        st.success("Đã cập nhật cấu hình thành công!")

# ---- Giao Diện Chính Với 3 Tabs ----
st.markdown('<div class="main-header">Hệ Thống Giám Sát Dừng Đỗ Xe Thông Minh</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Phát hiện phương tiện dừng đỗ trái phép trong vùng cấm bằng AI (YOLO + ByteTrack + Rule Engine)</div>', unsafe_allow_html=True)

tab_live, tab_roi, tab_evidence, tab_help = st.tabs([
    "🎬 Giám Sát & Điều Khiển Video", 
    "🖌️ Vẽ Vùng Cấm Đỗ (ROI)", 
    "📸 Bằng Chứng Vi Phạm",
    "ℹ️ Hướng Dẫn"
])

# ==============================================================================
# TAB 1: GIÁM SÁT TRỰC TIẾP & BỘ ĐIỀU KHIỂN VIDEO (PLAY / PAUSE / RESUME / STOP)
# ==============================================================================
with tab_live:
    full_video_path = os.path.join(ROOT_DIR, selected_video)
    
    # Thanh điều khiển Video (Player Controls)
    col_ctrl1, col_ctrl2, col_ctrl3, col_ctrl4, col_ctrl5, col_status = st.columns([1.2, 1.2, 1.2, 1.2, 1.2, 2])
    
    with col_ctrl1:
        if st.button("▶️ Phát / Tiếp tục", type="primary", use_container_width=True):
            st.session_state.is_playing = True
            st.rerun()

    with col_ctrl2:
        if st.button("⏸️ Tạm Dừng", use_container_width=True):
            st.session_state.is_playing = False
            st.rerun()

    with col_ctrl3:
        if st.button("⏹️ Dừng Lại", use_container_width=True):
            st.session_state.is_playing = False
            st.session_state.current_frame = 0
            st.session_state.rule_engine = ParkingRuleEngine(config)
            st.rerun()

    with col_ctrl4:
        if st.button("⏮️ Xem Từ Đầu", use_container_width=True):
            st.session_state.current_frame = 0
            st.session_state.rule_engine = ParkingRuleEngine(config)
            st.session_state.is_playing = True
            st.rerun()

    with col_ctrl5:
        step_clicked = st.button("⏭️ +1 Frame", use_container_width=True)

    with col_status:
        if st.session_state.is_playing:
            st.markdown('<span class="status-badge-running">● ĐANG PHÁT VIDEO</span>', unsafe_allow_html=True)
        elif st.session_state.current_frame > 0:
            st.markdown('<span class="status-badge-paused">❚❚ ĐANG TẠM DỪNG</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-badge-stopped">■ ĐÃ DỪNG</span>', unsafe_allow_html=True)

    # Khởi tạo mô hình (Hỗ trợ chuyển đổi nhanh và đường lui)
    active_weights = st.session_state.get('selected_weights', config['model'].get('weights', 'output_runs/best.pt'))
    model, loaded_path, is_fallback = get_cached_model(active_weights, config['model'].get('device', 'cuda:0'))
    target_classes = get_target_classes(model, config['model'].get('classes', None))
    model_info = get_model_info(model, loaded_path)

    # Hiển thị Banner trạng thái mô hình
    fallback_badge = " <span style='color: #FFC107; font-weight: bold;'>[FALLBACK ACTIVE]</span>" if is_fallback else ""
    st.markdown(f"""
    <div style="background: linear-gradient(90deg, #1E1E1E, #263238); border-left: 4px solid #00E676; padding: 6px 14px; border-radius: 6px; margin-bottom: 12px; font-size: 0.88rem;">
        🤖 <b>Mô hình AI:</b> <code style="color: #00E676;">{model_info['filename']}</code> ({model_info['type']}) | 
        <b>Lớp nhận diện:</b> <code>{target_classes}</code> | 
        <b>Dung lượng:</b> <code>{model_info['size_mb']} MB</code>{fallback_badge}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Bố cục màn hình: Video bên trái (75%), Thông số thời gian thực bên phải (25%)
    col_video, col_kpi = st.columns([3, 1])

    with col_video:
        video_placeholder = st.empty()
        progress_placeholder = st.empty()

    with col_kpi:
        st.markdown("#### 📊 Thống Kê Thời Gian Thực")
        kpi_time = st.empty()
        kpi_cars = st.empty()
        kpi_in_roi = st.empty()
        kpi_violations = st.empty()
        
        st.markdown("#### 🚨 Cảnh Báo Mới")
        alert_placeholder = st.empty()

    # Logic xử lý phát video hoặc nhảy 1 frame
    if os.path.exists(full_video_path):
        cap = cv2.VideoCapture(full_video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 960
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 540

        # Nếu người dùng bấm +1 Frame khi đang dừng
        if step_clicked and not st.session_state.is_playing:
            cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.current_frame)
            ret, frame = cap.read()
            if ret:
                st.session_state.current_frame += 1
                cur_t = st.session_state.current_frame / fps
                
                # Tracking với target_classes tự động phân giải
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
                boxes_info = []
                if results[0].boxes is not None and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    ids = results[0].boxes.id.cpu().numpy().astype(int)
                    for b, tid in zip(boxes, ids):
                        cx = (b[0] + b[2]) / 2
                        cy = (b[1] + b[3]) / 2
                        tracks.append((tid, cx, cy, b[0], b[1], b[2], b[3]))
                        boxes_info.append((tid, b[0], b[1], b[2], b[3]))

                violations = st.session_state.rule_engine.update(tracks, cur_t)
                violation_ids = [v['track_id'] for v in violations]

                # Vẽ ROI
                cv2.polylines(frame, [st.session_state.rule_engine.roi_polygon], True, (0, 0, 255), 3)
                overlay = frame.copy()
                cv2.fillPoly(overlay, [st.session_state.rule_engine.roi_polygon], (0, 0, 255))
                cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

                # Vẽ Bboxes
                for tid, bx1, by1, bx2, by2 in boxes_info:
                    x1, y1, x2, y2 = map(int, [bx1, by1, bx2, by2])
                    rec = st.session_state.rule_engine.history.get(tid)

                    if tid in violation_ids:
                        color = (0, 0, 255) # Đỏ vi phạm
                        dur = rec.get('dwell_time', 0.0) if rec else 0.0
                        status_txt = f"VIOLATION ID {tid} | {dur:.1f}s"
                    elif rec and rec.get('is_stopped') and rec.get('dwell_time', 0) > 0:
                        color = (0, 215, 255) # Vàng cam khi xe dừng đỗ
                        dur = rec['dwell_time']
                        status_txt = f"ID {tid} [DUNG DO] | {dur:.1f}s"
                    elif rec and rec.get('in_roi'):
                        color = (0, 255, 0) # Xanh lá: xe đang chạy qua ROI
                        status_txt = f"ID {tid} [CHAY]"
                    else:
                        color = (200, 200, 200) # Trắng xám: ngoài ROI
                        status_txt = f"ID {tid}"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    (tw, th), _ = cv2.getTextSize(status_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                    cv2.rectangle(frame, (x1, max(0, y1 - 22)), (x1 + tw + 6, max(0, y1)), color, -1)
                    cv2.putText(frame, status_txt, (x1 + 3, max(16, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0) if color == (0, 215, 255) else (255, 255, 255), 2)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, use_container_width=True)
                progress_placeholder.progress(min(1.0, st.session_state.current_frame / total_frames))
                kpi_time.metric("Thời Gian Video", f"{cur_t:.1f}s / {total_frames/fps:.1f}s", f"Frame {st.session_state.current_frame}/{total_frames}")
                kpi_cars.metric("Tổng Số Xe", len(tracks))
                active_in_roi = sum(1 for r in st.session_state.rule_engine.history.values() if r.get('in_roi'))
                active_stopped = sum(1 for r in st.session_state.rule_engine.history.values() if r.get('is_stopped'))
                kpi_in_roi.metric("Xe Trong Vùng ROI", f"{active_in_roi} (Đỗ: {active_stopped})")
                kpi_violations.metric("Xe Vi Phạm", len(violations))

        # Vòng lặp phát Video khi is_playing = True
        elif st.session_state.is_playing:
            cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.current_frame)
            
            while cap.isOpened() and st.session_state.is_playing:
                ret, frame = cap.read()
                if not ret:
                    st.session_state.is_playing = False
                    st.session_state.current_frame = 0
                    st.info("Đã phát hết video.")
                    break

                st.session_state.current_frame += 1
                cur_time = st.session_state.current_frame / fps

                # Inference YOLO Tracking với target_classes tự động phân giải
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
                boxes_info = []
                if results[0].boxes is not None and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    ids = results[0].boxes.id.cpu().numpy().astype(int)
                    for b, tid in zip(boxes, ids):
                        cx = (b[0] + b[2]) / 2
                        cy = (b[1] + b[3]) / 2
                        tracks.append((tid, cx, cy, b[0], b[1], b[2], b[3]))
                        boxes_info.append((tid, b[0], b[1], b[2], b[3]))

                # Cập nhật Rule Engine
                violations = st.session_state.rule_engine.update(tracks, cur_time)
                violation_ids = [v['track_id'] for v in violations]

                # Lưu ảnh vi phạm mới
                for v in violations:
                    if v.get('is_new'):
                        tid = v['track_id']
                        dur = v['duration']
                        t_str = datetime.now().strftime('%Y%m%d_%H%M%S')
                        img_name = f"violation_{tid}_{t_str}.jpg"
                        img_path = os.path.join(EVIDENCE_DIR, img_name)

                        # Lưu ảnh kèm watermark
                        ev_img = frame.copy()
                        cv2.polylines(ev_img, [st.session_state.rule_engine.roi_polygon], True, (0, 0, 255), 3)
                        cv2.putText(ev_img, f"VIOLATION - ID: {tid} - DWELL: {dur:.1f}s", (30, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                        cv2.imwrite(img_path, ev_img)

                        with open(EVIDENCE_LOG_PATH, 'a', newline='', encoding='utf-8') as f:
                            csv.writer(f).writerow([t_str, tid, round(dur, 1), img_name])
                        
                        alert_placeholder.error(f"🚨 Phát hiện xe ID {tid} đỗ quá {dur:.1f}s!")

                # Vẽ Vùng ROI (Viền đỏ + mờ bên trong)
                cv2.polylines(frame, [st.session_state.rule_engine.roi_polygon], True, (0, 0, 255), 3)
                overlay = frame.copy()
                cv2.fillPoly(overlay, [st.session_state.rule_engine.roi_polygon], (0, 0, 255))
                cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

                # Nhãn Vùng Cấm Đỗ
                if len(st.session_state.rule_engine.roi_polygon) > 0:
                    roi_pt = st.session_state.rule_engine.roi_polygon[0]
                    cv2.putText(frame, "VUNG CAM DUNG DO", (roi_pt[0] + 5, roi_pt[1] + 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Vẽ Bounding Box & Thời Gian Đỗ
                for tid, bx1, by1, bx2, by2 in boxes_info:
                    x1, y1, x2, y2 = map(int, [bx1, by1, bx2, by2])
                    rec = st.session_state.rule_engine.history.get(tid)
                    
                    if tid in violation_ids:
                        color = (0, 0, 255) # Đỏ vi phạm
                        dur = rec.get('dwell_time', 0.0) if rec else 0.0
                        status_txt = f"VIOLATION ID {tid} | {dur:.1f}s"
                    elif rec and rec.get('is_stopped') and rec.get('dwell_time', 0) > 0:
                        color = (0, 215, 255) # Vàng cam khi đang đỗ trong ROI
                        dur = rec['dwell_time']
                        status_txt = f"ID {tid} [DUNG DO] | {dur:.1f}s"
                    elif rec and rec.get('in_roi'):
                        color = (0, 255, 0) # Xanh lá cho xe đang chạy qua ROI
                        status_txt = f"ID {tid} [CHAY]"
                    else:
                        color = (200, 200, 200) # Trắng xám: xe ngoài ROI
                        status_txt = f"ID {tid}"

                    # Vẽ hộp & nền chữ
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    (tw, th), _ = cv2.getTextSize(status_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                    cv2.rectangle(frame, (x1, max(0, y1 - 22)), (x1 + tw + 6, max(0, y1)), color, -1)
                    cv2.putText(frame, status_txt, (x1 + 3, max(16, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0) if color == (0, 215, 255) else (255, 255, 255), 2)

                # Cập nhật giao diện
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, use_container_width=True)
                progress_placeholder.progress(min(1.0, st.session_state.current_frame / total_frames))

                kpi_time.metric("Thời Gian Video", f"{cur_time:.1f}s / {total_frames/fps:.1f}s", f"Frame {st.session_state.current_frame}/{total_frames}")
                kpi_cars.metric("Tổng Số Xe", len(tracks))
                active_in_roi = sum(1 for r in st.session_state.rule_engine.history.values() if r.get('in_roi'))
                active_stopped = sum(1 for r in st.session_state.rule_engine.history.values() if r.get('is_stopped'))
                kpi_in_roi.metric("Xe Trong Vùng ROI", f"{active_in_roi} (Đỗ: {active_stopped})")
                kpi_violations.metric("Xe Vi Phạm", len(violations))

            cap.release()
        
        else:
            # Hiển thị frame tĩnh khi đang dừng
            cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.current_frame)
            ret, frame = cap.read()
            if ret:
                # Vẽ ROI xem trước
                cv2.polylines(frame, [st.session_state.rule_engine.roi_polygon], True, (0, 0, 255), 3)
                overlay = frame.copy()
                cv2.fillPoly(overlay, [st.session_state.rule_engine.roi_polygon], (0, 0, 255))
                cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, use_container_width=True)
                cur_t = st.session_state.current_frame / fps
                kpi_time.metric("Thời Gian Video", f"{cur_t:.1f}s / {total_frames/fps:.1f}s", f"Frame {st.session_state.current_frame}/{total_frames}")
                kpi_cars.metric("Tổng Số Xe", "-")
                kpi_in_roi.metric("Xe Trong Vùng Cấm", "-")
                kpi_violations.metric("Xe Vi Phạm", "-")
            cap.release()
    else:
        st.error(f"Không tìm thấy file video: {full_video_path}")


# ==============================================================================
# TAB 2: VẼ VÀ BIÊN TẬP VÙNG CẤM ĐỖ (ROI TOOL) TRỰC TIẾP TRÊN WEB
# ==============================================================================
with tab_roi:
    st.markdown("### 🖌️ Công Cụ Thiết Kế & Vẽ Vùng Cấm Đỗ (ROI) Trực Tiếp Trên Trình Duyệt")
    st.markdown("Chấm điểm trực tiếp trên ảnh video bằng chuột hoặc kéo vẽ đa giác/hình chữ nhật. Vùng cấm đỗ sẽ được cập nhật thời gian thực vào mô hình giám sát.")

    roi_video_path = os.path.join(ROOT_DIR, selected_video)
    
    if not os.path.exists(roi_video_path):
        st.error(f"Không tìm thấy video: {roi_video_path}")
    else:
        # Lấy thông tin video và tổng số frame
        cap_info = cv2.VideoCapture(roi_video_path)
        roi_total_frames = int(cap_info.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        roi_fps = cap_info.get(cv2.CAP_PROP_FPS) or 25.0
        cap_info.release()

        # 1. THANH CHỌN FRAME VIDEO MẪU
        with st.expander("🎞️ **1. Chọn Khung Hình Video Mẫu Để Chấm Điểm / Vẽ** (Nhấn để mở/thu gọn)", expanded=True):
            col_f_slide, col_f_live, col_f_first = st.columns([3, 1.2, 1.2])
            with col_f_slide:
                chosen_frame_idx = st.slider(
                    "Trượt chọn frame rõ nét nhất (không bị xe che khuất vạch đường):",
                    min_value=0,
                    max_value=max(0, roi_total_frames - 1),
                    value=min(st.session_state.roi_selected_frame_idx, max(0, roi_total_frames - 1)),
                    step=1
                )
                st.session_state.roi_selected_frame_idx = chosen_frame_idx
            with col_f_live:
                st.write("")
                st.write("")
                if st.button("📸 Lấy Frame Tab Giám Sát", use_container_width=True, help="Lấy đúng frame mà bạn vừa tạm dừng ở Tab 1"):
                    st.session_state.roi_selected_frame_idx = min(st.session_state.current_frame, max(0, roi_total_frames - 1))
                    st.rerun()
            with col_f_first:
                st.write("")
                st.write("")
                if st.button("⏮️ Frame Đầu Tiên (0)", use_container_width=True):
                    st.session_state.roi_selected_frame_idx = 0
                    st.rerun()

        # Đọc frame đã chọn
        cap_roi = cv2.VideoCapture(roi_video_path)
        cap_roi.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.roi_selected_frame_idx)
        ret_roi, raw_frame = cap_roi.read()
        cap_roi.release()

        if not ret_roi or raw_frame is None:
            st.error("Không thể đọc frame từ video.")
        else:
            img_h, img_w = raw_frame.shape[:2]

            # Bố cục 2 cột: Cột trái (Vẽ trực quan 68%), Cột phải (Bảng điều khiển & Tọa độ 32%)
            col_draw_area, col_ctrl_area = st.columns([2.1, 1])

            with col_draw_area:
                # Lựa chọn chế độ vẽ
                mode_options = ["🖱️ Chấm Điểm Trực Tiếp (Click-to-Point)"]
                if st_canvas is not None:
                    mode_options.append("🎨 Kéo Vẽ Canvas (Fabric.js)")
                mode_options.extend(["📐 Mẫu Sẵn Có (Presets)", "🔢 Nhập Số Tọa Độ (Manual)"])

                selected_mode = st.radio(
                    "**Chọn phương thức thiết lập vùng cấm:**",
                    options=mode_options,
                    horizontal=True
                )

                # ==============================================================
                # CHẾ ĐỘ 1: CHẤM ĐIỂM TRỰC TIẾP TRÊN ẢNH (CLICK-TO-POINT)
                # ==============================================================
                if selected_mode.startswith("🖱️"):
                    st.markdown("""
                    <div class="roi-tip">
                        💡 <b>Hướng dẫn</b>: Click chuột trái trực tiếp lên ảnh bên dưới để thêm các đỉnh của vùng cấm (P1, P2, P3, P4...).
                        Đa giác viền đỏ dạ quang và lớp phủ cảnh báo sẽ tự động nối lại tức thì!
                    </div>
                    """, unsafe_allow_html=True)

                    # Tạo ảnh hiển thị với các điểm và đa giác hiện có
                    disp_img = raw_frame.copy()
                    current_pts = st.session_state.roi_points

                    # Vẽ đa giác và các đỉnh nếu có
                    if len(current_pts) > 0:
                        pts_arr = np.array(current_pts, np.int32)
                        
                        # Tô màu phủ nếu >= 3 điểm
                        if len(current_pts) >= 3:
                            overlay = disp_img.copy()
                            cv2.fillPoly(overlay, [pts_arr], (0, 0, 255))
                            cv2.addWeighted(overlay, 0.28, disp_img, 0.72, 0, disp_img)
                            cv2.polylines(disp_img, [pts_arr], True, (0, 0, 255), 3)
                            
                            # Nhãn vùng cấm
                            p0 = current_pts[0]
                            cv2.putText(disp_img, "VUNG CAM DUNG DO (ROI)", (max(10, p0[0]), max(25, p0[1] - 12)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
                        elif len(current_pts) == 2:
                            # Nối 2 điểm đầu
                            cv2.line(disp_img, tuple(current_pts[0]), tuple(current_pts[1]), (0, 255, 255), 2)

                        # Vẽ từng đỉnh
                        for idx, (px, py) in enumerate(current_pts):
                            # Vòng ngoài xanh neon, tâm đỏ
                            cv2.circle(disp_img, (px, py), 8, (0, 255, 0), -1)
                            cv2.circle(disp_img, (px, py), 10, (255, 255, 255), 2)
                            label_txt = f"P{idx+1}"
                            cv2.rectangle(disp_img, (px + 10, py - 18), (px + 45, py + 2), (0, 0, 0), -1)
                            cv2.putText(disp_img, label_txt, (px + 12, py - 4),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                    disp_rgb = cv2.cvtColor(disp_img, cv2.COLOR_BGR2RGB)
                    disp_pil = Image.fromarray(disp_rgb)

                    # Bắt sự kiện click chuột trực tiếp trên trình duyệt
                    if streamlit_image_coordinates is not None:
                        click_coords = streamlit_image_coordinates(
                            disp_pil,
                            key=f"roi_click_comp_{st.session_state.roi_selected_frame_idx}_{len(current_pts)}",
                            cursor="crosshair"
                        )

                        if click_coords is not None:
                            click_id = click_coords.get('unix_time') or (click_coords.get('x'), click_coords.get('y'))
                            if click_id != st.session_state.get('last_click_key'):
                                st.session_state.last_click_key = click_id
                                dw = click_coords.get('width', img_w)
                                dh = click_coords.get('height', img_h)
                                if dw and dh and dw > 0 and dh > 0:
                                    sx = img_w / float(dw)
                                    sy = img_h / float(dh)
                                    rx = int(round(click_coords['x'] * sx))
                                    ry = int(round(click_coords['y'] * sy))
                                    rx = max(0, min(img_w - 1, rx))
                                    ry = max(0, min(img_h - 1, ry))
                                    st.session_state.roi_points.append([rx, ry])
                                    st.rerun()
                    else:
                        st.image(disp_pil, use_container_width=True)
                        st.warning("Gói `streamlit-image-coordinates` chưa khả dụng. Hãy dùng chế độ Nhập Tọa Độ hoặc Canvas.")

                # ==============================================================
                # CHẾ ĐỘ 2: KÉO VẼ CANVAS (FABRIC.JS)
                # ==============================================================
                elif selected_mode.startswith("🎨") and st_canvas is not None:
                    st.markdown("""
                    <div class="roi-tip">
                        💡 <b>Hướng dẫn</b>: Chọn công cụ Đa Giác (Click nối đỉnh và click đúp để đóng) hoặc Hình Chữ Nhật (kéo thả chuột).
                        Sau khi vẽ xong, bấm <b>Nạp Tọa Độ Từ Canvas</b> bên dưới.
                    </div>
                    """, unsafe_allow_html=True)

                    c_tool = st.radio(
                        "Công cụ vẽ:",
                        ["polygon", "rect", "transform"],
                        format_func=lambda x: {"polygon": "📐 Vẽ Đa Giác", "rect": "⬛ Kéo Hình Chữ Nhật", "transform": "✋ Di chuyển/Sửa"}[x],
                        horizontal=True
                    )

                    canvas_w = min(800, img_w)
                    canvas_h = int(round(canvas_w * img_h / img_w))
                    raw_rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
                    bg_img = Image.fromarray(raw_rgb).resize((canvas_w, canvas_h))

                    canvas_res = st_canvas(
                        fill_color="rgba(255, 0, 0, 0.25)",
                        stroke_width=3,
                        stroke_color="#FF0000",
                        background_image=bg_img,
                        update_streamlit=True,
                        height=canvas_h,
                        width=canvas_w,
                        drawing_mode=c_tool,
                        key=f"canvas_draw_{c_tool}"
                    )

                    if st.button("📥 Nạp Tọa Độ Từ Canvas Vào Bộ Nhớ", use_container_width=True):
                        if canvas_res.json_data is not None and "objects" in canvas_res.json_data:
                            objs = canvas_res.json_data["objects"]
                            if len(objs) > 0:
                                last_obj = objs[-1]
                                scale_x = img_w / float(canvas_w)
                                scale_y = img_h / float(canvas_h)
                                new_pts = []

                                if last_obj.get('type') == 'rect':
                                    l = last_obj.get('left', 0) * scale_x
                                    t = last_obj.get('top', 0) * scale_y
                                    w = last_obj.get('width', 0) * last_obj.get('scaleX', 1) * scale_x
                                    h = last_obj.get('height', 0) * last_obj.get('scaleY', 1) * scale_y
                                    new_pts = [
                                        [int(l), int(t)],
                                        [int(l + w), int(t)],
                                        [int(l + w), int(t + h)],
                                        [int(l), int(t + h)]
                                    ]
                                elif last_obj.get('type') == 'path' and 'path' in last_obj:
                                    for cmd in last_obj['path']:
                                        if len(cmd) >= 3 and cmd[0] in ['M', 'L']:
                                            new_pts.append([int(round(cmd[1] * scale_x)), int(round(cmd[2] * scale_y))])
                                elif last_obj.get('type') == 'polygon' and 'points' in last_obj:
                                    for p in last_obj['points']:
                                        new_pts.append([int(round(p['x'] * scale_x)), int(round(p['y'] * scale_y))])

                                if len(new_pts) >= 3:
                                    st.session_state.roi_points = new_pts
                                    st.success(f"Đã trích xuất {len(new_pts)} đỉnh từ Canvas!")
                                    st.rerun()
                                else:
                                    st.warning("Vui lòng vẽ ít nhất 1 hình hoàn chỉnh trên Canvas trước khi nạp.")
                            else:
                                st.warning("Canvas chưa có hình vẽ nào.")

                # ==============================================================
                # CHẾ ĐỘ 3: MẪU CÓ SẴN (PRESETS)
                # ==============================================================
                elif selected_mode.startswith("📐"):
                    st.markdown("Chọn nhanh một trong các mẫu vùng cấm phổ biến cho camera giao thông:")
                    col_p1, col_p2 = st.columns(2)
                    with col_p1:
                        if st.button("🛣️ Cả Lòng Đường", use_container_width=True):
                            st.session_state.roi_points = [
                                [int(img_w * 0.05), int(img_h * 0.35)],
                                [int(img_w * 0.95), int(img_h * 0.35)],
                                [int(img_w * 0.98), int(img_h * 0.98)],
                                [int(img_w * 0.02), int(img_h * 0.98)]
                            ]
                            st.rerun()
                        if st.button("🛑 Lề Đường Bên Phải", use_container_width=True):
                            st.session_state.roi_points = [
                                [int(img_w * 0.55), int(img_h * 0.35)],
                                [int(img_w * 0.95), int(img_h * 0.35)],
                                [int(img_w * 0.98), int(img_h * 0.98)],
                                [int(img_w * 0.60), int(img_h * 0.98)]
                            ]
                            st.rerun()
                    with col_p2:
                        if st.button("🛑 Lề Đường Bên Trái", use_container_width=True):
                            st.session_state.roi_points = [
                                [int(img_w * 0.05), int(img_h * 0.35)],
                                [int(img_w * 0.45), int(img_h * 0.35)],
                                [int(img_w * 0.40), int(img_h * 0.98)],
                                [int(img_w * 0.02), int(img_h * 0.98)]
                            ]
                            st.rerun()
                        if st.button("🎯 Làn Giữa Đường", use_container_width=True):
                            st.session_state.roi_points = [
                                [int(img_w * 0.30), int(img_h * 0.35)],
                                [int(img_w * 0.70), int(img_h * 0.35)],
                                [int(img_w * 0.75), int(img_h * 0.98)],
                                [int(img_w * 0.25), int(img_h * 0.98)]
                            ]
                            st.rerun()

                # ==============================================================
                # CHẾ ĐỘ 4: NHẬP SỐ TỌA ĐỘ THỦ CÔNG
                # ==============================================================
                else:
                    st.markdown("Nhập danh sách tọa độ các đỉnh (mỗi dòng một cặp X, Y cách nhau bởi dấu phẩy):")
                    curr_str = "\n".join([f"{p[0]}, {p[1]}" for p in st.session_state.roi_points])
                    edited_str = st.text_area("Tọa độ đỉnh:", value=curr_str, height=140)
                    if st.button("Cập Nhật Danh Sách Điểm Từ Text", use_container_width=True):
                        try:
                            parsed = []
                            for line in edited_str.strip().split('\n'):
                                line = line.strip()
                                if line:
                                    parts = [int(float(v.strip())) for v in line.split(',')]
                                    if len(parts) == 2:
                                        parsed.append(parts)
                            if len(parsed) >= 3:
                                st.session_state.roi_points = parsed
                                st.success(f"Đã cập nhật {len(parsed)} điểm!")
                                st.rerun()
                            else:
                                st.error("Cần tối thiểu 3 đỉnh!")
                        except Exception as e:
                            st.error(f"Lỗi cú pháp: {e}")

                # NÚT ĐIỀU KHIỂN NHANH DƯỚI KHUNG VẼ
                st.markdown("---")
                col_btn_undo, col_btn_clear, col_btn_reload = st.columns([1.2, 1.2, 1.2])
                with col_btn_undo:
                    if st.button("↩️ Hoàn Tác (Xóa Điểm Cuối)", use_container_width=True, disabled=len(st.session_state.roi_points) == 0):
                        if st.session_state.roi_points:
                            st.session_state.roi_points.pop()
                            st.rerun()
                with col_btn_clear:
                    if st.button("🗑️ Xóa Tất Cả Điểm", use_container_width=True, disabled=len(st.session_state.roi_points) == 0):
                        st.session_state.roi_points = []
                        st.rerun()
                with col_btn_reload:
                    if st.button("🔄 Nạp Lại ROI Đang Dùng", use_container_width=True):
                        st.session_state.roi_points = [list(map(int, p)) for p in config['rules'].get('roi', [])]
                        st.rerun()

            # CỘT PHẢI: BẢNG TỌA ĐỘ & LƯU VÙNG ROI
            with col_ctrl_area:
                st.markdown("#### 📋 Thông Tin Vùng Cấm (ROI)")
                pts_now = st.session_state.roi_points
                n_pts = len(pts_now)

                if n_pts >= 3:
                    st.success(f"✅ Đa giác hợp lệ: **{n_pts} đỉnh**")
                elif n_pts > 0:
                    st.warning(f"⚠️ Đã có {n_pts} đỉnh (cần tối thiểu 3 đỉnh)")
                else:
                    st.info("ℹ️ Chưa có đỉnh nào. Hãy click lên ảnh để chấm điểm!")

                # Hiển thị tọa độ từng đỉnh dạng badge
                st.markdown("**Danh sách tọa độ các đỉnh:**")
                if n_pts > 0:
                    badge_html = ""
                    for i, pt in enumerate(pts_now):
                        badge_html += f'<span class="roi-badge">P{i+1}: ({pt[0]}, {pt[1]})</span>'
                    st.markdown(badge_html, unsafe_allow_html=True)
                else:
                    st.caption("Chưa có điểm nào.")

                st.markdown(f"**Độ phân giải video:** `{img_w} x {img_h}` px")
                st.markdown(f"**Khung hình tham chiếu:** Frame `{st.session_state.roi_selected_frame_idx}/{roi_total_frames}`")

                st.markdown("---")
                st.markdown("#### 💾 Lưu Cấu Hình Vào Hệ Thống")
                st.caption("Bấm nút bên dưới để ghi đè vùng ROI mới vào file cấu hình và cập nhật ngay cho mô hình AI ở Tab 1:")

                if st.button("💾 LƯU & ÁP DỤNG VÙNG CẤM", type="primary", use_container_width=True, disabled=n_pts < 3):
                    config['rules']['roi'] = [list(map(int, p)) for p in pts_now]
                    save_config(config)
                    st.session_state.rule_engine = ParkingRuleEngine(config)
                    st.success(f"🎉 Đã lưu thành công vùng cấm gồm {n_pts} đỉnh!")
                    st.toast("✅ Đã cập nhật Vùng Cấm Đỗ (ROI) thành công!", icon="🚗")
                    st.rerun()

                # Xem trước vùng ROI trên ảnh gốc
                if n_pts >= 3:
                    st.markdown("---")
                    st.markdown("#### 👁️ Xem Trước ROI Hoàn Chỉnh")
                    pv_img = raw_frame.copy()
                    pv_arr = np.array(pts_now, np.int32)
                    cv2.polylines(pv_img, [pv_arr], True, (0, 0, 255), 3)
                    pv_ov = pv_img.copy()
                    cv2.fillPoly(pv_ov, [pv_arr], (0, 0, 255))
                    cv2.addWeighted(pv_ov, 0.25, pv_img, 0.75, 0, pv_img)
                    for idx, (px, py) in enumerate(pts_now):
                        cv2.circle(pv_img, (px, py), 6, (0, 255, 0), -1)
                        cv2.putText(pv_img, f"P{idx+1}", (px + 6, py - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    pv_rgb = cv2.cvtColor(pv_img, cv2.COLOR_BGR2RGB)
                    st.image(pv_rgb, caption="Hình ảnh thực tế khi hệ thống giám sát", use_container_width=True)


# ==============================================================================
# TAB 3: BẰNG CHỨNG VI PHẠM & XUẤT BÁO CÁO
# ==============================================================================
with tab_evidence:
    st.markdown("### 📸 Nhật Ký & Thư Viện Bằng Chứng Vi Phạm")
    
    col_btn_export, col_btn_clear, _ = st.columns([1.5, 1.5, 4])
    
    # Đọc file CSV an toàn (tự động gán header nếu file không có header)
    if os.path.exists(EVIDENCE_LOG_PATH):
        import pandas as pd
        col_names = ['Timestamp', 'Track_ID', 'Dwell_Time_Seconds', 'Image_Path']
        try:
            # Kiểm tra dòng đầu có phải header không
            first_line = ""
            with open(EVIDENCE_LOG_PATH, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
            
            if 'Timestamp' in first_line:
                df = pd.read_csv(EVIDENCE_LOG_PATH)
            else:
                df = pd.read_csv(EVIDENCE_LOG_PATH, header=None, names=col_names)
        except Exception:
            df = pd.DataFrame(columns=col_names)
        
        with col_btn_export:
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Tải Báo Cáo CSV",
                data=csv_data,
                file_name=f"violation_report_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )

        with col_btn_clear:
            if st.button("🗑️ Xóa Lịch Sử Vi Phạm", use_container_width=True):
                with open(EVIDENCE_LOG_PATH, 'w', newline='', encoding='utf-8') as f:
                    csv.writer(f).writerow(['Timestamp', 'Track_ID', 'Dwell_Time_Seconds', 'Image_Path'])
                st.success("Đã xóa sạch lịch sử!")
                st.rerun()

        st.markdown(f"**Tổng số vi phạm đã ghi nhận: {len(df)} vụ**")
        st.dataframe(df, use_container_width=True)

        st.markdown("---")
        st.markdown("#### 🖼️ Hình Ảnh Bằng Chứng Chụp Tự Động")
        
        if len(df) > 0 and 'Image_Path' in df.columns:
            # Hiển thị tối đa 8 ảnh vi phạm gần nhất dạng gallery
            recent_evs = df.tail(8).iloc[::-1]
            cols = st.columns(4)
            for idx, (_, row) in enumerate(recent_evs.iterrows()):
                c = cols[idx % 4]
                img_f = os.path.join(EVIDENCE_DIR, str(row['Image_Path']))
                if os.path.exists(img_f):
                    im = Image.open(img_f)
                    tid_val = row.get('Track_ID', 'N/A')
                    dur_val = row.get('Dwell_Time_Seconds', 'N/A')
                    t_val = row.get('Timestamp', '')
                    c.image(im, caption=f"ID {tid_val} | Đỗ: {dur_val}s\n{t_val}", use_container_width=True)
                else:
                    c.warning(f"Ảnh không tồn tại: {row.get('Image_Path')}")
        else:
            st.info("Chưa có vi phạm nào được ghi nhận.")
    else:
        st.info("Chưa có nhật ký vi phạm nào.")


# ==============================================================================
# TAB 4: HƯỚNG DẪN SỬ DỤNG
# ==============================================================================
with tab_help:
    st.markdown("""
    ### 📖 Hướng Dẫn Sử Dụng Dashboard
    
    1. **Giám Sát & Điều Khiển Video**:
       - Bấm **`▶️ Phát / Tiếp tục`** để chạy mô hình AI phát hiện và đếm giây dừng đỗ theo thời gian thực.
       - Bấm **`⏸️ Tạm Dừng`** bất cứ lúc nào để dừng lại quan sát khung hình.
       - Khi tạm dừng, bạn có thể bấm **`⏭️ +1 Frame`** để xem chuyển động từng frame một.
       - Bấm **`⏹️ Dừng Lại`** để reset trạng thái về ban đầu.

    2. **Vẽ Vùng Cấm Đỗ (ROI) Trực Tiếp Trên Web**:
       - Chuyển sang Tab **`🖌️ Vẽ Vùng Cấm Đỗ (ROI)`**.
       - Chọn khung hình rõ nét nhất bằng thanh trượt (hoặc bấm nút **`📸 Lấy Frame Tab Giám Sát`** để lấy đúng frame vừa tạm dừng ở Tab 1).
       - **Chấm điểm trực tiếp**: Di chuột lên ảnh và click chuột trái lần lượt vào các góc của vùng cấm (ví dụ 4 góc lòng đường $P_1, P_2, P_3, P_4$). Đường viền dạ quang và diện tích vùng cấm sẽ hiển thị tức thì.
       - **Kéo vẽ Canvas**: Có thể chuyển sang chế độ Canvas để kéo hình chữ nhật hoặc nối đa giác tự do bằng chuột.
       - **Hoàn tác & Xóa lại**: Bấm **`↩️ Hoàn Tác`** để xóa đỉnh vừa click nhầm, hoặc **`🗑️ Xóa Tất Cả`** để vẽ lại từ đầu.
       - **Lưu cấu hình**: Bấm **`💾 LƯU & ÁP DỤNG VÙNG CẤM`**. Hệ thống sẽ tự động cập nhật file cấu hình và Tab Giám Sát sẽ áp dụng ngay tức thì mà không cần khởi động lại server.

    3. **Xuất Báo Cáo & Xem Ảnh Bằng Chứng**:
       - Chuyển sang Tab **`📸 Bằng Chứng Vi Phạm`** để xem bảng log, xem ảnh bằng chứng chụp tự động khi xe vượt ngưỡng thời gian và tải file CSV về máy.

    4. **Đổi Mô Hình & Đường Lui (Rollback)**:
       - Tại Sidebar bên trái, mục **`🤖 Mô Hình Nhận Diện (Weights)`** cho phép bạn tự do chọn giữa:
         + `🎯 Custom Model: output_runs/best.pt`: Mô hình chuyên biệt đã train trên UA-DETRAC, siêu nhẹ (18.3 MB), FPS cao, độ tin cậy phát hiện xe cao (0.80).
         + `🛡️ Baseline Model: yolo11m.pt`: Mô hình gốc ban đầu (COCO), bao quát nhiều phương tiện ở xa.
       - **Đường lui an toàn**: Bất cứ lúc nào cảm thấy mô hình custom nhận diện kém hơn, bạn chỉ cần bấm **`↩️ Khôi Phục Gốc`** ở Sidebar, hệ thống sẽ lập tức quay về `yolo11m.pt` với 1 cú click!
       - Bấm **`💾 Đặt Mặc Định`** để ghi nhớ mô hình ưa thích của bạn vào file cấu hình.
    """)
