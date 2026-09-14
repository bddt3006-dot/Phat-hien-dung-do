import os
from ultralytics import YOLO

DEFAULT_BASELINE_WEIGHTS = 'yolo11m.pt'
COCO_VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck

def resolve_model_path(weights_path, root_dir=None):
    """
    Tìm đường dẫn tuyệt đối chính xác của file trọng số.
    """
    if root_dir is None:
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        
    # Thử đường dẫn trực tiếp
    if os.path.isabs(weights_path) and os.path.exists(weights_path):
        return weights_path
        
    # Thử kết hợp với root_dir
    candidate = os.path.join(root_dir, weights_path)
    if os.path.exists(candidate):
        return candidate
        
    # Thử tìm trong output_runs
    if weights_path in ['output_runs', 'best.pt', 'output_runs/best.pt']:
        candidates = [
            os.path.join(root_dir, 'output_runs', 'best.pt'),
            os.path.join(root_dir, 'output_runs', 'detrac_train', 'weights', 'best.pt'),
            os.path.join(root_dir, 'runs', 'detect', 'train-3', 'weights', 'best.pt'),
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
                
    return os.path.join(root_dir, weights_path)

def load_yolo_model(weights_path, device='cuda:0', fallback_path=DEFAULT_BASELINE_WEIGHTS, root_dir=None):
    """
    Tải mô hình YOLO với cơ chế tự động fallback về baseline nếu file custom bị lỗi/thiếu.
    """
    if root_dir is None:
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        
    resolved_path = resolve_model_path(weights_path, root_dir)
    
    if os.path.exists(resolved_path):
        try:
            model = YOLO(resolved_path)
            return model, resolved_path, False  # (model, path, is_fallback)
        except Exception as e:
            print(f"[CẢNH BÁO] Không thể tải '{resolved_path}' ({e}). Đang fallback về mô hình gốc...")
            
    # Fallback về baseline
    fallback_resolved = resolve_model_path(fallback_path, root_dir)
    if not os.path.exists(fallback_resolved):
        fallback_resolved = fallback_path  # Ultralytics tự tải nếu cần
        
    print(f"[FALLBACK] Đang nạp mô hình gốc an toàn: '{fallback_resolved}'")
    model = YOLO(fallback_resolved)
    return model, fallback_resolved, True

def get_target_classes(model, configured_classes=None):
    """
    Smart Class Resolver:
    Tự động phân giải danh sách lớp mục tiêu dựa trên mô hình:
    - Nếu mô hình custom chỉ có 1 lớp (VD: DETRAC {0: 'vehicle'}): trả về [0]
    - Nếu mô hình COCO gốc (80 lớp): trả về [2, 3, 5, 7] (car, motorcycle, bus, truck)
    - Nếu người dùng truyền configured_classes và các ID đó hợp lệ với model.names: dùng configured_classes
    """
    if model is None or not hasattr(model, 'names') or model.names is None:
        return configured_classes

    names = model.names
    num_classes = len(names)

    # 1. Mô hình custom đơn lớp (như model huấn luyện trên UA-DETRAC)
    if num_classes == 1:
        return [0]

    # 2. Kiểm tra nếu cấu hình cũ [2, 3, 5, 7] có hợp lệ trong model này không
    if configured_classes is not None and isinstance(configured_classes, (list, tuple)):
        # Chỉ giữ những class id thực sự tồn tại trong model
        valid = [c for c in configured_classes if c in names]
        if len(valid) > 0:
            return valid

    # 3. Mô hình gốc COCO (80 lớp chuẩn)
    if num_classes >= 80:
        return COCO_VEHICLE_CLASSES

    # 4. Trường hợp khác: phát hiện tất cả các lớp của mô hình
    return list(names.keys())

def get_model_info(model, weights_path):
    """
    Trả về thông tin tóm tắt thân thiện về mô hình đang dùng.
    """
    size_mb = 0.0
    if os.path.exists(weights_path):
        size_mb = os.path.getsize(weights_path) / (1024 * 1024)
        
    num_classes = len(model.names) if hasattr(model, 'names') and model.names else 0
    
    if num_classes == 1:
        model_type = "Mô hình Huấn luyện Custom (UA-DETRAC - 1 Class)"
        desc = "Tối ưu chuyên biệt cho phát hiện xe cộ, tốc độ cao."
    elif num_classes >= 80:
        model_type = "Mô hình Gốc Baseline (YOLO11 COCO - 80 Classes)"
        desc = "Đa năng, nhận diện nhiều loại phương tiện theo chuẩn COCO."
    else:
        model_type = f"Mô hình Custom ({num_classes} Classes)"
        desc = f"Danh sách lớp: {list(model.names.values())[:3]}"

    return {
        'path': weights_path,
        'filename': os.path.basename(weights_path),
        'type': model_type,
        'description': desc,
        'size_mb': round(size_mb, 1),
        'num_classes': num_classes,
        'classes_dict': model.names if hasattr(model, 'names') else {}
    }

def discover_available_weights(root_dir=None):
    """
    Quét và liệt kê tất cả trọng số có thể lựa chọn trong thư mục dự án.
    Trả về danh sách các tuple: (display_label, relative_path, is_custom)
    """
    if root_dir is None:
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        
    models = []
    
    # 1. Mô hình custom từ output_runs (ưu tiên số 1)
    custom_candidates = [
        ('output_runs/best.pt', '🎯 Custom Model: output_runs/best.pt (Mới huấn luyện - Siêu nhanh)'),
        ('output_runs/detrac_train/weights/best.pt', '🎯 Custom Model: detrac_train/weights/best.pt'),
        ('runs/detect/train-3/weights/best.pt', '📦 Custom Model: runs/detect/train-3 (YOLO11m DETRAC)'),
    ]
    seen_paths = set()
    for rel_path, label in custom_candidates:
        full_p = os.path.join(root_dir, rel_path)
        if os.path.exists(full_p) and rel_path not in seen_paths:
            models.append((label, rel_path, True))
            seen_paths.add(rel_path)
            
    # 2. Mô hình gốc Baseline COCO (đường lui an toàn)
    baseline_candidates = [
        ('yolo11m.pt', '🛡️ Baseline Model: yolo11m.pt (Mô hình gốc ban đầu - Chuẩn COCO)'),
        ('yolo26n.pt', '⚡ Baseline Model: yolo26n.pt (Siêu nhẹ)'),
    ]
    for rel_path, label in baseline_candidates:
        full_p = os.path.join(root_dir, rel_path)
        if os.path.exists(full_p) and rel_path not in seen_paths:
            models.append((label, rel_path, False))
            seen_paths.add(rel_path)
            
    return models
