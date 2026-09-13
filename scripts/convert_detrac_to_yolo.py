"""
Convert UA-DETRAC XML annotations to YOLO format.

Quy tắc:
- Chia tập theo SEQUENCE (video-level), KHÔNG theo frame → tránh data leakage.
- Map tất cả vehicle_type (car, bus, van, others) → class 0 (vehicle).
- Bỏ qua các target nằm trong ignored_region (IoU > 50%).
- Tọa độ YOLO: class x_center y_center width height (chuẩn hóa 0-1).

Cách dùng:
    python scripts/convert_detrac_to_yolo.py
"""

import xml.etree.ElementTree as ET
import os
import shutil
import random
from pathlib import Path

# ============================================================
# CẤU HÌNH
# ============================================================
BASE_DIR = Path(r"c:\Users\ADMIN\Module dừng đỗ xe")
DATA_DIR = BASE_DIR / "Data UA Detrac"
OUTPUT_DIR = BASE_DIR / "data"

# Đường dẫn dữ liệu gốc (nested folder structure)
IMAGES_DIR = DATA_DIR / "DETRAC-Images" / "DETRAC-Images"
TRAIN_XML_DIR = DATA_DIR / "DETRAC-Train-Annotations-XML" / "DETRAC-Train-Annotations-XML"
TEST_XML_DIR = DATA_DIR / "DETRAC-Test-Annotations-XML" / "DETRAC-Test-Annotations-XML"

# Kích thước ảnh UA-DETRAC (cố định)
IMG_WIDTH = 960
IMG_HEIGHT = 540

# Tỷ lệ chia tập TRAIN official (60 seq) thành Train/Val
VAL_RATIO = 0.2  # 20% của 60 seq ≈ 12 seq → Val, 48 seq → Train

# Seed cố định để tái tạo kết quả
RANDOM_SEED = 42


def parse_ignored_regions(sequence_root):
    """Trích xuất danh sách vùng bị bỏ qua (ignored_region) từ XML."""
    ignored = []
    ignored_elem = sequence_root.find("ignored_region")
    if ignored_elem is not None:
        for box in ignored_elem.findall("box"):
            left = float(box.get("left"))
            top = float(box.get("top"))
            width = float(box.get("width"))
            height = float(box.get("height"))
            ignored.append((left, top, left + width, top + height))
    return ignored


def compute_iou(box_a, box_b):
    """Tính IoU giữa 2 box dạng (x1, y1, x2, y2)."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter

    if union == 0:
        return 0.0
    return inter / union


def is_in_ignored_region(target_box, ignored_regions, iou_threshold=0.5):
    """Kiểm tra target có nằm trong vùng ignored hay không."""
    for ignored_box in ignored_regions:
        if compute_iou(target_box, ignored_box) > iou_threshold:
            return True
    return False


def convert_to_yolo(left, top, width, height, img_w, img_h):
    """Chuyển tọa độ (left, top, width, height) sang YOLO format (chuẩn hóa)."""
    x_center = (left + width / 2) / img_w
    y_center = (top + height / 2) / img_h
    w = width / img_w
    h = height / img_h

    # Clamp về [0, 1]
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))

    return x_center, y_center, w, h


def process_xml(xml_path, seq_name, output_images_dir, output_labels_dir):
    """Parse 1 file XML annotation và tạo các file .txt YOLO tương ứng.
    
    Returns:
        stats (dict): Thống kê số frame, số bbox đã convert.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Lấy ignored regions
    ignored_regions = parse_ignored_regions(root)

    # Tìm thư mục ảnh của sequence này
    seq_images_dir = IMAGES_DIR / seq_name
    if not seq_images_dir.exists():
        print(f"  [WARN] Image dir not found: {seq_images_dir}")
        return {"frames": 0, "boxes": 0, "skipped": 0}

    stats = {"frames": 0, "boxes": 0, "skipped": 0}

    for frame in root.findall("frame"):
        frame_num = int(frame.get("num"))
        img_filename = f"img{frame_num:05d}.jpg"
        img_src = seq_images_dir / img_filename

        if not img_src.exists():
            continue

        # Tên file duy nhất: SEQ_FRAMENUM (tránh trùng giữa các sequence)
        unique_name = f"{seq_name}_{img_filename.replace('.jpg', '')}"
        img_dst = output_images_dir / f"{unique_name}.jpg"
        label_dst = output_labels_dir / f"{unique_name}.txt"

        # Copy ảnh (dùng symlink nếu muốn tiết kiệm ổ đĩa, ở đây dùng copy)
        if not img_dst.exists():
            shutil.copy2(img_src, img_dst)

        # Parse targets trong frame
        target_list = frame.find("target_list")
        lines = []

        if target_list is not None:
            for target in target_list.findall("target"):
                box = target.find("box")
                left = float(box.get("left"))
                top = float(box.get("top"))
                width = float(box.get("width"))
                height = float(box.get("height"))

                # Kiểm tra có nằm trong ignored region không
                target_box = (left, top, left + width, top + height)
                if is_in_ignored_region(target_box, ignored_regions):
                    stats["skipped"] += 1
                    continue

                # Bỏ qua box quá nhỏ (< 10px)
                if width < 10 or height < 10:
                    stats["skipped"] += 1
                    continue

                # Convert sang YOLO format — class 0 = vehicle
                xc, yc, w, h = convert_to_yolo(left, top, width, height, IMG_WIDTH, IMG_HEIGHT)
                lines.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
                stats["boxes"] += 1

        # Ghi file label (kể cả file rỗng cho negative sample)
        with open(label_dst, "w") as f:
            f.write("\n".join(lines))

        stats["frames"] += 1

    return stats


def main():
    print("=" * 60)
    print("UA-DETRAC -> YOLO Format Converter")
    print("=" * 60)

    # Thu thập danh sách sequence TRAIN và TEST
    train_sequences = sorted([f.stem for f in TRAIN_XML_DIR.glob("*.xml")])
    test_sequences = sorted([f.stem for f in TEST_XML_DIR.glob("*.xml")])

    print(f"\nFound: {len(train_sequences)} train sequences, {len(test_sequences)} test sequences")

    # Chia Train official (60 seq) → Train (48) + Val (12) theo sequence-level
    random.seed(RANDOM_SEED)
    random.shuffle(train_sequences)
    n_val = max(1, int(len(train_sequences) * VAL_RATIO))
    val_sequences = train_sequences[:n_val]
    train_sequences_final = train_sequences[n_val:]

    print(f"\nSplit (sequence-level, seed={RANDOM_SEED}):")
    print(f"  Train: {len(train_sequences_final)} sequences")
    print(f"  Val:   {len(val_sequences)} sequences")
    print(f"  Test:  {len(test_sequences)} sequences")

    # Lưu danh sách chia tập để tái tạo
    split_info_dir = OUTPUT_DIR / "split_info"
    split_info_dir.mkdir(parents=True, exist_ok=True)

    for split_name, seq_list in [("train", train_sequences_final), ("val", val_sequences), ("test", test_sequences)]:
        with open(split_info_dir / f"{split_name}_sequences.txt", "w") as f:
            f.write("\n".join(seq_list))

    # Tạo thư mục output
    splits = {
        "train": (train_sequences_final, TRAIN_XML_DIR),
        "val": (val_sequences, TRAIN_XML_DIR),
        "test": (test_sequences, TEST_XML_DIR),
    }

    total_stats = {}
    for split_name, (seq_list, xml_dir) in splits.items():
        print(f"\n{'-' * 40}")
        print(f"Processing: {split_name.upper()} ({len(seq_list)} sequences)")
        print(f"{'-' * 40}")

        images_dir = OUTPUT_DIR / split_name / "images"
        labels_dir = OUTPUT_DIR / split_name / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        split_stats = {"frames": 0, "boxes": 0, "skipped": 0}

        for i, seq_name in enumerate(seq_list):
            xml_path = xml_dir / f"{seq_name}.xml"
            if not xml_path.exists():
                print(f"  [{i+1}/{len(seq_list)}] {seq_name}: XML not found, skipping")
                continue

            stats = process_xml(xml_path, seq_name, images_dir, labels_dir)
            split_stats["frames"] += stats["frames"]
            split_stats["boxes"] += stats["boxes"]
            split_stats["skipped"] += stats["skipped"]

            print(f"  [{i+1}/{len(seq_list)}] {seq_name}: {stats['frames']} frames, {stats['boxes']} boxes, {stats['skipped']} skipped")

        total_stats[split_name] = split_stats

    # In thống kê tổng hợp
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Split':<8} {'Sequences':>10} {'Frames':>10} {'Boxes':>10} {'Skipped':>10}")
    print("-" * 50)
    for split_name, (seq_list, _) in splits.items():
        s = total_stats[split_name]
        print(f"{split_name:<8} {len(seq_list):>10} {s['frames']:>10} {s['boxes']:>10} {s['skipped']:>10}")

    print(f"\n[DONE] Hoan tat! Du lieu YOLO duoc luu tai: {OUTPUT_DIR}")
    print(f"   Sequence lists saved to: {split_info_dir}")


if __name__ == "__main__":
    main()
