"""
Training Script for YOLO11n on Vehicle Detection (UA-DETRAC / Traffic Surveillance).

Đặc điểm tối ưu hóa:
1. Freeze Backbone (freeze=10): Khóa 10 tầng đầu (0-9) của YOLO11n để bảo tồn đặc trưng nhận diện
   phương tiện đa góc nhìn đã học từ pre-trained COCO -> chống Catastrophic Forgetting.
2. Learning Rate tối ưu cho Fine-tuning: lr0=0.001, lrf=0.01 với Cosine Annealing (cos_lr=True),
   tránh làm vỡ cấu trúc trọng số như mức mặc định 0.01.
3. Early Stopping (patience=7): Tự động dừng nếu val/loss hoặc mAP không cải thiện trong 7 epochs
   -> ngăn chặn hiện tượng Overfitting sau epoch 15-20.
4. Close Mosaic (close_mosaic=5): Tắt mosaic 5 epochs cuối để mô hình hội tụ bbox chuẩn xác.
"""

import os
import sys
import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Huấn luyện YOLO11n chuẩn cho Vehicle Detection")
    parser.add_argument("--data", type=str, default="data.yaml", help="Đường dẫn tới file data.yaml")
    parser.add_argument("--weights", type=str, default="yolo11n.pt", help="Trọng số khởi tạo (mặc định: yolo11n.pt)")
    parser.add_argument("--epochs", type=int, default=20, help="Số epochs tối đa (mặc định: 20 cho fine-tuning)")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (mặc định: 5)")
    parser.add_argument("--batch", type=int, default=32, help="Batch size (mặc định: 32 cho GPU Colab T4)")
    parser.add_argument("--imgsz", type=int, default=640, help="Kích thước ảnh đầu vào (mặc định: 640)")
    parser.add_argument("--freeze", type=int, default=10, help="Số tầng backbone cần khóa (mặc định: 10 tầng đầu)")
    parser.add_argument("--lr0", type=float, default=0.001, help="Initial learning rate (mặc định: 0.001)")
    parser.add_argument("--lrf", type=float, default=0.01, help="Final learning rate factor (mặc định: 0.01)")
    parser.add_argument("--project", type=str, default="runs/detect", help="Thư mục lưu kết quả train")
    parser.add_argument("--name", type=str, default="yolo11n_vehicle", help="Tên phiên huấn luyện")
    parser.add_argument("--device", type=str, default="", help="Cuda device (e.g. 0 hoặc cpu). Mặc định tự nhận diện.")
    parser.add_argument("--workers", type=int, default=4, help="Số dataloader workers")
    return parser.parse_args()


def train_yolo11n(args):
    print("=" * 70)
    print("🚀 BẮT ĐẦU HUẤN LUYỆN YOLO11n (CHUẨN HÓA CHO GIAO THÔNG & CAMERA ĐƯỜNG PHỐ)")
    print("=" * 70)
    print(f"• Model Base      : {args.weights}")
    print(f"• Data Config     : {args.data}")
    print(f"• Epochs / Patience: {args.epochs} / {args.patience}")
    print(f"• Batch / ImgSz   : {args.batch} / {args.imgsz}")
    print(f"• Freeze Layers   : {args.freeze} (Khóa toàn bộ Backbone 0-9)")
    print(f"• Learning Rate   : lr0={args.lr0}, lrf={args.lrf} (Cosine Annealing)")
    print("=" * 70)

    # 1. Khởi tạo mô hình pre-trained
    model = YOLO(args.weights)

    # 2. Huấn luyện với hyperparameter chuẩn chống Overfitting
    train_kwargs = {
        "data": args.data,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "freeze": args.freeze,       # Khóa 10 tầng backbone (0-9)
        "lr0": args.lr0,             # LR ban đầu nhẹ nhàng cho fine-tuning
        "lrf": args.lrf,             # LR cuối cùng = lr0 * lrf = 1e-5
        "cos_lr": True,              # Giảm LR theo hàm cosine mượt mà
        "warmup_epochs": 2,          # Làm ấm 2 epochs đầu
        "close_mosaic": 5,           # Tắt mosaic trong 5 epochs cuối
        "project": args.project,
        "name": args.name,
        "workers": args.workers,
        "exist_ok": True,
        "verbose": True,
        "plots": True
    }

    if args.device:
        train_kwargs["device"] = args.device

    results = model.train(**train_kwargs)

    best_weight = Path(args.project) / args.name / "weights" / "best.pt"
    print("\n" + "=" * 70)
    print("✅ HUẤN LUYỆN HOÀN TẤT THÀNH CÔNG!")
    print(f"📁 Trọng số tốt nhất (Best checkpoint): {best_weight}")
    print("=" * 70)
    return results, str(best_weight)


if __name__ == "__main__":
    args = parse_args()
    train_yolo11n(args)
