import argparse
from ultralytics import YOLO
import os

def train(args):
    os.makedirs("models", exist_ok=True)

    model = YOLO(args.model)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        optimizer=args.optimizer,
        lr0=args.lr,
        patience=args.patience,
        device=args.device,
        project="runs/train",
        name="rdd2022_yolov8",
        exist_ok=True
    )

    print("Training completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 on RDD2022 dataset")

    parser.add_argument("--data", type=str, default="docs/data.yaml")
    parser.add_argument("--model", type=str, default="yolov8m.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--optimizer", type=str, default="AdamW")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--device", type=str, default="0")

    args = parser.parse_args()
    train(args)
