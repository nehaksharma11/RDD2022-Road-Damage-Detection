from ultralytics import YOLO
import argparse

def evaluate(args):
    model = YOLO(args.model)

    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        device=args.device
    )

    print("Evaluation completed.")
    print(metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 model")

    parser.add_argument("--model", type=str, default="models/best_model.pt")
    parser.add_argument("--data", type=str, default="docs/data.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", type=str, default="0")

    args = parser.parse_args()
    evaluate(args)
