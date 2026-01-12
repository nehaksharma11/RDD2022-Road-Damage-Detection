import argparse
from ultralytics import YOLO
import os
import cv2

def predict(args):
    model = YOLO(args.model)

    os.makedirs(args.output, exist_ok=True)

    results = model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        save=False,
        device=args.device
    )

    for result in results:
        image_name = os.path.splitext(os.path.basename(result.path))[0]
        txt_path = os.path.join(args.output, f"{image_name}.txt")

        with open(txt_path, "w") as f:
            if result.boxes is None:
                continue

            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x, y, w, h = box.xywhn[0].tolist()
                f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f} {conf:.6f}\n")

    print("Predictions saved successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on test images")

    parser.add_argument("--model", type=str, default="models/best_model.pt")
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--output", type=str, default="samples/predictions")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", type=str, default="0")

    args = parser.parse_args()
    predict(args)
