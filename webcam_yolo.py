#!/usr/bin/env python3
# Minimal YOLO webcam: opens a window and draws detections. That's it.

import argparse
import cv2
from ultralytics import YOLO

def parse_args():
    p = argparse.ArgumentParser("Minimal YOLO webcam")
    p.add_argument("--weights", type=str, default="yolov8n.pt",
                   help="Model weights (e.g., yolov8n.pt, yolov8s.pt, or your best.pt)")
    p.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=0.50, help="NMS IoU threshold")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    p.add_argument("--device", type=str, default="", help="''(auto), 'cuda:0', 'cpu', or 'mps'")
    p.add_argument("--cam", type=int, default=0, help="Webcam index (0,1,...)")
    return p.parse_args()

def main():
    args = parse_args()

    # load model (optionally move to a device)
    model = YOLO(args.weights).to("cuda:0")
    if args.device:
        model.to(args.device)

    # open webcam
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open webcam index {args.cam}")

    win = "YOLO webcam"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # run yolo on this frame; ultralytics handles bgr->rgb & letterbox internally
        results = model.predict(
            source=frame,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            verbose=False,
        )
        r = results[0]

        # draw detections
        if r.boxes is not None and r.boxes.xyxy.numel() > 0:
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss  = r.boxes.cls.cpu().numpy().astype(int)
            names = r.names
            for (x1, y1, x2, y2), score, cid in zip(boxes, confs, clss):
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                label = f"{names[cid]} {score:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                y_text = max(y1 - 7, 15)
                cv2.rectangle(frame, (x1, y_text - th - 4), (x1 + tw + 4, y_text + 4), (0,0,0), -1)
                cv2.putText(frame, label, (x1 + 2, y_text),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)

        cv2.imshow(win, frame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
