#!/usr/bin/env python3
# YOLO webcam → warn if a detection's CENTER moves to a different quadrant (t-1 → t).
# Matching = IoU-first (greedy, same-class) with center-in-expanded-box fallback.

import cv2
import numpy as np
from ultralytics import YOLO
from typing import Tuple, Dict, List, Optional

# ---------- geometry ----------
#determine the center of a box
def center_of(xyxy: np.ndarray):
    x1, y1, x2, y2 = xyxy
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

#determine which quadrant a point is in
def quadrant_of(pt: Tuple[float, float], W: int, H: int):
    x, y = pt
    left = x < W / 2.0
    top  = y < H / 2.0 # y axis goes down
    if left and top: return "Q2"
    if not left and top: return "Q1"
    if left and not top: return "Q3"
    return "Q4"

#expands a box by a scale factor
def expand_box(xyxy: np.ndarray, scale: float, W: int, H: int) -> np.ndarray:
    x1, y1, x2, y2 = xyxy
    cx = 0.5 * (x1+x2)
    cy = 0.5 * (y1+y2)
    w  = (x2-x1) * scale
    h = (y2-y1) * scale
    nx1 = max(0.0, cx- 0.5 *w)
    ny1 = max(0.0, cy- 0.5* h)
    nx2 = min(float(W-1), cx +0.5 *w)
    ny2 = min(float(H-1), cy+ 0.5 *h)
    return np.array([nx1, ny1, nx2, ny2], dtype=np.float32)

#check if a point is inside a box
def point_in_box(pt: Tuple[float,float], box: np.ndarray):
    x, y = pt
    x1, y1, x2, y2 = box
    return (x1 <= x <= x2) and (y1 <= y <= y2)

#compute Intersection-over-Union (IoU) between two boxes
def box_iou_matrix(a_xyxy: np.ndarray, b_xyxy: np.ndarray) -> np.ndarray:
    Na, Nb = a_xyxy.shape[0], b_xyxy.shape[0]
    if Na == 0 or Nb == 0:
        return np.zeros((Na, Nb), dtype=np.float32)
    ax1, ay1, ax2, ay2 = a_xyxy[:,0], a_xyxy[:,1], a_xyxy[:,2], a_xyxy[:,3]
    bx1, by1, bx2, by2 = b_xyxy[:,0], b_xyxy[:,1], b_xyxy[:,2], b_xyxy[:,3]
    a_area = np.clip(ax2-ax1, 0, None) * np.clip(ay2-ay1, 0, None)
    b_area = np.clip(bx2-bx1, 0, None) * np.clip(by2-by1, 0, None)
    lt_x = np.maximum(ax1[:,None], bx1[None,:])
    lt_y = np.maximum(ay1[:,None], by1[None,:])
    rb_x = np.minimum(ax2[:,None], bx2[None,:])
    rb_y = np.minimum(ay2[:,None], by2[None,:])
    inter_w = np.clip(rb_x - lt_x, 0, None)
    inter_h = np.clip(rb_y - lt_y, 0, None)
    inter = inter_w * inter_h
    union = a_area[:,None] + b_area[None,:] - inter
    return inter / np.clip(union, 0, None)

#match current boxes to previous using IoU
def match_by_iou(cur_boxes, cur_cls, prev_boxes, prev_cls, iou_thr=0.20):
    Nc, Np = cur_boxes.shape[0], prev_boxes.shape[0]
    out: [] = [None] * Nc
    if Nc == 0 or Np == 0:
        return out
    iou = box_iou_matrix(cur_boxes, prev_boxes) #create the iou matrix for all pairs shape: (Nc, Np)
    same = (cur_cls[:,None] == prev_cls[None,:]) #same[i,j] is True if cur_boxes[i] and prev_boxes[j] are same class
    iou = np.where(same, iou, 0.0) #ignore different-class pairs by setting IoU to 0 for them. masking essentially
    best_prev = iou.argmax(axis=1) #for each current box, the index of the previous box with the highest IoU (best_prev[i] = index of prev box with highest IoU for cur box i)
    best_iou  = iou.max(axis=1) #for each current box, the highest IoU value (best_iou[i] = highest IoU value for cur box i)
    cands = [(int(i), int(best_prev[i]), float(best_iou[i])) for i in range(Nc) if best_iou[i] >= iou_thr] 
    #cands list is triple, (i, j, best_iou[i]) for each current box i that has a best IoU >= threshold, where j is the index of the matched previous box
    if not cands:
        return out
    cands.sort(key=lambda t: t[2], reverse=True) #sorting candidates by IoU score best_iou[i] above (high to low)
    used_i, used_j = set(), set()
    for i, j, _ in cands: #mapping current box i to previous box j. Skip if either is already used
        if i in used_i or j in used_j: continue
        out[i] = j
        used_i.add(i)
        used_j.add(j)
    return out # out is a list of length Nc, where out[i] is the index of the matched previous box for current box i, or None if no match

def hybrid_match_iou_then_center(cur_boxes, cur_cls, prev_boxes, prev_cls,
                                 imgW, imgH, iou_thr=0.20,
                                 expand_scale=1.8, max_dist_frac=0.4):
    Nc, Np = cur_boxes.shape[0], prev_boxes.shape[0]
    matches = match_by_iou(cur_boxes, cur_cls, prev_boxes, prev_cls, iou_thr=iou_thr)
    if Nc == 0 or Np == 0:
        return matches

    unmatched = [i for i, j in enumerate(matches) if j is None]
    if not unmatched:
        return matches

    cur_cent  = np.stack([0.5*(cur_boxes[:,0]+cur_boxes[:,2]), 0.5*(cur_boxes[:,1]+cur_boxes[:,3])], axis=1)
    prev_cent = np.stack([0.5*(prev_boxes[:,0]+prev_boxes[:,2]), 0.5*(prev_boxes[:,1]+prev_boxes[:,3])], axis=1)
    prev_exp  = np.stack([expand_box(prev_boxes[j], expand_scale, imgW, imgH) for j in range(Np)], axis=0)

    diag = (imgW ** 2 + imgH ** 2) ** 0.5
    max_dist = max_dist_frac * diag if max_dist_frac > 0 else float("inf")

    used_prev = set(j for j in matches if j is not None)
    for i in unmatched:
        cid = int(cur_cls[i])
        cx, cy = cur_cent[i]
        best = None 
        for j in range(Np):
            if j in used_prev: continue
            if int(prev_cls[j]) != cid: continue
            if not point_in_box((cx, cy), prev_exp[j]): continue
            dist = np.hypot(cx - prev_cent[j,0], cy - prev_cent[j,1])
            if dist <= max_dist and (best is None or dist < best[0]):
                best = (dist, j)
        if best is not None:
            matches[i] = best[1]
            used_prev.add(best[1])

    return matches


def main():
    model = YOLO("yolo11x.pt")
    model.to("cuda:0") 

    cap = cv2.VideoCapture("C:/Users/osman/Downloads/HighwayCV.mp4")
    if not cap.isOpened():
        raise SystemExit("Cannot open webcam 0")

    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps) if fps and fps > 0 else 1

    win = "Quadrant change detector"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    #initiating previous state variables
    prev_boxes = np.zeros((0,4), dtype=np.float32)
    prev_cls   = np.zeros((0,), dtype=np.int32)
    prev_quad: Dict[int, str] = {}
    names = None

    while True:
        ok, frame = cap.read()
        if not ok: break
        H, W = frame.shape[:2]

        raw = frame.copy()  
        r = model.predict(source=raw, conf=0.40, iou=0.50, imgsz=640, verbose=False)[0]
        if names is None: 
            names = r.names

        # draw quadrant crosshair
        cv2.line(frame, (W//2, 0), (W//2, H), (0,255,0), 1)
        cv2.line(frame, (0, H//2), (W, H//2), (0,255,0), 1)

        cur_boxes = np.zeros((0,4), dtype=np.float32)
        cur_cls   = np.zeros((0,), dtype=np.int32)
        

        if r.boxes is not None and r.boxes.xyxy.numel() > 0:
            cur_boxes = r.boxes.xyxy.cpu().numpy().astype(np.float32)
            cur_cls   = r.boxes.cls.cpu().numpy().astype(np.int32)

        # match (IoU first, if no match, center check in expanded box)
        match_idx = hybrid_match_iou_then_center(
            cur_boxes, cur_cls, prev_boxes, prev_cls,imgW=W, imgH=H,iou_thr=0.20, expand_scale=1.8, max_dist_frac=0.4)

        # draw YOLO boxes + center
        for i in range(cur_boxes.shape[0]):
            x1, y1, x2, y2 = cur_boxes[i]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cx, cy = (x1 + x2)//2, (y1 + y2)//2
            q_now = quadrant_of((cx, cy), W, H)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.circle(frame, (cx, cy), 4, (0,255,255), -1)

            j = match_idx[i]
            if j is not None and j in prev_quad:
                q_prev = prev_quad[j]
                if q_prev != q_now:
                    warn = f"WARNING: {names[cur_cls[i]]} center moved {q_prev} → {q_now}"
                    print(warn)
                    cv2.putText(frame, warn, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,165,255), 2, cv2.LINE_AA)

        # update previous
        prev_boxes = cur_boxes
        prev_cls   = cur_cls
        prev_quad  = {j: quadrant_of(center_of(prev_boxes[j]), W, H) for j in range(prev_boxes.shape[0])}

        cv2.imshow(win, frame)
        if (cv2.waitKey(delay) & 0xFF) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

#current problems:

#When some thing that is detected just in the middle of two quadrants, it flickers between the two quadrants issuing frequent warnings
#When two objects of the same class are close to each other, they can swap quadrants
#When an object is moving fast, it can jump quadrants (can be solved by increasing the frame rate or a better model/matching/tracking)
#Sometimes a detection is missed for a frame or two, which can cause false warnings

#Some ideas for improvements:

#Thinking of adding a filter for the warnings, so that a warning is only issued if the object has been in the new quadrant for a certain number of frames
#Thinking of adding a filter for when some object is around the edges of the quadrants for a certain time, then no warnings
#Thinking of adding a feature to prevent warnings like "car moved Q1 to Q2 and again Q1 to Q2 and again Q1 to Q2" as it doesn't make sense