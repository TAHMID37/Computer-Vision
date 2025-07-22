import cv2
import numpy as np
from ultralytics import YOLO
import sys

SLOWDOWN_MS = 50  # Increase for slower playback (e.g., 200ms = 5 FPS)

# Path to your video file
VIDEO_PATH = 'vehicles.mp4'  # Change if needed
MODEL_PATH = 'yolov8n.pt'   # Change if needed

# Annotation coordinates from 1198x675 image
ANNOT_LINE_IN_START = (174, 389)
ANNOT_LINE_IN_END = (574, 389)
ANNOT_LINE_OUT_START = (680, 413)
ANNOT_LINE_OUT_END = (1062, 413)
ANNOT_W, ANNOT_H = 1198, 

## yolo sota Object Detection  -> yolov8 yolov5  yolov11m nas 

# Vehicle class IDs in COCO dataset
VEHICLE_CLASS_IDS = [2, 3, 5, 7]  # car, motorcycle, bus, truck

def main():
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"\n[ERROR] Failed to load YOLO model from '{MODEL_PATH}'.")
        print("Make sure you are using the official yolov8n.pt from Ultralytics releases.")
        print("If you are using PyTorch >=2.6, try downgrading to <2.6 or updating Ultralytics.")
        print("Error details:", e)
        sys.exit(1)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print('Failed to open video file.')
        return

    # Read first frame to get original size
    ret, frame = cap.read()
    if not ret:
        print('Failed to read first frame.')
        return
    orig_h, orig_w = frame.shape[:2]

    # Scale annotation coordinates to original frame size
    def scale_point(pt):
        x, y = pt
        return (int(x * orig_w / ANNOT_W), int(y * orig_h / ANNOT_H))
    LINE_IN_START = scale_point(ANNOT_LINE_IN_START)
    LINE_IN_END = scale_point(ANNOT_LINE_IN_END)
    LINE_OUT_START = scale_point(ANNOT_LINE_OUT_START)
    LINE_OUT_END = scale_point(ANNOT_LINE_OUT_END)

    # Rewind video to first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    in_count = 0
    out_count = 0
    already_counted_in = set()
    already_counted_out = set()
    track_history = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Use YOLOv8's built-in DeepSORT tracking
        results = model.track(frame, persist=True, tracker="botsort.yaml", verbose=False)[0]
        boxes = results.boxes
        if boxes is None or boxes.xyxy is None:
            display_frame = cv2.resize(frame, (960, 540))
            cv2.imshow('Vehicle Counting', display_frame)
            key = cv2.waitKey(SLOWDOWN_MS) & 0xFF
            if key == ord('q'):
                break
            continue
        class_ids = boxes.cls.cpu().numpy().astype(int)
        xyxy = boxes.xyxy.cpu().numpy()
        track_ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [None]*len(xyxy)
        vehicle_indices = [i for i, cid in enumerate(class_ids) if cid in VEHICLE_CLASS_IDS]

        # Draw lines and overlays on the original frame
        cv2.line(frame, LINE_IN_START, LINE_IN_END, (255, 0, 0), 2)
        cv2.line(frame, LINE_OUT_START, LINE_OUT_END, (0, 0, 255), 2)

        # Draw and count
        for idx in vehicle_indices:
            x1, y1, x2, y2 = map(int, xyxy[idx])
            track_id = track_ids[idx]
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            # Track history for direction
            if track_id not in track_history:
                track_history[track_id] = []
            track_history[track_id].append(center)
            if len(track_history[track_id]) > 2:
                track_history[track_id] = track_history[track_id][-2:]
            # Only count if we have at least 2 points
            if len(track_history[track_id]) == 2:
                prev_center = track_history[track_id][0]
                curr_center = track_history[track_id][1]
                # For left line (in): count if crossing from above to below
                if (
                    prev_center[0] >= LINE_IN_START[0] and prev_center[0] <= LINE_IN_END[0] and
                    curr_center[0] >= LINE_IN_START[0] and curr_center[0] <= LINE_IN_END[0]
                ):
                    if prev_center[1] < LINE_IN_START[1] and curr_center[1] >= LINE_IN_START[1] and track_id not in already_counted_in:
                        in_count += 1
                        already_counted_in.add(track_id)
                # For right line (out): count if crossing from below to above
                if (
                    prev_center[0] >= LINE_OUT_START[0] and prev_center[0] <= LINE_OUT_END[0] and
                    curr_center[0] >= LINE_OUT_START[0] and curr_center[0] <= LINE_OUT_END[0]
                ):
                    if prev_center[1] > LINE_OUT_START[1] and curr_center[1] <= LINE_OUT_START[1] and track_id not in already_counted_out:
                        out_count += 1
                        already_counted_out.add(track_id)
            # Draw bbox and ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 255, 0), -1)
            cv2.putText(frame, f'ID {track_id}', (center[0] - 10, center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        # Make counts larger and bolder
        cv2.putText(frame, f'In: {in_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0,255,0), 5)
        cv2.putText(frame, f'Out: {out_count}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0,0,255), 5)

        display_frame = cv2.resize(frame, (960, 540))
        cv2.imshow('Vehicle Counting', display_frame)
        key = cv2.waitKey(SLOWDOWN_MS) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
