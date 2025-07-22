import cv2
import numpy as np
from ultralytics import YOLO

VIDEO_PATH = 'subway (1).mp4'
MODEL_PATH = 'yolov8n.pt'

# Polygon coordinates from annotation (on 562x997 image)
ANNOT_POLYGON = np.array([[174, 880], [230, 634], [536, 658], [554, 878], [167, 884]])
ANNOT_W, ANNOT_H = 562, 997

def scale_point(pt, orig_w, orig_h):
    x, y = pt
    return (int(x * orig_w / ANNOT_W), int(y * orig_h / ANNOT_H))

def main():
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print('Failed to open video file.')
        return

    # Get original frame size
    ret, frame = cap.read()
    if not ret:
        print('Failed to read first frame.')
        return
    orig_h, orig_w = frame.shape[:2]

    # Scale polygon to frame size
    scaled_polygon = np.array([scale_point(pt, orig_w, orig_h) for pt in ANNOT_POLYGON])

    # Rewind video to first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Draw polygon
        cv2.polylines(frame, [scaled_polygon], isClosed=True, color=(255, 0, 255), thickness=3)

        # Detect people
        results = model(frame)[0]
        boxes = results.boxes
        if boxes is not None and boxes.xyxy is not None:
            class_ids = boxes.cls.cpu().numpy().astype(int)
            xyxy = boxes.xyxy.cpu().numpy()
            person_indices = [i for i, cid in enumerate(class_ids) if cid == 0]
            count_in_zone = 0
            for i in person_indices:
                x1, y1, x2, y2 = map(int, xyxy[i])
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                # Check if center is inside polygon
                if cv2.pointPolygonTest(scaled_polygon, center, False) >= 0:
                    count_in_zone += 1
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, center, 5, (0, 255, 0), -1)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.circle(frame, center, 5, (0, 0, 255), -1)
        else:
            count_in_zone = 0

        # Show count
        cv2.putText(frame, f'People in zone: {count_in_zone}', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 5)

        # Resize for display to match annotation size
        display_frame = cv2.resize(frame, (562, 997))
        cv2.imshow('Zone People Counting', display_frame)
        key = cv2.waitKey(100) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
