import cv2
from ultralytics import YOLO
import supervision as sv

VIDEO_PATH = 'subway (1).mp4'
MODEL_PATH = 'yolov8n.pt'

def main():
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print('Failed to open video file.')
        return

    # Get frame size
    ret, frame = cap.read()
    if not ret:
        print('Failed to read first frame.')
        return

    # Initialize Supervision HeatMapAnnotator
    heatmap_annotator = sv.HeatMapAnnotator(
        opacity=0.6,
        radius=40,
        kernel_size=25,
        top_hue=0,      # red
        low_hue=120     # blue
    )

    # Rewind video
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect people
        result = model(frame)[0]
        detections = sv.Detections.from_ultralytics(result)
        # Filter for people (class_id == 0)
        mask = detections.class_id == 0
        detections.xyxy = detections.xyxy[mask]
        detections.confidence = detections.confidence[mask]
        detections.class_id = detections.class_id[mask]

        # Annotate heatmap
        frame_with_heatmap = heatmap_annotator.annotate(scene=frame.copy(), detections=detections)

        # Resize to annotation size for display
        display_frame = cv2.resize(frame_with_heatmap, (562, 997))
        cv2.imshow('Supervision Heatmap', display_frame)
        key = cv2.waitKey(1) & 0xFF  # Fastest playback
        if key == ord('q'):
            break

        frame_count += 1
        if frame_count % 50 == 0:
            print(f'Processed {frame_count} frames...')

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
