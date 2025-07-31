import supervision as sv
from rfdetr import RFDETRBase, RFDETRNano
from rfdetr.util.coco_classes import COCO_CLASSES

model = RFDETRNano()

def callback(frame, index):
    detections = model.predict(frame[:, :, ::-1].copy(), threshold=0.5)

    labels = [
        f"{COCO_CLASSES[class_id]} {confidence:.2f}"
        for class_id, confidence
        in zip(detections.class_id, detections.confidence)
    ]

    annotated_frame = frame.copy()
    annotated_frame = sv.BoxAnnotator().annotate(annotated_frame, detections)
    annotated_frame = sv.LabelAnnotator().annotate(annotated_frame, detections, labels)
    return annotated_frame

sv.process_video(
    source_path="video\market-square.mp4",
    target_path="video\market-square-output.mp4",
    callback=callback
)
