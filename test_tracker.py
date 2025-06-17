import cv2
from ultralytics.utils.plotting import Annotator
from Detection_and_Tracking import VideoTracker

tracker = VideoTracker("models/new_detection.pt") 

video_path = "videos/output_000.mp4"  
cap = cv2.VideoCapture(video_path)


if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break 

    output = tracker.run(frame)

    annotator = Annotator(frame, line_width=2)

    for box, cls, track_id in zip(output["boxes"], output["class"], output["track_id"]):
        label = f"{output['Names'][int(cls)]} {track_id}"
        annotator.box_label(box, label=label, color=(255, 0, 0)) 

    annotated_frame = annotator.result()

    cv2.imshow("Tracked Video", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
