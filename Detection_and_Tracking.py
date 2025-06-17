#Detection_and_tracking.py
from ultralytics import YOLO

class VideoTracker:
    def __init__(self, detection_model, conf=0.5):
        try:
            self.conf = conf
            self.model = YOLO(detection_model)
            self.names = self.model.model.names
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def run(self, frame):
        results = self.model.track(frame, persist=True, tracker="bytetrack.yaml",conf = self.conf)
        output = {"boxes": [], "class": [], "track_id": [], "Names": self.names}
        if results[0].boxes is not None and results[0].boxes.id is not None:
            output["boxes"] = results[0].boxes.xyxy.cpu().tolist()
            output["class"] = results[0].boxes.cls.cpu().tolist()
            output["track_id"] = results[0].boxes.id.int().cpu().tolist()
        return output