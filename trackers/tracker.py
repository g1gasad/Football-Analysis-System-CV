from ultralytics import YOLO
import supervision as sv

class Trackers:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames):
        batch_size = 20
        results = []
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i : i + batch_size] 
            batch_results = self.model.predict(batch_frames, conf=0.1)
            results.extend(batch_results)
            break
        return results  

    def get_object_tracks(self, frame):
        detections = self.detect_frames(frame)

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            # convert to supervision detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)
            print(detection_supervision)