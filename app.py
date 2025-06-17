# main.py
import os
import cv2
import time  
import numpy as np
from Detection_and_Tracking import VideoTracker
from person_recognizer import PersonRecognizer   
from ultralytics.utils.plotting import colors, Annotator   
import torch

detection_model = 'models/detection_model.pt'
yolo_face = 'models/face_detection_model.pt'
person_model = 'models/siamese_model.pt' 
person_data = 'dataset' 
video_path = 'videos/video_1.mp4'

threshold = 0.9
conf = 0.5
voting_frame_window = 60
device = 'cuda' if torch.cuda.is_available() else 'cpu'
show = True
output_video_path = 'videos/output_annotated.mp4'

tracker = VideoTracker(detection_model, conf)
recog = PersonRecognizer(yolo_face, person_model, person_data, threshold, voting_frame_window)

def process_frame_for_recognition(frame, do_recognition=False):
    tracking_results = tracker.run(frame)

    if not tracking_results or "boxes" not in tracking_results:
        tracking_results = {"boxes": [], "track_id": [], "class": []}
    
    recognition_results = recog.run(frame, tracking_results) if do_recognition else recog.output
    return tracking_results, recognition_results

def draw_annotations(frame, tracking_results, recognition_results):
    annotator = Annotator(frame, line_width=2) 
    
    boxes = tracking_results["boxes"]
    clss = tracking_results["class"]
    track_ids = tracking_results["track_id"]
    names = {0: 'Kid', 1: 'Caregiver'}

    DETECTION_COLOR = (0, 255, 0) 
    FACE_COLOR =  (255,255,0)    

    if isinstance(boxes, (np.ndarray, list)) and len(boxes) > 0:
        for box, cls, track_id in zip(boxes, clss, track_ids): 
            class_name = names[int(cls)] 
            x1, y1, x2, y2 = [int(coord) for coord in box]
            
            identity = recognition_results.get(track_id, {}).get('identity', 'Unknown')
            label = f"{identity} ({class_name} {track_id})"
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), DETECTION_COLOR, 2)
            annotator.box_label(box, label=f"{class_name} {track_id} {identity}", color=(255, 0, 0))
            
            text_x = x1 + 5
            text_y = y1 + 20
            # cv2.putText(frame, identity, (text_x, text_y), 
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, DETECTION_COLOR, 2)

    for track_id, info in recognition_results.items():
        det_x1, det_y1, det_x2, det_y2 = [int(coord) for coord in info['box']]
        
        person_crop = frame[det_y1:det_y2, det_x1:det_x2]
        face_boxes = recog.detect_face(person_crop)
        
        if len(face_boxes) > 0:
            for face_box in face_boxes:
                face_x1, face_y1, face_x2, face_y2 = [int(coord) for coord in face_box]
                abs_face_x1 = det_x1 + face_x1
                abs_face_y1 = det_y1 + face_y1
                abs_face_x2 = det_x1 + face_x2
                abs_face_y2 = det_y1 + face_y2
                cv2.rectangle(frame, (abs_face_x1, abs_face_y1), (abs_face_x2, abs_face_y2), 
                            FACE_COLOR, 2)

    texts = [f"{info['identity']}" for info in recognition_results.values()]
    if texts:
        cv2.putText(frame, "Detected Persons:", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "---------------", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        y_pos = 80
        for text in texts:
            cv2.putText(frame, text, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_pos += 25
    
    return frame

def regulate_frame_processing(camera_fps, recognition_spf, video_path, show=False, output_path=None):
    frame_delay = 1 / camera_fps
    recognition_frame_interval = recognition_spf * camera_fps

    recognition_counter = 0
    frame_count = 0
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream")
            break

        frame_count += 1
        do_recognition = (recognition_counter >= recognition_frame_interval)
        
        tracking_results, recognition_results = process_frame_for_recognition(frame, do_recognition)
        
        annotated_frame = draw_annotations(frame, tracking_results, recognition_results)
        
        # Write the annotated frame to the output video
        out.write(annotated_frame)
        
        if show:
            cv2.imshow('Frame', annotated_frame)
        
        if do_recognition:
            recognition_counter = 0
        else:
            recognition_counter += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # time.sleep(frame_delay)  # Uncomment if you need to regulate frame rate

    cap.release()
    out.release()  # Release the VideoWriter
    cv2.destroyAllWindows()
    print(f"Annotated video saved to: {output_path}")

camera_fps = 20
recognition_spf = 0.1
regulate_frame_processing(camera_fps, recognition_spf, video_path, show=show, output_path=output_video_path)


