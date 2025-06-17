# person_recognizer.py
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from collections import defaultdict, Counter
from ultralytics import YOLO
from image_similarity import ImageSimilarity

class PersonRecognizer:
    def __init__(self, face_model_path, person_model_path, reference_folder_path, threshold=0.8, voting_frame_window=60):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.face_model = YOLO(face_model_path)  
        self.person_model_path = person_model_path
        self.reference_folder_path = reference_folder_path
        self.threshold = threshold
        self.voting_frame_window = voting_frame_window
        
        self.identity_votes = defaultdict(list)
        self.output = {}
        
        self.transformation = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def load_and_transform_image(self, cropped_image):
        cropped_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
        return self.transformation(cropped_image).unsqueeze(0).to(self.device)

    def predict_identity(self, face_crop):
        image_similarity = ImageSimilarity(
            model_path=self.person_model_path,
            reference_image=face_crop,
            folder_path=self.reference_folder_path,
            threshold=self.threshold,
            device=self.device
        )
        return image_similarity.find_most_similar_image()

    def detect_face(self, person_crop):
        results = self.face_model(person_crop, verbose=False, conf = 0.6)
        return results[0].boxes.xyxy.cpu().numpy()

    def run(self, frame, tracking_output=None):
        if tracking_output: 
            person_boxes = tracking_output["boxes"]
            track_ids = tracking_output["track_id"]
            clss = tracking_output["class"]
        else:  
            results = self.face_model(frame)
            person_boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = range(len(person_boxes))  
            clss = [0] * len(person_boxes) 

        self.output.clear()
        
        for person_box, track_id, cls in zip(person_boxes, track_ids, clss):
            x1, y1, x2, y2 = map(int, person_box)
            person_crop = frame[y1:y2, x1:x2]
            
            face_boxes = self.detect_face(person_crop)
            
            if len(face_boxes) == 0:
                identity = "Unknown" if not self.identity_votes[track_id] else Counter(self.identity_votes[track_id]).most_common(1)[0][0]
            else:
                face_x1, face_y1, face_x2, face_y2 = map(int, face_boxes[0])
                face_crop = person_crop[face_y1:face_y2, face_x1:face_x2]
                identity = self.predict_identity(face_crop)
                
                self.identity_votes[track_id].append(identity)
                if len(self.identity_votes[track_id]) > self.voting_frame_window:
                    self.identity_votes[track_id].pop(0)
                identity = Counter(self.identity_votes[track_id]).most_common(1)[0][0]

            self.output[track_id] = {
                'track_id': track_id,
                'identity': identity,
                'class': "Kid" if cls == 0 else "Caregiver",
                'box': [x1, y1, x2, y2]
            }
        
        seen_identities = set()
        duplicates = []
        for track_id, info in self.output.items():
            if info["identity"] in seen_identities:
                duplicates.append(track_id)
            else:
                seen_identities.add(info["identity"])
        for dup_id in duplicates:
            del self.output[dup_id]
        
        return self.output