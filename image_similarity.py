# image_similarity.py
import os
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from ultralytics import YOLO
from Siamese import SiameseNetwork
import torch.nn as nn
class ImageSimilarity:
    def __init__(self, model_path, reference_image, folder_path, threshold, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        yolo_model = YOLO('yolo11n-cls.pt')
        self.siamese_model = SiameseNetwork(
            feature_extractor=nn.Sequential(*list(yolo_model.model.children())[:-1]),
            embedding_dim=256
        ).to(self.device)
        
        if os.path.exists(model_path):
            self.siamese_model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded Siamese model from {model_path}")
        else:
            raise FileNotFoundError(f"No trained Siamese model found at {model_path}")
        
        self.siamese_model.eval()
        
        self.transformation = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        if isinstance(reference_image, np.ndarray):
            reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)
            reference_pil = Image.fromarray(reference_image)
        else:
            reference_pil = reference_image
        
        self.reference_image = self.transformation(reference_pil).unsqueeze(0).to(self.device)
        self.folder_path = folder_path
        self.threshold = threshold

    def find_most_similar_image(self):
        lowest_distance = float('inf')
        most_similar_subfolder = None
        
        with torch.no_grad():
            ref_embedding = self.siamese_model(self.reference_image)
        
        for subfolder in os.listdir(self.folder_path):
            subfolder_path = os.path.join(self.folder_path, subfolder)
            if os.path.isdir(subfolder_path):  
                
                for filename in os.listdir(subfolder_path):
                    image_path = os.path.join(subfolder_path, filename)
                    if os.path.isfile(image_path):
                        candidate_image = Image.open(image_path).convert('RGB')
                        candidate_tensor = self.transformation(candidate_image).unsqueeze(0).to(self.device)
                        
                        with torch.no_grad():
                            candidate_embedding = self.siamese_model(candidate_tensor)
                        
                        distance = torch.norm(ref_embedding - candidate_embedding, p=2).item()
                        
                        if distance < lowest_distance:
                            lowest_distance = distance
                            most_similar_subfolder = subfolder

        if most_similar_subfolder and lowest_distance < self.threshold:
            return most_similar_subfolder
        return 'Unknown'