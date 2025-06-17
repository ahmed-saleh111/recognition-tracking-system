# Siamese.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseNetwork(nn.Module):
    def __init__(self, feature_extractor, embedding_dim=256):
        super(SiameseNetwork, self).__init__()
        self.feature_extractor = feature_extractor
        
        with torch.no_grad():
            sample_input = torch.randn(1, 3, 128, 128)
            features = feature_extractor(sample_input)
            feature_dim = features.view(features.size(0), -1).shape[1]
        
        self.dropout = nn.Dropout(0.3)
        self.projection = nn.Linear(feature_dim, embedding_dim)

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        features = self.dropout(features)
        embedding = self.projection(features)
        return F.normalize(embedding, p=2, dim=1)