import torch
import numpy as np
from transformers import DPTFeatureExtractor, DPTForDepthEstimation

class DepthEstimator:
    def __init__(self, model_name='Intel/dpt-hybrid-midas', device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.extractor = DPTFeatureExtractor.from_pretrained(model_name)
        self.model = DPTForDepthEstimation.from_pretrained(model_name).to(self.device).eval()

    def estimate(self, frame):
        inputs = self.extractor(images=frame, return_tensors="pt").to(self.device)
        with torch.no_grad():
            prediction = self.model(**inputs).predicted_depth[0].cpu().numpy()
        return prediction 