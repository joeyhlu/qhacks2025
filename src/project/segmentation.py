# segmentation.py

import cv2
import numpy as np
import torch
import torchvision
from torchvision.transforms import functional as F

class SegmentationModel:
    def __init__(self, target_class='person'):
        # Load pre-trained Mask R-CNN model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        self.model.to(self.device)
        self.model.eval()
        
        # COCO dataset class names
        self.COCO_CLASSES = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
            'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 
            'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        
        # Set target class
        self.target_class = target_class
        self.target_class_id = self.COCO_CLASSES.index(target_class)

    def preprocess_image(self, frame):
        # Convert BGR to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert to PyTorch tensor and normalize
        image = F.to_tensor(image)
        return image

    def get_segmentation(self, frame):
        # Preprocess the image
        image = self.preprocess_image(frame)
        
        # Add batch dimension
        image = image.unsqueeze(0)
        image = image.to(self.device)

        with torch.no_grad():
            prediction = self.model(image)[0]

        # Get masks for all detected objects
        masks = prediction['masks']
        scores = prediction['scores']
        labels = prediction['labels']

        # Filter predictions for target class with confidence > 0.5
        mask_threshold = 0.5
        target_indices = (labels == self.target_class_id) & (scores > mask_threshold)
        
        if target_indices.sum() == 0:
            return np.zeros(frame.shape[:2], dtype=bool)

        # Get the mask of the highest confidence detection for target class
        best_score_idx = scores[target_indices].argmax()
        target_mask = masks[target_indices][best_score_idx][0] > mask_threshold

        # Convert to numpy array
        return target_mask.cpu().numpy()

# Initialize the segmentation model
segmentation_model = None

def get_segmentation_model(target_class='person'):
    """
    Get or create a segmentation model for the specified target class
    Args:
        target_class (str): Class name to segment (e.g., 'person', 'chair', 'dog')
    """
    global segmentation_model
    if segmentation_model is None or segmentation_model.target_class != target_class:
        segmentation_model = SegmentationModel(target_class)
    return segmentation_model

def segment_frame(frame, target_class='person'):
    """
    Perform segmentation on the input frame using Mask R-CNN
    Args:
        frame: Input image frame
        target_class (str): Class name to segment (e.g., 'person', 'chair', 'dog')
    """
    model = get_segmentation_model(target_class)
    return model.get_segmentation(frame)

# Keep the dummy segmentation as fallback
def dummy_segmentation(frame):
    h, w, _ = frame.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask, (w//4, h//4), (3*w//4, 3*h//4), 255, -1)
    return mask > 0
