# depth_estimator.py

import cv2
import torch
import numpy as np

def load_midas_model():
    # Load a small MiDaS model (may download if not cached)
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas.to(device)
    midas.eval()
    # Load MiDaS transforms
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.small_transform
    return midas, device, transform

def process_frame(frame, midas, device, transform, segmentation_mask):
    # Convert BGR to RGB (MiDaS expects RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Apply the MiDaS transform to prepare the frame
    input_batch = transform(rgb_frame).to(device)
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=rgb_frame.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze()
    depth_map = prediction.cpu().numpy()
    # Mask the depth map to include only the segmented object
    depth_map_masked = depth_map * segmentation_mask
    return depth_map_masked
