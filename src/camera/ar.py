import os
import cv2
import time
import torch
import requests
import numpy as np
import matplotlib.pyplot as plt

from io import BytesIO
from PIL import Image
from torchvision.transforms import functional as F

from Tracker import CombinedTracker
from GenerateAndMap import get_design_bgra, warp_and_blend
from ObjectDetection import ObjectDetection
from data import COCO_CLASSES


def main():
    calibration_matrix = np.array([
        [815.1466689653845, 0, 638.4755231076894],
        [0, 814.8708730117189, 361.6966488002967],
        [0, 0, 1]
    ])

    # ===== 2. Instantiate your CombinedTracker (Step 1) =====
    #     - Set capture_index=0 for webcam or pass path to a video file
    #     - cls='person' or any other class in COCO_CLASSES
    tracker = CombinedTracker(
        calibration_matrix=calibration_matrix,
        capture_index=0,        # webcam index or you can provide a video file path
        cls='bottle',           # target class in COCO_CLASSES
        confidence_threshold=0.5
    )

    # ===== 3. Prompt user to generate or load design (Step 2) =====
    design_bgra = get_design_bgra(device="cuda", is_tattoo=True)
    if design_bgra is None:
        print("[ERROR] No design was loaded or generated. Exiting.")
        return

    # ===== 4. Capture frames and overlay design =====
    cap = cv2.VideoCapture(tracker.capture_index)  # from your CombinedTracker
    if not cap.isOpened():
        print("[ERROR] Cannot open video/camera index:", tracker.capture_index)
        return

    print("[INFO] Starting real-time overlay...")
    while True:
        print("?")
        ret, frame = cap.read()
        if not ret:
            print("[INFO] No more frames or camera feed ended.")
            break

        # ----- Step 1: Get segmentation masks -----
        person_mask, nonperson_mask = tracker.get_masks_fullframe(frame)

        # Decide which mask to overlay on; if you're focusing on "person", use person_mask:
        object_mask = nonperson_mask

        # ----- Step 2: Warp & blend design onto the mask -----
        augmented_frame = warp_and_blend(frame, design_bgra, object_mask)

        # Display the result
        cv2.imshow("Augmented Reality View", augmented_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
