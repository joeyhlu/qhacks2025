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
from GenerateAndMap import get_design_bgra, warp_and_blend, compute_pose_transform
from ObjectDetection import ObjectDetection
from data import COCO_CLASSES

def smooth_pose_transform(prev_transform, curr_transform, alpha=0.9):
    if prev_transform is None or curr_transform is None:
        return curr_transform
    return alpha * prev_transform + (1 - alpha) * curr_transform

def main():
    calibration_matrix = np.array([
        [815.1466689653845, 0, 638.4755231076894],
        [0, 814.8708730117189, 361.6966488002967],
        [0, 0, 1]
    ])

    # ===== Instantiate CombinedTracker (Step 1) =====
    tracker = CombinedTracker(
        calibration_matrix=calibration_matrix,
        capture_index=0,  # Webcam index or video file path
        cls='person',  # Target class in COCO_CLASSES
        confidence_threshold=0.5
    )

    # ===== Prompt user to generate or load design (Step 2) =====
    design_bgra = get_design_bgra(device="cuda", is_tattoo=True)
    if design_bgra is None:
        print("[ERROR] No design was loaded or generated. Exiting.")
        return
    print("[DEBUG] Design BGRA Shape:", design_bgra.shape)
    print("[DEBUG] Design Alpha Channel Unique Values:", np.unique(design_bgra[..., 3]))
    cv2.imshow("Design BGRA", design_bgra[..., :3])  # Display the design

    # ===== Capture frames and overlay design =====
    cap = cv2.VideoCapture(tracker.capture_index)  # Use the tracker for input
    if not cap.isOpened():
        print("[ERROR] Cannot open video/camera index:", tracker.capture_index)
        return

    print("[INFO] Starting real-time overlay...")
    prev_keypoints = None
    prev_descriptors = None
    pose_transform = None
    prev_pose_transform = None  # For smoothing

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Step 1: Get masks
        person_mask, nonperson_mask = tracker.get_masks_fullframe(frame)
        object_mask = nonperson_mask

        # Display the mask overlay
        overlay = cv2.addWeighted(frame, 0.6, cv2.cvtColor(object_mask, cv2.COLOR_GRAY2BGR), 0.4, 0)
        cv2.imshow("Mask Overlay", overlay)

        print("[DEBUG] Object Mask Unique Values:", np.unique(object_mask))  # Debugging mask
        if np.sum(object_mask) == 0:
            print("[ERROR] Object mask is empty. Skipping frame.")
            continue

        # Step 2: Extract keypoints and descriptors
        orb = cv2.ORB_create()
        curr_keypoints, curr_descriptors = orb.detectAndCompute(frame, object_mask)

        # Debugging keypoints
        print(f"[DEBUG] Number of Keypoints: {len(curr_keypoints) if curr_keypoints else 0}")

        if prev_keypoints is not None and prev_descriptors is not None:
            pose_transform = compute_pose_transform(prev_keypoints, curr_keypoints, prev_descriptors, curr_descriptors)
            pose_transform = smooth_pose_transform(prev_pose_transform, pose_transform)
            prev_pose_transform = pose_transform

            print("[DEBUG] Pose Transform Matrix:", pose_transform)
            if pose_transform is None:
                print("[INFO] Pose transform could not be computed for this frame.")

        prev_keypoints, prev_descriptors = curr_keypoints, curr_descriptors

        # Step 3: Warp and blend the design
        if pose_transform is None:
            print("[INFO] Using static placement due to missing pose transform.")
            augmented_frame = warp_and_blend(frame, design_bgra, object_mask)
        else:
            augmented_frame = warp_and_blend(frame, design_bgra, object_mask, pose_transform)

        # Debugging warped design and final frame
        cv2.imshow("Warped Design", design_bgra[..., :3])  # Display the warped design
        cv2.imshow("Augmented Frame", augmented_frame)  # Display the augmented frame

        # Display result
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
