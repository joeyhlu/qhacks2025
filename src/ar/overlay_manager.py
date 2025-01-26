# File: src/ar/overlay_manager.py

import cv2
import numpy as np

class AROverlayManager:
    """
    Handles overlay of a design image (with optional RGBA channels)
    onto a region of a camera frame (e.g., bounding box) with pose adjustments.
    """
    def __init__(self, design_bgra: np.ndarray):
        """
        design_bgra: an RGBA or BGRA design image from generate.py
                     shape [height, width, 4].
        """
        self.design_bgra = design_bgra

    def overlay_on_bbox(self, frame: np.ndarray, bbox: tuple, pose_transform: np.ndarray = None) -> np.ndarray:
        """
        Overlays the design onto 'frame' at the given bounding box, applying pose transformations.

        bbox: (x1, y1, x2, y2) in integer pixel coordinates.
        pose_transform: 3x3 transformation matrix for perspective warping. If None, simple overlay.
        Returns the augmented frame.
        """
        (x1, y1, x2, y2) = bbox

        # Ensure bbox is within frame bounds
        h_frame, w_frame = frame.shape[:2]
        x1, x2 = sorted([max(0, min(x1, w_frame)), max(0, min(x2, w_frame))])
        y1, y2 = sorted([max(0, min(y1, h_frame)), max(0, min(y2, h_frame))])

        if x2 - x1 <= 0 or y2 - y1 <= 0:
            return frame  # Invalid bounding box => return original

        # Define destination points based on bbox
        dst_pts = np.array([
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2]
        ], dtype=np.float32)

        # If pose_transform is provided, adjust destination points
        if pose_transform is not None:
            dst_pts = cv2.perspectiveTransform(dst_pts.reshape(-1, 1, 2), pose_transform).reshape(-1, 2)

        # Define source points from design image
        design_h, design_w = self.design_bgra.shape[:2]
        src_pts = np.array([
            [0, 0],
            [design_w, 0],
            [design_w, design_h],
            [0, design_h]
        ], dtype=np.float32)

        # Compute perspective transform matrix if needed
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        # Warp the design image to the destination
        warped_design = cv2.warpPerspective(self.design_bgra, M, (w_frame, h_frame), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)

        # Create mask from alpha channel
        if warped_design.shape[2] == 4:
            alpha_mask = warped_design[:, :, 3] / 255.0
            alpha_inv = 1.0 - alpha_mask
            for c in range(3):
                frame[:, :, c] = (alpha_inv * frame[:, :, c] + alpha_mask * warped_design[:, :, c])
        else:
            # No alpha channel, simple overlay
            roi = frame[y1:y2, x1:x2]
            frame[y1:y2, x1:x2] = warped_design

        return frame
