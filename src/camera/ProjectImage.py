import cv2
import numpy as np
import os
from .VisualOdometry import VisualOdometry

class ProjectImage:
    def __init__(self, calibration_matrix):
        self.vo = VisualOdometry(calibration_matrix)
        
    def create_object_mask(self, frame_bgr):
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([80, 255, 255])
        return 255 - cv2.inRange(hsv, lower_green, upper_green)

    def erode_mask(self, mask, erode_px=10):
        kernel = np.ones((erode_px, erode_px), np.uint8)
        return cv2.erode(mask, kernel, iterations=1)

    def order_corners(self, pts):
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def warp_and_blend(self, frame_bgr, design_bgra, mask_region):
        out = frame_bgr.copy()
        if np.sum(mask_region) == 0:
            return out

        contours, _ = cv2.findContours(mask_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return out

        largest_ct = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(largest_ct, True)
        approx = cv2.approxPolyDP(largest_ct, 0.02 * peri, True)

        if len(approx) != 4:
            x, y, w, h = cv2.boundingRect(largest_ct)
            approx = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)
        else:
            approx = approx.reshape(-1, 2).astype(np.float32)

        h_des, w_des = design_bgra.shape[:2]
        src_corners = np.array([[0, 0], [w_des, 0], [w_des, h_des], [0, h_des]], dtype=np.float32)

        kp = self.vo.process_frame(frame_bgr, mask_region)
        if kp is not None and len(self.vo.poses) > 1:
            pose_transform = self.vo.poses[-1]
            try:
                approx = cv2.perspectiveTransform(approx[np.newaxis, :, :], pose_transform[:3]).squeeze()
            except cv2.error:
                pass

        M = cv2.getPerspectiveTransform(src_corners, approx)
        warped = cv2.warpPerspective(design_bgra, M, (frame_bgr.shape[1], frame_bgr.shape[0]))

        mask_f = mask_region.astype(float) / 255.0
        alpha = (warped[..., 3] / 255.0) * mask_f
        design_rgb = warped[..., :3]

        for c in range(3):
            out[..., c] = design_rgb[..., c] * alpha + out[..., c] * (1 - alpha)

        return out

    def process_video(self, input_video, output_video, design_bgra, erode_px=10):
        if not os.path.exists(input_video):
            return

        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        os.makedirs(os.path.dirname(output_video), exist_ok=True)
        writer = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_w, frame_h))

        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            mask = self.create_object_mask(frame_bgr)
            mask_eroded = self.erode_mask(mask, erode_px)
            final_frame = self.warp_and_blend(frame_bgr, design_bgra, mask_eroded)
            writer.write(final_frame)

        cap.release()
        writer.release()