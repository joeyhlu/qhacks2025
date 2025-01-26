import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from ObjectSegmentation import ObjectSegmentation
from ObjectDetection import ObjectDetection
from VisualOdometry import VisualOdometry

from dict import COCO_CLASSES

class CombinedTracker:
    def __init__(self, calibration_matrix: np.ndarray, capture_index: int, cls: str, confidence_threshold: float = 0.5):
        """
        Use YOLO to generate bounding boxes and masks.
        Skip object VO if label=1 (person).
        """
        self.conf_threshold = confidence_threshold
        self.coco_class = cls
        self.capture_index = capture_index

        # Imported code (ObjectSegmentation) for drawing MediaPipe poses
        self.seg = ObjectSegmentation(False)
        self.detect = ObjectDetection(capture_index=self.capture_index)

        # Two VO instances: background, object
        self.vo_background = VisualOdometry(calibration_matrix, label="BackgroundVO")
        self.vo_object = VisualOdometry(calibration_matrix, label="ObjectVO")

        # Matplotlib figure for x-z plane
        self.fig, self.ax = plt.subplots()
        self.line_bg, = self.ax.plot([], [], 'b-', label="Camera Trajectory (BG)")
        self.line_obj, = self.ax.plot([], [], 'g-', label="Object Trajectory")

    def get_masks_fullframe(self, frame):
        """
        Returns:
        person_mask: All pixels that belong to any detection with label=1 within the bounding box.
        nonperson_mask: All pixels that belong to non-person detections within the bounding box.
        """
        h, w = frame.shape[:2]
        person_mask = np.zeros((h, w), dtype=np.uint8)
        nonperson_mask = np.zeros((h, w), dtype=np.uint8)

        # Get YOLO bounding box for the target class
        detection = self.detect.find_bounding_box(frame, self.coco_class)

        if detection is None:
            # Return empty masks if no bounding box is found
            return person_mask, nonperson_mask

        (x1, y1, x2, y2), _ = detection

        # Create a mask within the bounding box
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255

        if self.coco_class == "person":
            person_mask = mask
        else:
            nonperson_mask = mask

        return person_mask, nonperson_mask

    def update_2d_plot(self):
        bg_x = self.vo_background.x_coords
        bg_z = self.vo_background.z_coords
        self.line_bg.set_data(bg_x, bg_z)

        obj_x = self.vo_object.x_coords
        obj_z = self.vo_object.z_coords
        self.line_obj.set_data(obj_x, obj_z)

        all_x = np.concatenate([bg_x, obj_x])
        all_z = np.concatenate([bg_z, obj_z])

        if len(all_x) > 0:
            pad = 1.0
            self.ax.set_xlim(np.min(all_x) - pad, np.max(all_x) + pad)
            self.ax.set_ylim(np.min(all_z) - pad, np.max(all_z) + pad)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def run(self):
        cap = cv.VideoCapture(self.capture_index)
        if not cap.isOpened():
            return

        self.ax.set_title("X-Z Trajectories")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Z")
        self.ax.legend()
        plt.ion()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Get person_mask & nonperson_mask from YOLO
            person_mask, nonperson_mask = self.get_masks_fullframe(frame)

            combined_mask = cv.bitwise_or(person_mask, nonperson_mask)
            background_mask = cv.bitwise_not(combined_mask)

            bg_kp = self.vo_background.process_frame(frame, mask=background_mask)
            obj_kp = self.vo_object.process_frame(frame, mask=nonperson_mask)

            self.update_2d_plot()
            display_frame = frame.copy()

            # Mark background as red
            disp_bg = cv.merge([
                background_mask,
                np.zeros_like(background_mask),
                np.zeros_like(background_mask)
            ])

            # Mark all objects (person + non-person) in green
            disp_obj = cv.merge([
                np.zeros_like(combined_mask),
                combined_mask,
                np.zeros_like(combined_mask)
            ])

            display_frame = cv.addWeighted(display_frame, 1.0, disp_obj, 0.4, 0)

            # Draw ORB keypoints
            cv.drawKeypoints(display_frame, bg_kp, display_frame, color=(255, 0, 0))
            cv.drawKeypoints(display_frame, obj_kp, display_frame, color=(0, 255, 0))

            # Segment person pose if a person is detected
            if np.count_nonzero(person_mask) > 0:
                display_frame, _ = self.seg.body_segmentation(display_frame)

            cv.imshow("Combined Tracker + Person Pose", display_frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv.destroyAllWindows()
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    K = np.array([
        [815.1466689653845, 0, 638.4755231076894],
        [0, 814.8708730117189, 361.6966488002967],
        [0, 0, 1]
    ])

    tracker = CombinedTracker(calibration_matrix=K, capture_index=1, cls="person")
    tracker.run()
