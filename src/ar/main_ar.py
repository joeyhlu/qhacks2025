import cv2  # For video capture and frame manipulation
import numpy as np  # For matrix and mask operations
import torch  # For PyTorch operations
from ..camera.Tracker import CombinedTracker
from generate import get_design_bgra, warp_and_blend  # Import the overlay tools


def main(calibration_matrix, capture_index, target_class, design_source=None):
    """
    Combines Tracker.py and generate.py to apply a design overlay on detected objects.
    """
    # Initialize the CombinedTracker from Tracker.py
    tracker = CombinedTracker(calibration_matrix, capture_index, target_class)

    # Load or generate the design/logo from generate.py
    design_bgra = get_design_bgra(
        source_type="file" if design_source else "prompt",
        source_value=design_source or "Minimalist black-and-white bottle logo",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Open the video capture
    cap = cv2.VideoCapture(capture_index)
    if not cap.isOpened():
        print("[ERROR] Unable to open video capture.")
        return

    print("[INFO] Starting tracker with overlay. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame capture failed, exiting.")
            break

        # Use the tracker to get the masks
        person_mask, nonperson_mask = tracker.get_masks_fullframe(frame)

        # Combine the masks if necessary (optional)
        combined_mask = cv2.bitwise_or(person_mask, nonperson_mask)

        # Apply the overlay design to the combined mask
        augmented_frame = warp_and_blend(frame.copy(), design_bgra, combined_mask)

        # Display the augmented frame
        cv2.imshow("Augmented Frame", augmented_frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Tracking session ended.")


if __name__ == "__main__":
    # Example calibration matrix
    calibration_matrix = np.array([
        [815.1466689653845, 0, 638.4755231076894],
        [0, 814.8708730117189, 361.6966488002967],
        [0, 0, 1]
    ])

    # Run the combined tracker with overlay
    main(
        calibration_matrix=calibration_matrix,
        capture_index=0,  # Adjust based on your webcam index
        target_class="bottle",  # Replace with the class label of your target object
        design_source=None  # Use None to generate a design or provide a file path/URL
    )
