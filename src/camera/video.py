import os
import cv2
import numpy as np
import streamlit as st
from Tracker import CombinedTracker
from GenerateAndMap import get_design_bgra, warp_and_blend, compute_pose_transform
from ObjectSegmentation import ObjectSegmentation

# Streamlit App Title
st.title("Real-Time Object Design Overlay")

# Streamlit Sidebar Inputs
st.sidebar.header("Configuration")
calibration_matrix = np.array([
    [st.sidebar.number_input("Calibration Matrix [0,0]", 815.1467), 0, st.sidebar.number_input("Calibration Matrix [0,2]", 638.4755)],
    [0, st.sidebar.number_input("Calibration Matrix [1,1]", 814.8709), st.sidebar.number_input("Calibration Matrix [1,2]", 361.6966)],
    [0, 0, 1]
])

capture_index = st.sidebar.number_input("Camera Index", value=0, min_value=0, step=1)
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)
object_class = st.sidebar.selectbox("Object Class", ["person", "bottle"], index=0)
device = st.sidebar.selectbox("Device", ["cuda", "cpu"], index=0)

# Ask whether to generate a prompt
generate_prompt = st.sidebar.checkbox("Generate a Prompt?", value=False)
prompt = None
if generate_prompt:
    prompt = st.sidebar.text_input("Enter your prompt", value="Default Prompt")

is_tattoo = st.sidebar.checkbox("Use Tattoo Design", value=True)

# Helper Function for Pose Smoothing
def smooth_pose_transform(prev_transform, curr_transform, alpha=0.9):
    if prev_transform is None or curr_transform is None:
        return curr_transform
    return alpha * prev_transform + (1 - alpha) * curr_transform

def create_arm_mask(frame, result, side="right"):
    """
    Create a binary mask for the arm region based on pose landmarks.
    """
    if not result.pose_landmarks:
        print("[WARN] No pose landmarks detected.")
        return np.zeros(frame.shape[:2], dtype=np.uint8)

    # Select arm landmark indices
    if side == "right":
        indices = [12, 14, 16]  # Right shoulder, elbow, wrist
    elif side == "left":
        indices = [11, 13, 15]  # Left shoulder, elbow, wrist
    else:
        raise ValueError("Invalid side specified. Use 'right' or 'left'.")

    # Get the landmark positions
    h, w = frame.shape[:2]
    arm_points = []
    for idx in indices:
        lm = result.pose_landmarks.landmark[idx]
        if lm.visibility > 0.5:  # Filter by visibility threshold
            arm_points.append((int(lm.x * w), int(lm.y * h)))

    if len(arm_points) < 3:  # Not enough points to form a valid region
        print("[WARN] Insufficient landmarks for arm region.")
        return np.zeros(frame.shape[:2], dtype=np.uint8)

    # Create a mask for the arm region
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    arm_points = np.array(arm_points, dtype=np.int32)
    cv2.fillPoly(mask, [arm_points], 255)
    return mask

# Run Function
def run():
    tracker = CombinedTracker(
        calibration_matrix=calibration_matrix,
        capture_index=capture_index,
        cls=object_class,
        confidence_threshold=confidence_threshold
    )

    segmenter = ObjectSegmentation(plot=False)

    design_bgra = get_design_bgra(device=device, is_tattoo=is_tattoo, answer=prompt if generate_prompt else None)
    if design_bgra is None:
        st.error("No design was loaded or generated. Exiting.")
        return

    cap = cv2.VideoCapture(tracker.capture_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        st.error(f"Cannot open video/camera index: {tracker.capture_index}")
        return

    st.info("Starting real-time overlay...")
    prev_keypoints, prev_descriptors, prev_pose_transform = None, None, None

    # Streamlit Display Area
    video_frame = st.empty()
    augmented_frame_display = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Get masks
        person_mask, nonperson_mask = tracker.get_masks_fullframe(frame)
        object_mask = person_mask if object_class == 'person' else nonperson_mask

        if object_class != 'person':
            orb = cv2.ORB_create()
            curr_keypoints, curr_descriptors = orb.detectAndCompute(frame, object_mask)
            if prev_keypoints is not None and prev_descriptors is not None:
                pose_transform = compute_pose_transform(prev_keypoints, curr_keypoints, prev_descriptors, curr_descriptors)
                pose_transform = smooth_pose_transform(prev_pose_transform, pose_transform)
                prev_pose_transform = pose_transform
            prev_keypoints, prev_descriptors = curr_keypoints, curr_descriptors
        else:
            # Process person class
            processed_frame, result = segmenter.body_segmentation(frame)
            pose_transform = None
            if result.pose_world_landmarks:
                arm_mask = create_arm_mask(frame, result, side="right")
                if np.sum(arm_mask) > 0:
                    object_mask = arm_mask
                else:
                    object_mask = None

        # Warp and Blend
        if object_mask is not None and np.sum(object_mask) > 0:
            augmented_frame = warp_and_blend(frame, design_bgra, object_mask, pose_transform)
            video_frame.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
            augmented_frame_display.image(cv2.cvtColor(augmented_frame, cv2.COLOR_BGR2RGB), channels="RGB")
        else:
            st.warning("Skipping frame as mask is empty or invalid.")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the App
if st.sidebar.button("Start"):
    run()
