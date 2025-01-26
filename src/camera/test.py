import cv2
import numpy as np

# =====================================================
# 1. Load reference images (object & overlay design)
# =====================================================
object_img = cv2.imread("./src/board.png", cv2.IMREAD_GRAYSCALE)
design_img = cv2.imread("./src/cute.png")

if object_img is None:
    raise IOError("Could not load object_reference.jpg")
if design_img is None:
    raise IOError("Could not load my_logo.png")

# For better matching, we might keep the object reference in grayscale
# The design (logo) can remain in color

# =====================================================
# 2. Initialize the ORB feature detector
# =====================================================
orb = cv2.ORB_create()

# Compute keypoints and descriptors for the reference object
kp_object, des_object = orb.detectAndCompute(object_img, None)

# =====================================================
# 3. Create a function to detect the object and get homography
# =====================================================
def detect_object_and_compute_homography(frame):
    """
    Detects the planar object in the given frame using ORB keypoints 
    and computes a homography matrix if enough matches are found.
    Returns H (3x3 homography) or None if not found.
    """
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect keypoints and descriptors in the live frame
    kp_frame, des_frame = orb.detectAndCompute(gray_frame, None)
    if des_frame is None or len(kp_frame) < 2:
        return None

    # Use a BFMatcher to match descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_object, des_frame)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Filter "good" matches (tweak the number as needed)
    good_matches = matches[:50]  # top 50 matches, for example

    # We need at least 4 matches to compute a homography
    if len(good_matches) < 4:
        return None

    # Extract the matched keypoints’ locations
    src_pts = np.float32([kp_object[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
    dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

    # Compute homography using RANSAC
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return H

# =====================================================
# 4. Start capturing video from the default camera
# =====================================================
cap = cv2.VideoCapture(0)  # Change index if you have multiple cameras

# =====================================================
# 5. Main loop
# =====================================================
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from camera.")
        break

    # Compute homography
    H = detect_object_and_compute_homography(frame)

    if H is not None:
        # If we have a valid homography, warp the design image 
        # to the perspective of the object in the current frame.

        # Get dimensions of the camera frame
        h_frame, w_frame, _ = frame.shape

        # Warp the design to match the perspective of the object
        warped_design = cv2.warpPerspective(design_img, H, (w_frame, h_frame))

        # Create a mask from the warped design where pixels are non-zero
        warped_gray = cv2.cvtColor(warped_design, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(warped_gray, 1, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        # Convert single-channel masks to 3-channel
        mask_3c = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_inv_3c = cv2.cvtColor(mask_inv, cv2.COLOR_GRAY2BGR)

        # Black-out the region in the original frame where the design will go
        frame_bg = cv2.bitwise_and(frame, mask_inv_3c)

        # Place the warped design on top
        final_frame = cv2.add(frame_bg, warped_design)

        # Show the augmented frame
        cv2.imshow("AR Overlay", final_frame)
    else:
        # If no homography was found, just show the original frame
        cv2.imshow("AR Overlay", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
