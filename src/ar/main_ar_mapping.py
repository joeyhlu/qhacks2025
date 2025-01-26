# File: src/ar/main_ar_mapping.py

import cv2
import torch

# Import these directly from generate.py
from generate import (
    generate_ai_design,
    download_image_bgra,
    pillow_to_bgra,
    get_design_bgra,
    create_object_mask,
    erode_mask,
    warp_and_blend,
)

def main_ar_mapping(camera_index=0, design_source=None, is_tattoo=False):
    """
    Real-time AR that uses the color-based approach in generate.py.
    It finds a region (e.g., green color) in each camera frame, then 
    warps/blends an RGBA design onto that region.
    """

    # 1. Capture from webcam
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[ERROR] Unable to open camera index {camera_index}.")
        return

    # 2. Load or generate the design
    device = "cuda" if torch.cuda.is_available() else "cpu"
    design_bgra = None

    if design_source:
        # If a URL or local file path is provided
        design_bgra = download_image_bgra(design_source)
        if design_bgra is None:
            print(f"[ERROR] Could not load design from '{design_source}'.")
            return
    else:
        # Otherwise, interactively prompt or just do a minimal example 
        # with AI generation:
        prompt = "A simple black logo"
        neg_prompt = "blurry, people, text"
        print("[INFO] Generating design via Stable Diffusion pipeline.")
        pil_img = generate_ai_design(
            prompt=prompt,
            negative_prompt=neg_prompt,
            device=device
        )
        design_bgra = pillow_to_bgra(pil_img)

    # 2b. If you want the tattoo approach (transparent, partial alpha)
    if is_tattoo:
        from generate import remove_background_white_from_bgra, lower_tattoo_opacity
        # Remove near-white background & lower overall alpha
        design_bgra = remove_background_white_from_bgra(design_bgra, threshold=240)
        design_bgra = lower_tattoo_opacity(design_bgra, alpha_value=128)

    if design_bgra is None:
        print("[ERROR] No valid design loaded or generated.")
        return

    print("[INFO] Starting real-time color-based AR mapping.")
    print("      Press 'q' to quit the window.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 3. Create a mask for the region we want to overlay (e.g., green object)
        #    You can tweak create_object_mask to look for a different color range
        mask = create_object_mask(frame)

        # 4. Optionally erode or refine the mask
        mask_eroded = erode_mask(mask, erode_px=10)

        # 5. Warp and blend the design onto the masked region
        #    This does perspective transform if a 4-corner contour is found
        #    Otherwise it does boundingRect overlay
        output_frame = warp_and_blend(frame, design_bgra, mask_eroded)

        cv2.imshow("Real-Time AR Mapping", output_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Example usage:
    #  Provide a path/URL or leave `design_source=None` to generate with AI
    main_ar_mapping(
        camera_index=0,
        design_source="data/logo.png",  # or "http://somewhere/my_logo.png"
        is_tattoo=False
    )
