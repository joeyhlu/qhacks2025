import cv2
import numpy as np
import torch
import requests
from io import BytesIO
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import os


###############################################################################
# DESIGN GENERATION & IMAGE CONVERSIONS
###############################################################################

def generate_ai_design(
    prompt: str,
    negative_prompt: str = "blurry, anything unrelated to the prompt, people, texture",
    model_name: str = "runwayml/stable-diffusion-v1-5",
    width: int = 512,
    height: int = 512,
    device: str = "cuda",
    guidance_scale: float = 30,
    steps: int = 20,
) -> Image.Image:
    """
    Generates an AI-based design using Stable Diffusion, returning a PIL RGBA image.
    """
    print(f"[INFO] Loading model: {model_name}")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    print("[INFO] Prompt:", prompt)
    print("[INFO] Negative Prompt:", negative_prompt)
    print("[INFO] Generating design...")

    # Casting autocast if CUDA, else normal no_grad
    with torch.autocast(device) if device == "cuda" else torch.no_grad():
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=steps
        )

    pil_img = result.images[0].convert("RGBA")
    return pil_img


def pillow_to_bgra(pil_img: Image.Image) -> np.ndarray:
    """
    Converts a PIL RGBA image into an OpenCV BGRA numpy array.
    """
    arr = np.array(pil_img, dtype=np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)


def download_image_bgra(url_or_path: str) -> np.ndarray:
    """
    Loads an image from either a URL or a local path, converting it to BGRA format.
    Returns None if loading fails.
    """
    try:
        if url_or_path.strip().lower().startswith("http"):
            resp = requests.get(url_or_path, timeout=10)
            resp.raise_for_status()
            pil_img = Image.open(BytesIO(resp.content)).convert("RGBA")
        else:
            if not os.path.exists(url_or_path):
                raise FileNotFoundError(f"File not found: {url_or_path}")
            pil_img = Image.open(url_or_path).convert("RGBA")
    except Exception as e:
        print(f"[ERROR] Could not load image from '{url_or_path}': {e}")
        return None

    return pillow_to_bgra(pil_img)


###############################################################################
# OPTIONAL IMAGE MANIPULATIONS (TATTOO, BG REMOVAL, ETC.)
###############################################################################

def remove_background_white_from_bgra(bgra: np.ndarray, threshold=240) -> np.ndarray:
    """
    Makes near-white pixels (>= threshold) fully transparent by adjusting alpha channel.
    Only works if the array has 4 channels (BGRA).
    """
    if bgra.shape[2] != 4:
        print("[WARN] remove_background_white_from_bgra: image missing alpha channel?")
        return bgra

    white_mask = np.all(bgra[..., :3] >= threshold, axis=2)
    bgra[white_mask, 3] = 0
    return bgra


def lower_tattoo_opacity(bgra: np.ndarray, alpha_value=128) -> np.ndarray:
    """
    Lowers the alpha channel to alpha_value on all non-transparent pixels,
    giving a more subtle "tattoo" effect.
    """
    if bgra.shape[2] != 4:
        print("[WARN] no alpha channel, can't lower opacity.")
        return bgra

    out = bgra.copy()
    mask_non_transparent = out[..., 3] > 0
    out[mask_non_transparent, 3] = alpha_value
    return out


###############################################################################
# FLEXIBLE DESIGN LOADING FUNCTION (NO INTERACTIVE PROMPTS)
###############################################################################

def get_design_bgra(
    source_type: str = "prompt",
    source_value: str = "",
    device: str = "cuda",
    is_tattoo: bool = False,
    remove_white_bg: bool = False,
    alpha_value: int = 128,
    neg_prompt: str = "blurry, anything unrelated to the prompt, people, texture",
) -> np.ndarray:
    """
    Retrieves a design BGRA image either by generating (Stable Diffusion) or
    loading from a URL/local path. Provides optional tattoo-like opacity adjustments.
    
    :param source_type: "prompt" (AI generation) or "file" (local/URL image).
    :param source_value: If source_type="prompt", this is the text prompt.
                        If source_type="file", this is the path or URL.
    :param device: "cuda" or "cpu".
    :param is_tattoo: If True, apply black/white ink style + alpha-lowering.
    :param remove_white_bg: If True, remove near-white background.
    :param alpha_value: If lowering alpha for a tattoo effect, set it here.
    :param neg_prompt: Negative prompt for AI generation.
    :return: BGRA numpy array or None if fails.
    """
    design_bgra = None

    if source_type == "file":
        # Load from local path or URL
        design_bgra = download_image_bgra(source_value)
        if design_bgra is None:
            print(f"[ERROR] Could not load design from file/URL: {source_value}")
            return None
    else:
        # Generate via Stable Diffusion
        # If is_tattoo, you might add a style hint to the prompt
        if is_tattoo:
            source_value += ", black and white ink style"

        pil_img = generate_ai_design(
            prompt=source_value,
            negative_prompt=neg_prompt,
            device=device
        )
        design_bgra = pillow_to_bgra(pil_img)

    if design_bgra is None:
        return None

    if is_tattoo:
        if remove_white_bg:
            design_bgra = remove_background_white_from_bgra(design_bgra, threshold=240)
        design_bgra = lower_tattoo_opacity(design_bgra, alpha_value=alpha_value)

    return design_bgra


###############################################################################
# COLOR-BASED MASKING & WARPING
###############################################################################

def create_object_mask(frame_bgr: np.ndarray):
    """
    By default, inverts a green mask. 
    If you want to detect green objects, you can adapt the color range below.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    # Return inverted mask => so everything not green is white
    # If you actually want just the green region, omit the invert.
    return 255 - green_mask


def erode_mask(mask: np.ndarray, erode_px=10) -> np.ndarray:
    """
    Erodes the mask to remove small noises. Adjust erode_px as needed.
    """
    kernel = np.ones((erode_px, erode_px), np.uint8)
    return cv2.erode(mask, kernel, iterations=1)


def order_corners(pts):
    """
    Orders a 4-point contour (e.g., from approxPolyDP) in the order:
    [top-left, top-right, bottom-right, bottom-left].
    """
    rect = np.zeros((4,2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def warp_and_blend(frame_bgr: np.ndarray, design_bgra: np.ndarray, mask_region: np.ndarray) -> np.ndarray:
    """
    Warps `design_bgra` onto the largest contour in `mask_region`. 
    If the contour has 4 corners, do a perspective warp to those corners.
    Otherwise, do a boundingRect overlay.
    """
    out = frame_bgr.copy()

    cts, _ = cv2.findContours(mask_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cts:
        return out
    largest_ct = max(cts, key=cv2.contourArea)

    peri = cv2.arcLength(largest_ct, True)
    approx = cv2.approxPolyDP(largest_ct, 0.0000000000001 * peri, True)

    if len(approx) == 4:
        approx = approx.reshape(-1, 2).astype(np.float32)
        approx = order_corners(approx)

        h_des, w_des = design_bgra.shape[:2]
        src_corners = np.array([[0, 0], [w_des, 0], [w_des, h_des], [0, h_des]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(src_corners, approx)
        warped = cv2.warpPerspective(design_bgra, M, (out.shape[1], out.shape[0]))

        alpha_design = warped[..., 3].astype(float) / 255.0
        design_rgb = warped[..., :3]
        mask_f = mask_region.astype(float) / 255.0

        for c in range(3):
            out[..., c] = design_rgb[..., c] * alpha_design * mask_f + out[..., c] * (1 - alpha_design * mask_f)
    else:
        # If not a quadrilateral, fallback to boundingRect overlay
        x, y, w, h = cv2.boundingRect(largest_ct)
        if w <= 0 or h <= 0:
            return out

        design_resized = cv2.resize(design_bgra, (w, h), interpolation=cv2.INTER_AREA)
        if design_resized.shape[2] == 4:
            alpha_design = design_resized[..., 3].astype(float) / 255.0
            design_rgb = design_resized[..., :3]
        else:
            alpha_design = np.ones((h, w), dtype=np.float32)
            design_rgb = design_resized

        roi = out[y : y + h, x : x + w]
        mask_f = (mask_region[y : y + h, x : x + w].astype(float)) / 255.0
        final_alpha = alpha_design * mask_f
        for c in range(3):
            roi[..., c] = design_rgb[..., c] * final_alpha + roi[..., c] * (1 - final_alpha)

        out[y : y + h, x : x + w] = roi

    return out


###############################################################################
# MULTI-SECTION OVERLAY (E.G., "CAP" VS "BODY")
###############################################################################

def fill_mask_with_two_designs(frame_bgr: np.ndarray, design_bgra_cap: np.ndarray, design_bgra_body: np.ndarray,
                               mask_eroded: np.ndarray, cap_ratio=0.2) -> np.ndarray:
    """
    Splits the masked region into two parts: 'cap' (top portion) and 'body' (the rest).
    Then warps each design separately into each portion.
    Useful for bottle designs with a distinct cap vs. body region.
    """
    out = frame_bgr.copy()
    cts, _ = cv2.findContours(mask_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cts:
        return out
    largest = max(cts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    if w <= 0 or h <= 0:
        return out

    cap_height = int(h * cap_ratio)
    if cap_height < 1:
        cap_height = 1
    cap_y = y
    body_y = y + cap_height
    body_height = h - cap_height

    sub_mask_cap = np.zeros_like(mask_eroded)
    sub_mask_body = np.zeros_like(mask_eroded)

    sub_mask_cap[cap_y : cap_y + cap_height, x : x + w] = mask_eroded[cap_y : cap_y + cap_height, x : x + w]
    sub_mask_body[body_y : body_y + body_height, x : x + w] = mask_eroded[body_y : body_y + body_height, x : x + w]

    out = warp_and_blend(out, design_bgra_cap, sub_mask_cap)
    out = warp_and_blend(out, design_bgra_body, sub_mask_body)
    return out


###############################################################################
# OFFLINE VIDEO PROCESSING FUNCTIONS (OPTIONAL)
###############################################################################

def process_video_single_design(
    input_video: str,
    output_video: str,
    design_bgra: np.ndarray,
    erode_px: int = 10
) -> None:
    """
    Processes a given input video frame-by-frame, applying warp_and_blend
    with a single design. Ideal for color-based overlay offline.
    """
    if not os.path.exists(input_video):
        print(f"[ERROR] Input video '{input_video}' doesn't exist.")
        return

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"[ERROR] cannot open '{input_video}'")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Single design: {frame_w}x{frame_h} @ {fps} fps, total frames={total_frames}")

    out_dir = os.path.dirname(output_video)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video, fourcc, fps, (frame_w, frame_h))

    frame_count = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        mask = create_object_mask(frame_bgr)
        mask_eroded = erode_mask(mask, erode_px=erode_px)
        final_frame = warp_and_blend(frame_bgr, design_bgra, mask_eroded)

        writer.write(final_frame)
        frame_count += 1
        if frame_count % 50 == 0:
            print(f"[INFO] Single design processed {frame_count}/{total_frames}...")

    cap.release()
    writer.release()
    print(f"[INFO] Output saved => {output_video}")


def process_video_bottle(
    input_video: str,
    output_video: str,
    design_bgra_cap: np.ndarray,
    design_bgra_body: np.ndarray,
    erode_px: int = 10,
    cap_ratio: float = 0.2
) -> None:
    """
    Processes a given input video, splitting the color-based mask region
    into a 'cap' portion vs. 'body' portion, applying two different designs.
    """
    if not os.path.exists(input_video):
        print(f"[ERROR] '{input_video}' doesn't exist.")
        return
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"[ERROR] cannot open '{input_video}'")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Bottle: {frame_w}x{frame_h} @ {fps} fps, total frames={total_frames}")

    out_dir = os.path.dirname(output_video)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video, fourcc, fps, (frame_w, frame_h))

    frame_count = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        mask = create_object_mask(frame_bgr)
        mask_eroded = erode_mask(mask, erode_px=erode_px)
        final_frame = fill_mask_with_two_designs(
            frame_bgr, design_bgra_cap, design_bgra_body, mask_eroded, cap_ratio=cap_ratio
        )

        writer.write(final_frame)
        frame_count += 1
        if frame_count % 50 == 0:
            print(f"[INFO] Bottle processed {frame_count}/{total_frames} frames...")

    cap.release()
    writer.release()
    print(f"[INFO] Output saved => {output_video}")
