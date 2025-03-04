import cv2
import numpy as np
import pygame
from pygame.locals import QUIT
import torch
import os

from project.depth_estimator import load_midas_model, process_frame
from project.segmentation import segment_frame
from project.mesh_generation import create_mesh_from_depth
from project.opengl_renderer import (init_pygame_opengl, create_buffers, update_buffers,
                             create_texture, update_texture, compile_shaders, set_uniform_matrices)
from project.shaders import vertex_shader_code, fragment_shader_code

# OpenGL imports (unchanged)
from OpenGL.GL import (
    GL_TEXTURE_2D, GL_RGB, GL_RGBA, GL_UNSIGNED_BYTE, GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER, GL_LINEAR, GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T,
    GL_CLAMP_TO_EDGE, glTexParameteri, glTexImage2D, glBindTexture, 
    glClear, GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT,
    glUseProgram, glEnableVertexAttribArray, glBindBuffer,
    glVertexAttribPointer, glDrawElements, GL_ARRAY_BUFFER, 
    GL_FLOAT, GL_FALSE, GL_ELEMENT_ARRAY_BUFFER, GL_TRIANGLES, 
    GL_UNSIGNED_INT, glDisableVertexAttribArray, GL_TEXTURE0, glActiveTexture, glGetUniformLocation, glUniform1i, glViewport, glEnable, GL_DEPTH_TEST, glClearColor
)

def init_display(width, height):
    """Initialize display for both original and rendered images."""
    total_width = width * 2  # Double width for side-by-side display
    pygame.init()
    display = pygame.display.set_mode((total_width, height), pygame.DOUBLEBUF | pygame.OPENGL)
    pygame.display.set_caption("3D Object Viewer")
    
    # Enable OpenGL features
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_TEXTURE_2D)
    # Set clear color to light gray for debugging
    glClearColor(0.8, 0.8, 0.8, 1.0)
    
    return display

def process_image(image_path, target_class='bottle'):
    """Process a single image."""
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Unable to load image: {image_path}")
        return
    return process_frame_data(frame, target_class)

def process_video(video_path, target_class='bottle'):
    """Process a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Unable to open video: {video_path}")
        return
    
    ret, frame = cap.read()
    if not ret:
        print("Unable to read video frame")
        return
    
    return cap, process_frame_data(frame, target_class)

def process_frame_data(frame, target_class):
    """Common processing for both image and video frames."""
    height, width, _ = frame.shape
    fx, fy = 500, 500
    cx, cy = width / 2, height / 2
    
    # Initialize MiDaS model for depth estimation
    midas, device, transform = load_midas_model()
    
    # Initialize display and OpenGL (only once per run)
    display = init_display(width, height)
    init_pygame_opengl(width, height)
    VBO, TBO, EBO = create_buffers()
    texture_id = create_texture(width, height)
    shader_program = compile_shaders(vertex_shader_code, fragment_shader_code)
    set_uniform_matrices(shader_program, width, height)
    
    return {
        'frame_size': (width, height),
        'camera_params': (fx, fy, cx, cy),
        'midas_data': (midas, device, transform),
        'gl_data': (VBO, TBO, EBO, texture_id, shader_program),
        'screen': display,
        'target_class': target_class
    }

def main():
    # Data paths
    data_dir = "src/data"
    image_path = os.path.join(data_dir, "water.png")
    video_path = os.path.join(data_dir, "test.mp4")
    texture_path = os.path.join(data_dir, "image.png")
    
    mode = input("Select mode (1: Image, 2: Video): ").strip()
    
    if mode == "1":
        data = process_image(image_path)
        if data is None:
            return
        
        running = True
        clock = pygame.time.Clock()
        
        while running:
            print("bruh")
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
            
            frame = cv2.imread(image_path)
            process_frame_gl(frame, data, texture_path)
            clock.tick(30)
    
    elif mode == "2":
        cap, data = process_video(video_path)
        if data is None:
            return
        
        running = True
        clock = pygame.time.Clock()
        
        while running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
            
            ret, frame = cap.read()
            if not ret:
                break
                
            process_frame_gl(frame, data, texture_path)
            pygame.display.flip()
            clock.tick(30)
            
        cap.release()
    
    cv2.destroyAllWindows()
    pygame.quit()

def process_frame_gl(frame, data, texture_path):
    """Process a frame with OpenGL rendering."""
    width, height = data['frame_size']
    fx, fy, cx, cy = data['camera_params']
    midas, device, transform = data['midas_data']
    VBO, TBO, EBO, texture_id, shader_program = data['gl_data']
    target_class = data['target_class']

    # Display the original frame on the left using pygame
   # cv2.imshow("Original Frame", frame)
    
    # Load and display texture image for debugging
    texture_img = cv2.imread(texture_path, cv2.IMREAD_UNCHANGED)
    if texture_img is not None:
        #cv2.imshow("Texture Image", texture_img)
        pass
    
    # --- Compute Full Depth Map First ---
    # Compute depth for the entire frame (using a full mask of ones)
    full_mask = np.ones((frame.shape[0], frame.shape[1]), dtype=bool)
    depth_map_full = process_frame(frame, midas, device, transform, full_mask)
    
    # --- Apply Segmentation Afterwards ---
    mask = segment_frame(frame, target_class=target_class)
    if mask is not None:
        cv2.imshow("Segmentation Mask", (mask * 255).astype(np.uint8))
    
    # Now apply the segmentation mask to the full depth map
    depth_map = depth_map_full * mask
    if depth_map is not None:
        # Normalize depth map for visualization
        depth_viz = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-6) * 255
        depth_viz = depth_viz.astype(np.uint8)
        cv2.imshow("Depth Map", depth_viz)
    
    cv2.waitKey(1)
    
    # Clear the color and depth buffers
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
    # Draw original frame on left side (using pygame's 2D blit)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
    data['screen'].blit(frame_surface, (0, 0))
    
    # Set viewport for the right half of the window
    glViewport(width, 0, width, height)
    
    # Load and preprocess texture image
    texture_img = cv2.imread(texture_path, cv2.IMREAD_UNCHANGED)
    if texture_img is None:
        print(f"Unable to load texture: {texture_path}")
        return
    
    if texture_img.shape[2] == 4:
        texture_img = cv2.cvtColor(texture_img, cv2.COLOR_BGRA2RGBA)
    else:
        texture_img = cv2.cvtColor(texture_img, cv2.COLOR_BGR2RGB)
    texture_img = cv2.flip(texture_img, 0)
    
    # Bind the shader program
    glUseProgram(shader_program)
    
    # Set up texture
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    
    # Upload texture data
    if texture_img.shape[2] == 4:
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, texture_img.shape[1], texture_img.shape[0],
                    0, GL_RGBA, GL_UNSIGNED_BYTE, texture_img)
    else:
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, texture_img.shape[1], texture_img.shape[0],
                    0, GL_RGB, GL_UNSIGNED_BYTE, texture_img)
    
    # --- Fix: Set the correct texture uniform ---
    tex_loc = glGetUniformLocation(shader_program, "videoTexture")
    glUniform1i(tex_loc, 0)
    
    # Create mesh from the depth map (using the segmentation mask)
    mesh_data = create_mesh_from_depth(depth_map, mask, fx, fy, cx, cy, step=8)
    if mesh_data[0] is not None:
        vertices, tex_coords, indices = mesh_data
        update_buffers(VBO, TBO, EBO, vertices, tex_coords, indices)
        
        glEnableVertexAttribArray(0)
        glEnableVertexAttribArray(1)
        
        glBindBuffer(GL_ARRAY_BUFFER, VBO)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        
        glBindBuffer(GL_ARRAY_BUFFER, TBO)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, None)
        
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
        glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)
        
        glDisableVertexAttribArray(1)
        glDisableVertexAttribArray(0)
    
    glUseProgram(0)
    cv2.imshow("Texture", texture_img)

    pygame.display.flip()

if __name__ == '__main__':
    main()
