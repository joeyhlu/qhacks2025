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

# Update the imports at the top of the file
from OpenGL.GL import (
    GL_TEXTURE_2D, GL_RGB, GL_RGBA, GL_UNSIGNED_BYTE, GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER, GL_LINEAR, GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T,
    GL_CLAMP_TO_EDGE, glTexParameteri, glTexImage2D, glBindTexture, 
    glClear, GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT,
    glUseProgram, glEnableVertexAttribArray, glBindBuffer,
    glVertexAttribPointer, glDrawElements, GL_ARRAY_BUFFER, 
    GL_FLOAT, GL_FALSE, GL_ELEMENT_ARRAY_BUFFER, GL_TRIANGLES, 
    GL_UNSIGNED_INT, glDisableVertexAttribArray, GL_TEXTURE0, glActiveTexture, glGetUniformLocation, glUniform1i, glViewport
)

def init_display(width, height):
    """Initialize display for both original and rendered images"""
    total_width = width * 2  # Double width for side-by-side display
    pygame.init()
    # Create a single window with double width
    display = pygame.display.set_mode((total_width, height), pygame.DOUBLEBUF | pygame.OPENGL)
    pygame.display.set_caption("3D Object Viewer")
    
    # Set up viewport for split screen
    glViewport(width, 0, width, height)  # OpenGL renders to right half
    
    return display

def process_image(image_path, target_class='bottle'):
    """Process a single image"""
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Unable to load image: {image_path}")
        return
    return process_frame_data(frame, target_class)

def process_video(video_path, target_class='bottle'):
    """Process a video file"""
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
    """Common processing for both image and video frames"""
    height, width, _ = frame.shape
    fx, fy = 500, 500
    cx, cy = width / 2, height / 2
    
    # Initialize MiDaS model for depth estimation
    midas, device, transform = load_midas_model()
    
    # Initialize display
    display = init_display(width, height)
    
    # Initialize OpenGL
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
        'screen': display
    }

def main():
    # Data paths
    data_dir = "src/data"
    image_path = os.path.join(data_dir, "water.png")
    video_path = os.path.join(data_dir, "test.mp4")
    texture_path = os.path.join(data_dir, "image.png")
    
    # Choose processing mode
    mode = input("Select mode (1: Image, 2: Video): ").strip()
    
    if mode == "1":
        # Process single image
        data = process_image(image_path)
        if data is None:
            return
        
        running = True
        clock = pygame.time.Clock()
        
        while running:
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
        # Process video
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
    
    pygame.quit()

def process_frame_gl(frame, data, texture_path):
    """Process a frame with OpenGL rendering"""
    width, height = data['frame_size']
    fx, fy, cx, cy = data['camera_params']
    midas, device, transform = data['midas_data']
    VBO, TBO, EBO, texture_id, shader_program = data['gl_data']
    
    # Draw original frame on left side
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
    data['screen'].blit(frame_surface, (0, 0))
    
    # Set up OpenGL viewport for right side
    glViewport(width, 0, width, height)
    
    # Load and preprocess texture image
    texture_img = cv2.imread(texture_path, cv2.IMREAD_UNCHANGED)
    if texture_img is None:
        print(f"Unable to load texture: {texture_path}")
        return
    
    # Convert BGR(A) to RGB(A) and flip vertically
    if texture_img.shape[2] == 4:
        texture_img = cv2.cvtColor(texture_img, cv2.COLOR_BGRA2RGBA)
    else:
        texture_img = cv2.cvtColor(texture_img, cv2.COLOR_BGR2RGB)
    texture_img = cv2.flip(texture_img, 0)  # Flip vertically for OpenGL
    
    # Set up texture parameters
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    
    # Upload texture data to GPU
    if texture_img.shape[2] == 4:
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, texture_img.shape[1], texture_img.shape[0],
                    0, GL_RGBA, GL_UNSIGNED_BYTE, texture_img)
    else:
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, texture_img.shape[1], texture_img.shape[0],
                    0, GL_RGB, GL_UNSIGNED_BYTE, texture_img)
    
    # Obtain segmentation mask for bottle
    mask = segment_frame(frame, target_class='bottle')
    
    # Process the frame to obtain a masked depth map
    depth_map = process_frame(frame, midas, device, transform, mask)
    
    # Create a mesh from the depth data
    mesh_data = create_mesh_from_depth(depth_map, mask, fx, fy, cx, cy, step=8)
    if mesh_data[0] is None:
        return
    vertices, tex_coords, indices = mesh_data
    
    # Update OpenGL buffers
    update_buffers(VBO, TBO, EBO, vertices, tex_coords, indices)
    update_texture(texture_id, texture_img)
    
    # Clear buffers with color
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
    # Bind shader and set uniforms
    glUseProgram(shader_program)
    
    # Set texture unit and bind texture
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    loc = glGetUniformLocation(shader_program, "tex")
    glUniform1i(loc, 0)  # Set texture uniform
    
    # Draw mesh
    glEnableVertexAttribArray(0)
    glEnableVertexAttribArray(1)
    
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
    
    glBindBuffer(GL_ARRAY_BUFFER, TBO)
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, None)
    
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
    glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)
    
    # Debug info
    print(f"Vertices: {len(vertices)}, Indices: {len(indices)}")
    print(f"Texture size: {texture_img.shape}")
    
    # Cleanup
    glDisableVertexAttribArray(1)
    glDisableVertexAttribArray(0)
    glUseProgram(0)
    
    # Ensure frame is displayed
    pygame.display.flip()

if __name__ == '__main__':
    main()
