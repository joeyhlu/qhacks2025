# main.py

import cv2
import numpy as np
import pygame
from pygame.locals import QUIT
import torch

from project.depth_estimator import load_midas_model, process_frame
from project.segmentation import segment_frame
from project.mesh_generation import create_mesh_from_depth
from project.opengl_renderer import (init_pygame_opengl, create_buffers, update_buffers,
                             create_texture, update_texture, compile_shaders, set_uniform_matrices)
from project.shaders import vertex_shader_code, fragment_shader_code

def main():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if not ret:
        print("Unable to capture video")
        return
    height, width, _ = frame.shape

    # Example camera intrinsics (calibrate these for your camera)
    fx, fy = 500, 500
    cx, cy = width / 2, height / 2

    # Initialize MiDaS model for depth estimation
    midas, device, transform = load_midas_model()

    # Initialize pygame and OpenGL
    init_pygame_opengl(width, height)
    VBO, TBO, EBO = create_buffers()
    texture_id = create_texture(width, height)
    shader_program = compile_shaders(vertex_shader_code, fragment_shader_code)
    set_uniform_matrices(shader_program, width, height)

    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False

        ret, frame = cap.read()
        if not ret:
            break

        # Obtain segmentation mask for a specific object (e.g., 'person')
        mask = segment_frame(frame, target_class='person')
        # Process the frame to obtain a masked depth map
        depth_map = process_frame(frame, midas, device, transform, mask)
        # Create a mesh from the depth data using our camera intrinsics
        mesh_data = create_mesh_from_depth(depth_map, mask, fx, fy, cx, cy, step=8)
        if mesh_data[0] is None:
            continue
        vertices, tex_coords, indices = mesh_data

        # Update the OpenGL buffers with the new mesh data
        update_buffers(VBO, TBO, EBO, vertices, tex_coords, indices)
        # Update the video texture with the current frame
        update_texture(texture_id, frame)

        # Clear the screen and depth buffer
        from OpenGL.GL import glClear, GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Render the mesh with the video texture
        from OpenGL.GL import (glUseProgram, glEnableVertexAttribArray, glBindBuffer, 
                               glVertexAttribPointer, glBindTexture, glDrawElements, 
                               GL_ARRAY_BUFFER, GL_FLOAT, GL_FALSE, GL_ELEMENT_ARRAY_BUFFER,
                               GL_TRIANGLES, GL_UNSIGNED_INT, glDisableVertexAttribArray, 
                               GL_TEXTURE0, glActiveTexture)
        glUseProgram(shader_program)
        glEnableVertexAttribArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, VBO)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(1)
        glBindBuffer(GL_ARRAY_BUFFER, TBO)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, None)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)
        glDisableVertexAttribArray(0)
        glDisableVertexAttribArray(1)

        pygame.display.flip()
        clock.tick(30)  # Limit to 30 FPS

    cap.release()
    pygame.quit()

if __name__ == '__main__':
    main()
