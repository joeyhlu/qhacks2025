# opengl_renderer.py

import pygame
from pygame.locals import DOUBLEBUF, OPENGL, QUIT
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import cv2
from math import tan, radians

def init_pygame_opengl(width, height):
    """Initialize Pygame and OpenGL context"""
    pygame.init()
    display = pygame.display.set_mode((width, height), pygame.DOUBLEBUF | pygame.OPENGL)
    pygame.display.set_caption("3D Object Viewer")
    
    # Enable necessary OpenGL features
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_TEXTURE_2D)
    glClearColor(0.2, 0.2, 0.2, 1.0)  # Dark gray background
    
    return display

def create_buffers():
    VBO = glGenBuffers(1)
    TBO = glGenBuffers(1)
    EBO = glGenBuffers(1)
    return VBO, TBO, EBO

def update_buffers(VBO, TBO, EBO, vertices, tex_coords, indices):
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_DYNAMIC_DRAW)
    glBindBuffer(GL_ARRAY_BUFFER, TBO)
    glBufferData(GL_ARRAY_BUFFER, tex_coords.nbytes, tex_coords, GL_DYNAMIC_DRAW)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_DYNAMIC_DRAW)

def create_texture(width, height):
    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    # Allocate empty texture storage
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
    return texture_id

def update_texture(texture_id, frame):
    glBindTexture(GL_TEXTURE_2D, texture_id)
    # Convert the frame to RGB and flip vertically (to match OpenGL coordinates)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb = np.flipud(frame_rgb)
    frame_data = np.ascontiguousarray(frame_rgb, dtype=np.uint8)
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, frame_rgb.shape[1], frame_rgb.shape[0],
                    GL_RGB, GL_UNSIGNED_BYTE, frame_data)

def compile_shaders(vertex_shader_code, fragment_shader_code):
    shader_program = compileProgram(
        compileShader(vertex_shader_code, GL_VERTEX_SHADER),
        compileShader(fragment_shader_code, GL_FRAGMENT_SHADER)
    )
    return shader_program

def create_perspective_matrix(fov_y, aspect, near, far):
    """Create a perspective projection matrix"""
    f = 1.0 / tan(radians(fov_y) / 2)
    return np.array([
        [f/aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far+near)/(near-far), (2*far*near)/(near-far)],
        [0, 0, -1, 0]
    ], dtype=np.float32)

def create_lookat_matrix(eye, target, up):
    """Create a look-at view matrix"""
    forward = np.array(target) - np.array(eye)
    forward = forward / np.linalg.norm(forward)
    
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    
    new_up = np.cross(right, forward)
    
    rotation = np.array([
        [right[0], right[1], right[2], 0],
        [new_up[0], new_up[1], new_up[2], 0],
        [-forward[0], -forward[1], -forward[2], 0],
        [0, 0, 0, 1]
    ])
    
    translation = np.array([
        [1, 0, 0, -eye[0]],
        [0, 1, 0, -eye[1]],
        [0, 0, 1, -eye[2]],
        [0, 0, 0, 1]
    ])
    
    return rotation @ translation

def set_uniform_matrices(shader_program, width, height):
    """Set the projection and modelview matrices as uniforms"""
    glUseProgram(shader_program)
    proj_loc = glGetUniformLocation(shader_program, "projection")
    modelview_loc = glGetUniformLocation(shader_program, "modelview")

    # Create perspective projection matrix
    projection_matrix = create_perspective_matrix(45, width/height, 0.1, 100.0)
    
    # Create modelview matrix (camera looking at origin from z=2)
    eye = np.array([0, 0, 2.0])
    target = np.array([0, 0, 0])
    up = np.array([0, 1, 0])
    modelview_matrix = create_lookat_matrix(eye, target, up)

    # Set matrices as uniforms
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection_matrix)
    glUniformMatrix4fv(modelview_loc, 1, GL_FALSE, modelview_matrix)
