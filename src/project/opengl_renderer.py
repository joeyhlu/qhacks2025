# opengl_renderer.py

import pygame
from pygame.locals import DOUBLEBUF, OPENGL, QUIT
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import cv2

def init_pygame_opengl(width, height):
    pygame.init()
    pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
    glViewport(0, 0, width, height)
    glEnable(GL_DEPTH_TEST)
    # Set up projection matrix using gluPerspective
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    from OpenGL.GLU import gluPerspective
    gluPerspective(45, width / height, 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)

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

def set_uniform_matrices(shader_program, width, height):
    glUseProgram(shader_program)
    proj_loc = glGetUniformLocation(shader_program, "projection")
    modelview_loc = glGetUniformLocation(shader_program, "modelview")
    import pygame
    # Create perspective projection and modelview matrices using pygame's math module
    projection_matrix = pygame.math.Matrix44.perspective_projection(45, width/height, 0.1, 100.0)
    modelview_matrix = pygame.math.Matrix44.look_at(
        pygame.math.Vector3(0, 0, 2.0),  # camera position
        pygame.math.Vector3(0, 0, 0),    # target
        pygame.math.Vector3(0, 1, 0)     # up vector
    )
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection_matrix)
    glUniformMatrix4fv(modelview_loc, 1, GL_FALSE, modelview_matrix)
