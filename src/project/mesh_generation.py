# mesh_generation.py

import numpy as np

def create_mesh_from_depth(depth_map, mask, fx, fy, cx, cy, step=8):
    h, w = depth_map.shape
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        return None, None, None  # No valid region found
    min_y, max_y = int(ys.min()), int(ys.max())
    min_x, max_x = int(xs.min()), int(xs.max())
    
    # Create a grid over the masked region with a given step size
    grid_ys = np.arange(min_y, max_y, step)
    grid_xs = np.arange(min_x, max_x, step)
    num_rows, num_cols = len(grid_ys), len(grid_xs)
    
    vertices = []
    tex_coords = []
    for i, y in enumerate(grid_ys):
        for j, x in enumerate(grid_xs):
            z = depth_map[y, x]
            # If the depth is invalid, assign a small default value.
            if z <= 0:
                z = 0.1
            # Back-project the pixel (x, y) into 3D using the pinhole model:
            X = (x - cx) * z / fx
            Y = (y - cy) * z / fy
            vertices.append([X, -Y, -z])  # Flip Y and Z as needed for OpenGL
            # Normalize texture coordinates to [0, 1]
            tex_coords.append([x / w, y / h])
    vertices = np.array(vertices, dtype=np.float32)
    tex_coords = np.array(tex_coords, dtype=np.float32)
    
    # Create triangle indices from the grid
    indices = []
    for i in range(num_rows - 1):
        for j in range(num_cols - 1):
            idx = i * num_cols + j
            indices.extend([
                idx, idx + 1, idx + num_cols,
                idx + 1, idx + num_cols + 1, idx + num_cols
            ])
    indices = np.array(indices, dtype=np.uint32)
    
    return vertices, tex_coords, indices
