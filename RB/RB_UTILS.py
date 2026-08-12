import numpy as np
import scipy.sparse as sp

def get_initial_matrix (in_data, grid_size):
    active_grid = (grid_size - 2) * (grid_size - 2)
    
    NUM_snapshots = in_data.shape[0]
    data_without_borders = in_data[:, 1:-1, 1:-1]
    to_return = data_without_borders.reshape(NUM_snapshots, -1).T
    
    return to_return

def get_new_base(in_initial_mat, n_nodes):
    U, S_vals, Vt = np.linalg.svd(in_initial_mat, full_matrices = False)
    
    return U[:, : n_nodes]

def get_A_rb(grid_size, new_base):
    h = 1.0 / (grid_size - 1)
    valid_size = grid_size - 2
    A_II = np.zeros((valid_size * valid_size, valid_size * valid_size))
    
    for i in range (valid_size):
        for j in range (valid_size):
            idx = valid_size * i + j
            
            A_II[idx][idx] = -4.0/ h**2
            if (j > 0):
                A_II[idx, idx - 1] = 1.0 / h**2
            if (j < (valid_size - 1)):
                A_II[idx, idx + 1] = 1.0 / h**2
            if (i > 0):
                A_II[idx, idx - valid_size] = 1.0 / h**2
            if (i < (valid_size - 1)):
                A_II[idx, idx + valid_size] = 1.0 / h**2
    
    return new_base.T @ A_II @ new_base

def get_F_rb(border, source, grid_size, new_base):
    h = 1.0 / (grid_size - 1)
    valid_size = grid_size - 2
    
    F_interior = np.zeros(valid_size * valid_size)
    
    for i in range(valid_size):
        for j in range(valid_size):
            idx = valid_size * i + j
            
            F_interior[idx] += source[i + 1, j + 1]
            
            if j == 0:
                F_interior[idx] -= border[i + 1, 0] / (h ** 2)
            if j == (valid_size - 1):
                F_interior[idx] -= border[i + 1, grid_size - 1] / (h ** 2)
            if i == 0:
                F_interior[idx] -= border[0, j + 1] / (h ** 2)
            if i == (valid_size - 1):
                F_interior[idx] -= border[grid_size - 1, j + 1] / (h ** 2)
    return new_base.T @ F_interior

def get_F_rb_batch(border_data_grids, source_grid, grid_size, new_base):
    n_samples = border_data_grids.shape[0]
    valid_size = grid_size - 2
    h = 1.0 / (grid_size - 1)
    
    F_interior = np.zeros((n_samples, valid_size * valid_size))
    
    
    src_flat = source_grid[1:-1, 1:-1].ravel()
    F_interior += src_flat[None, :]
        
    for i in range(valid_size):
        for j in range(valid_size):
            idx = valid_size * i + j

            if j == 0:
                F_interior[:, idx] -= border_data_grids[:, i + 1, 0] / (h ** 2)
            if j == (valid_size - 1):
                F_interior[:, idx] -= border_data_grids[:, i + 1, grid_size - 1] / (h ** 2)
            if i == 0:
                F_interior[:, idx] -= border_data_grids[:, 0, j + 1] / (h ** 2)
            if i == (valid_size - 1):
                F_interior[:, idx] -= border_data_grids[:, grid_size - 1, j + 1] / (h ** 2)
                
    return F_interior @ new_base