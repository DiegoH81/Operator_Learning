import torch
import numpy as np

def sin_frontier(in_matrix, in_x, in_random_A, in_random_B):
    in_matrix[0, :] = np.sin(np.pi * in_x) * in_random_A
    in_matrix[-1, :] = np.sin(np.pi * in_x) * in_random_B
    in_matrix[:, 0] = 0.0
    in_matrix[:, -1] = 0.0
    

def laplace_numerical_solutions(n_samples, grid_size, f_frontier, f_source, max_iters = 5000):

    border_data = []
    interior_data = []

    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)

    X, Y = np.meshgrid(x, y)
    step = x[1] - x[0]
    #print ('X',X)
    #print ('Y',Y)
    
    fun_matrix = f_source(X, Y)
    for _ in range (n_samples):
        matrix = np.zeros((grid_size, grid_size))

        A = np.random.rand()
        B = np.random.rand()
        f_frontier(matrix, x, A, B)
        
        for _ in range (max_iters):
            new_matrix = matrix.copy()

            new_matrix[1:-1, 1:-1] = 0.25 * (
                matrix[1:-1, 2:] + # Right
                matrix[1:-1, :-2] + # Left
                matrix[2:, 1:-1] + # Up
                matrix[:-2,  1:-1] - # Down
                step**2 * fun_matrix[1:-1, 1:-1]
            )

            # Frontiers
            f_frontier(new_matrix, x, A, B)
            
            matrix = new_matrix

        # Top, down, left, right
        border_data.append(np.concatenate([matrix[0, :], matrix[-1, :], matrix[:, 0], matrix[:, -1]]))

        interior_data.append(matrix)

    return np.array(border_data), np.array(interior_data)

def fun_zeros(x, y):
    return np.zeros_like(x)

def fun_zeros_torch(x, y):
    return torch.zeros_like(x)