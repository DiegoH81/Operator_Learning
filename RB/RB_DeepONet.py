import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time

import my_utils as utils
import RB_UTILS as RB_UTILS

class RBBranch(nn.Module):
    def __init__(self, input_dim, N, hidden_size=128):
        super().__init__()
        self.net = nn.Sequential( nn.Linear(input_dim, hidden_size), nn.Tanh(),
                                  nn.Linear(hidden_size, hidden_size), nn.Tanh(),
                                  nn.Linear(hidden_size, hidden_size), nn.Tanh(),
                                  nn.Linear(hidden_size, N) )

    def forward(self, kaug):
        return self.net(kaug)


def rb_residual_loss(c_var, A_rb, F_rb):
    pred = c_var @ A_rb.T
    r = pred - F_rb
    return torch.mean(torch.sum(r ** 2, dim=1))


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    np.random.seed(0)

    # Parameters
    grid_size = 32
    N_base = 12
    n_snapshots = 400
    n_train = 4000
    n_test = 200

    hidden_size = 128
    batch_size = 128
    lr = 2e-3
    n_epochs = 3000
    
    source_fn = utils.fun_benchmark2
    frontier = utils.sin_frontier
    
    t_start_offline = time.perf_counter()
    border_data, interior_data, fun_matrix = utils.laplace_numerical_solutions(n_snapshots, grid_size, frontier, source_fn)
    
    #RB
    print("Starting offline")
    initial = RB_UTILS.get_initial_matrix(interior_data, grid_size)
    new_base = RB_UTILS.get_new_base(initial, N_base)
    A_rb = RB_UTILS.get_A_rb(grid_size, new_base)
    all_F_rb = RB_UTILS.get_F_rb_batch(interior_data, fun_matrix, grid_size, new_base)
    print("End offline")
    
    time_offline = time.perf_counter() - t_start_offline
    
    
    # Tensor
    kaug_t = torch.tensor(border_data, dtype=torch.float32).to(device)
    all_F_rb_t = torch.tensor(all_F_rb, dtype=torch.float32).to(device)
    A_rb_t = torch.tensor(A_rb, dtype=torch.float32).to(device)

    # 
    input_dim = border_data.shape[1]
    model = RBBranch(input_dim = input_dim, N = N_base, hidden_size = hidden_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"Training on: {device}")
    t_start_train = time.perf_counter()
    for epoch in range(1, n_epochs + 1):
        model.train()
        
        # Batch sel
        idx = np.random.choice(n_snapshots, batch_size, replace = False)
        batch_kaug = kaug_t[idx]
        batch_F_rb = all_F_rb_t[idx]

        # 
        c_var = model(batch_kaug)
        loss = rb_residual_loss(c_var, A_rb_t, batch_F_rb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}/{n_epochs} Loss: {loss.item():.6e}")

    time_train = time.perf_counter() - t_start_train
    # Eval
    print("Evaluating")
    
    border_test, interior_test, _ = utils.laplace_numerical_solutions(n_test, grid_size, frontier, source_fn)
    kaug_test_t = torch.tensor(border_test, dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        c_pred = model(kaug_test_t).cpu().numpy()

    # Re-build
    valid_size = grid_size - 2
    u_pred_flat = c_pred @ new_base.T
    u_pred_interior = u_pred_flat.reshape(n_test, valid_size, valid_size)
    u_true_interior = interior_test[:, 1:-1, 1:-1]

    # Data error
    l2_errors = np.zeros(n_test)
    rmse_errors = np.zeros(n_test)
    max_errors = np.zeros(n_test)

    for i in range(n_test):
        diff = u_true_interior[i] - u_pred_interior[i]
        l2_errors[i] = np.linalg.norm(diff) / np.linalg.norm(u_true_interior[i])
        rmse_errors[i] = np.sqrt(np.mean(diff ** 2))
        max_errors[i] = np.max(np.abs(diff))

    print(f"\nTesting done in {n_test} escenarios (N_rb = {N_base}):")
    print(f"\tL2 rel error: {l2_errors.mean():.4e} +- {l2_errors.std():.4e}")
    print(f"\tRMSE: {rmse_errors.mean():.4e} +- {rmse_errors.std():.4e}")
    print(f"\tMax error: {max_errors.mean():.4e} +- {max_errors.std():.4e}")
    print(f"\tWorst case L2: {l2_errors.max():.4e} (scenario {l2_errors.argmax()})")
    print(f"\tOffline time and Datagen: {time_offline:8.4f} s")
    print(f"\tTraining neural network: {time_train:8.4f} s")

    # Selection
    order = np.argsort(l2_errors)
    scenarios_to_plot = list(order[:2]) + list(order[-2:])
    labels = ["Best #1", "Best #2", "Worst #1", "Worst #2"]

    fig, axes = plt.subplots(4, 3, figsize=(12, 16))

    for row, (idx, label) in enumerate(zip(scenarios_to_plot, labels)):
        true_grid = interior_test[idx].copy()
        
        pred_grid = true_grid.copy()
        pred_grid[1:-1, 1:-1] = u_pred_interior[idx]
        
        err_grid = np.zeros((grid_size, grid_size))
        err_grid[1:-1, 1:-1] = np.abs(pred_grid[1:-1, 1:-1] - true_grid[1:-1, 1:-1])

        vmin, vmax = true_grid.min(), true_grid.max()

        
        im0 = axes[row, 0].imshow(true_grid, origin="lower", vmin=vmin, vmax=vmax, cmap="viridis")
        axes[row, 0].set_title(f"{label} (esc. {idx}) - Real")
        plt.colorbar(im0, ax=axes[row, 0], fraction=0.046, pad=0.04)

        im1 = axes[row, 1].imshow(pred_grid, origin="lower", vmin=vmin, vmax=vmax, cmap="viridis")
        axes[row, 1].set_title(f"RB-Branch (L2={l2_errors[idx]:.4e})")
        plt.colorbar(im1, ax=axes[row, 1], fraction=0.046, pad=0.04)

        im2 = axes[row, 2].imshow(err_grid, origin="lower", cmap="inferno", vmin = 0.0, vmax = 0.05)
        axes[row, 2].set_title(f"Error abs (max={max_errors[idx]:.4e})")
        plt.colorbar(im2, ax=axes[row, 2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    path_name = "rb_test.png"
    plt.savefig(path_name, dpi=150)
    print(f"\nSaved plot at: {path_name}")
    plt.show()