import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import my_utils as utils

class LaplaceEquation(nn.Module):
    def __init__(self, input_branch_dim, input_trunk_dim, out_dim):
        super().__init__()
        self.branch = nn.Sequential(nn.Linear(input_branch_dim, 256),
                                    nn.Tanh(),
                                    nn.Linear(256, 256),
                                    nn.Tanh(),
                                    nn.Linear(256, out_dim))
        
        self.trunk = nn.Sequential( nn.Linear(input_trunk_dim, 256),
                                    nn.Tanh(),
                                    nn.Linear(256, 256),
                                    nn.Tanh(),
                                    nn.Linear(256, out_dim) )
        
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, u, y):
        return torch.sum(self.branch(u) * self.trunk(y), dim = 1, keepdim = True) + self.bias


def loss_function(model,
                  
                   u_pde, y_pde,
                   u_data, y_data, real_data,
                   u_bc, y_bc, real_bc,

                   w_pde, w_data, w_border,
                   current_epoch, f_source_torch):

    # Loss PDE
    y_pde = y_pde.requires_grad_(True)
    u_pred = model(u_pde, y_pde)

    grads = torch.autograd.grad(u_pred, y_pde,
                                grad_outputs = torch.ones_like(u_pred),
                                create_graph = True)[0]
    du_dx  = grads[:, 0:1]
    du_dy  = grads[:, 1:2]
    du_dxx = torch.autograd.grad(du_dx, y_pde,
                                 grad_outputs = torch.ones_like(du_dx),
                                 create_graph = True)[0][:, 0:1]
    du_dyy = torch.autograd.grad(du_dy, y_pde,
                                 grad_outputs = torch.ones_like(du_dy),
                                 create_graph = True)[0][:, 1:2]

    
    
    if (current_epoch < 15000):
        l_pde = torch.mean( (du_dxx + du_dyy - f_source_torch(y_pde[:,0:1], y_pde[:, 1:2]))**2 )
    else:
        F_SCALE = 50 * np.pi**2
        l_pde = torch.mean( ((du_dxx + du_dyy - f_source_torch(y_pde[:,0:1], y_pde[:, 1:2])) / F_SCALE)**2 )
        

    # Loss DATA
    l_data = torch.mean((model(u_data, y_data) - real_data)**2)


    # Loss Border
    l_border = torch.mean((model(u_bc, y_bc) - real_bc)**2)

    if current_epoch < 15000:
        loss = w_pde * l_pde + w_data * l_data + w_border * l_border
    else:
        loss = w_pde * l_pde + w_data * l_data + w_border * 20 * l_border
        
    
    #if (current_epoch < 10000):
    #    loss = w_data * l_data + w_border * l_border
    #else:
    #    loss = w_pde * l_pde + w_data * l_data + w_border * l_border
    
    
    
    
    
    if (current_epoch % 500 == 0):
        print("loss_pde", l_pde.item())
        print("loss_data", l_data.item())
        print("loss_border", l_border.item())
        
    
        
    return loss


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Parameters
    output_dim = 256

    grid_size = 32
    num_scenarios = 1000

    batch_size = 128
    n_col = 200
    n_data = 500

    lr = 1e-3
    n_epochs = 25000

    w_pde, w_data, w_border = 8.0, 10.0, 10.0
    

    torch.manual_seed(0)
    np.random.seed(0)


    # Lambda to tensor
    to_t = lambda a: torch.tensor(a, dtype=torch.float32).to(device)

    # Data generation
    print("Generating data")
    border_data, interior_data = utils.laplace_numerical_solutions(num_scenarios, grid_size,
                                                                   utils.sin_frontier, utils.fun_benchmark2)

    # Coords    
    x_coords = np.linspace(0, 1, grid_size)
    y_coords = np.linspace(0, 1, grid_size)
    
    X, Y = np.meshgrid(x_coords, y_coords)
    xy_grid = np.stack([X.ravel(), Y.ravel()], axis = 1)

    
    sol_flat = interior_data.reshape(num_scenarios, -1) # (ARRAY DE DATA) * 500

    # TRAINING
    model = LaplaceEquation(input_branch_dim = 4 * grid_size, input_trunk_dim = 2, out_dim = output_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr = lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = n_epochs, eta_min = 1e-5)


    print("\nStarting training")
    for epoch in range(1, n_epochs + 1):

        if epoch == 15000:
            optimizer = optim.Adam(model.parameters(), lr=5e-4)  # LR fresco para fase 2
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10000, eta_min=1e-5)
        
        # Batch Data
        idx_selected = np.random.choice(num_scenarios, batch_size, replace = False)
        u_batch = border_data[idx_selected]

        # Collocation points for Batch 0 - 1
        col_pts_x_y = np.random.uniform(1e-3, 1 - 1e-3,(n_col, 2))

        u_pde_np = np.repeat(u_batch, n_col, axis = 0)
        y_pde_np = np.tile(col_pts_x_y, (batch_size, 1))


        u_pde_tensor = to_t(u_pde_np)
        y_pde_tensor = to_t(y_pde_np)


        # Data points for Batch
        data_idx  = np.random.choice(grid_size ** 2, n_data, replace = False)
        y_dat_np = xy_grid[data_idx]
        
        
        sub_matrix = sol_flat[idx_selected]
        real_dat_np = sub_matrix[:, data_idx]
        real_data_np = real_dat_np.ravel()
        real_data_np = real_data_np.reshape(-1, 1)

        u_data_np = np.repeat(u_batch, n_data, axis=0)
        y_data_np = np.tile(y_dat_np, (batch_size, 1))

        u_data_tensor = to_t(u_data_np)
        y_data_tensor = to_t(y_data_np)
        real_data_tensor = to_t(real_data_np)

        # Border conditions
        x_bc = np.linspace(0, 1, grid_size)
        y_bot_np = np.stack([x_bc, np.zeros(grid_size)], axis = 1)
        y_top_np = np.stack([x_bc, np.ones(grid_size)], axis = 1)
        
        y_left_np = np.stack([np.zeros(grid_size), x_bc], axis = 1)
        y_right_np = np.stack([np.ones(grid_size), x_bc], axis = 1)



        y_bc_np  = np.concatenate([y_bot_np, y_top_np, y_left_np, y_right_np])              

        u_bc_np   = np.repeat(u_batch, 4 * grid_size, axis=0)
        y_bc_tiled = np.tile(y_bc_np, (batch_size, 1))

        real_bc_np = u_batch.ravel().reshape(-1, 1)


        u_bc_tensor = to_t(u_bc_np)
        y_bc_tensor = to_t(y_bc_tiled)
        real_bc_tensor = to_t(real_bc_np)



        loss = loss_function(model,
                             u_pde_tensor, y_pde_tensor,
                             u_data_tensor, y_data_tensor, real_data_tensor,
                             u_bc_tensor, y_bc_tensor, real_bc_tensor,
                             w_pde, w_data, w_border, epoch,
                             utils.fun_benchmark2_torch)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if epoch % 500 == 0:
            print(f"Epoch {epoch}")


    #----------------------------------------------------------#
    # Evaluation
    np.random.seed(99)
    n_test = 100

    
    border_test, interior_test = utils.laplace_numerical_solutions(n_test, grid_size, utils.sin_frontier, utils.fun_benchmark2)

    model.eval()
    predictions = np.zeros((n_test, grid_size, grid_size))

    with torch.no_grad():
        xy_t = to_t(xy_grid)

        for i in range(n_test):
            u0_rep = to_t(np.tile(border_test[i], (grid_size ** 2, 1)))
            pred_i = model(u0_rep, xy_t).cpu().numpy().reshape(grid_size, grid_size)
            predictions[i] = pred_i

    l2_errors = []
    rmse_errors = []
    max_errors = []

    for i in range(n_test):
        true_i = interior_test[i]
        pred_i = predictions[i]

        l2_rel = np.linalg.norm(pred_i - true_i) / np.linalg.norm(true_i)
        rmse_error = np.sqrt(np.mean((pred_i - true_i)**2))
        max_err = np.abs(pred_i - true_i).max()

        l2_errors.append(l2_rel)
        max_errors.append(max_err)
        rmse_errors.append(rmse_error)

    l2_errors = np.array(l2_errors)
    max_errors = np.array(max_errors)
    rmse_errors = np.array(rmse_errors)

    print(f"Testing done in {n_test} scenarios:")
    print(f"  L2 error:  {l2_errors.mean():.4f} +- {l2_errors.std():.4f}")
    print(f"  RMSE:  {rmse_errors.mean():.4f} +- {rmse_errors.std():.4f}")
    print(f"  Max error: {max_errors.mean():.4f} +- {max_errors.std():.4f}")
    print(f"  Worst case(L2): {l2_errors.max():.4f} (scenario {l2_errors.argmax()})")
    
    
    order = np.argsort(l2_errors)
    best_idx = order[:2]
    worst_idx = order[-2:]

    scenarios_to_plot = list(best_idx) + list(worst_idx)
    labels = ["Best #1", "Best #2", "Worst #1", "Worst #2"]

    fig, axes = plt.subplots(4, 3, figsize=(12, 16))

    for row, (idx, label) in enumerate(zip(scenarios_to_plot, labels)):
        true_i = interior_test[idx]
        pred_i = predictions[idx]
        err_i  = np.abs(pred_i - true_i)

        vmin = min(true_i.min(), pred_i.min())
        vmax = max(true_i.max(), pred_i.max())

        im0 = axes[row, 0].imshow(true_i, origin="lower", vmin=vmin, vmax=vmax, cmap="viridis")
        axes[row, 0].set_title(f"{label} (esc. {idx}) - Real")
        plt.colorbar(im0, ax=axes[row, 0], fraction = 0.046)

        im1 = axes[row, 1].imshow(pred_i, origin="lower", vmin=vmin, vmax=vmax, cmap="viridis")
        axes[row, 1].set_title(f"Predicted (L2={l2_errors[idx]:.4f}), (RMSE = {rmse_errors[idx]:.4f})")
        plt.colorbar(im1, ax=axes[row, 1], fraction = 0.046)

        im2 = axes[row, 2].imshow(err_i, origin="lower", cmap="inferno")
        axes[row, 2].set_title(f"Error abs (max={max_errors[idx]:.4f})")
        plt.colorbar(im2, ax=axes[row, 2], fraction = 0.046)

    plt.tight_layout()
    plt.savefig("best_worst_scenarios.png", dpi = 150)
    plt.show()