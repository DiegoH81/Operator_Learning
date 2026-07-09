import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


class LaplaceEquation(nn.Module):
    def __init__(self, input_branch, input_trunk, out_dim):
        super().__init__()
        self.branch = nn.Sequential(nn.Linear(input_branch, 128),
                                    nn.Tanh(),
                                    nn.Linear(128, 128),
                                    nn.Tanh(),
                                    nn.Linear(128, out_dim))
        
        self.trunk = nn.Sequential( nn.Linear(input_trunk, 128),
                                    nn.Tanh(),
                                    nn.Linear(128, 128),
                                    nn.Tanh(),
                                    nn.Linear(128, out_dim) )
        
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, u, y):
        return torch.sum(self.branch(u) * self.trunk(y), dim=1, keepdim=True) + self.bias



def laplace_numerical_solutions(n_samples, grid_size):

    border_data = []
    interior_data = []

    x = np.linspace(0, 1, grid_size)
    for _ in range (n_samples):
        u = np.zeros((grid_size, grid_size))

        A = np.random.rand()
        B = np.random.rand()

        u[0, :] = np.sin(np.pi * x) * A
        u[-1, :] = np.sin(np.pi * x) * B
        u[:, 0] = 0.0
        u[:, -1] = 0.0


        
        for _ in range (5000):
            u_new = u.copy()

            u_new[1:-1, 1:-1] = 0.25 * (
                u[1:-1, 2:]   +
                u[1:-1, :-2]  +
                u[2:,   1:-1] +
                u[:-2,  1:-1]  
            )

            # Frontiers (again)
            u_new[0, :]  = A * np.sin(np.pi * x)
            u_new[-1, :] = B * np.sin(np.pi * x)
            u_new[:, 0]  = 0.0
            u_new[:, -1] = 0.0
            u = u_new

        border_data.append(np.concatenate([u[0, :], u[-1, :]]))

        interior_data.append(u)


    return np.array(border_data), np.array(interior_data)


def loss_function(model,
                  
                   u_pde, y_pde,
                   u_data, y_data, real_data,
                   u_bc, y_bc, real_bc,

                   w_pde, w_data, w_border,
                   current_epoch):

    # Loss PDE
    y_pde = y_pde.requires_grad_(True)
    u_pred = model(u_pde, y_pde)

    grads = torch.autograd.grad(u_pred, y_pde,
                                grad_outputs=torch.ones_like(u_pred),
                                create_graph=True)[0]
    du_dx  = grads[:, 0:1]
    du_dy  = grads[:, 1:2]
    du_dxx = torch.autograd.grad(du_dx, y_pde,
                                 grad_outputs=torch.ones_like(du_dx),
                                 create_graph=True)[0][:, 0:1]
    du_dyy = torch.autograd.grad(du_dy, y_pde,
                                 grad_outputs=torch.ones_like(du_dy),
                                 create_graph=True)[0][:, 1:2]

    l_pde = torch.mean( (du_dxx + du_dyy)**2 )

    # Loss DATA
    l_data = torch.mean((model(u_data, y_data) - real_data)**2)


    # Loss Border
    l_border = torch.mean((model(u_bc, y_bc) - real_bc)**2)

    if (current_epoch < 10000):
        loss = w_data * l_data
    else:
        loss = w_pde * l_pde + w_data * l_data + w_border * l_border
        
    return loss


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    

    out_dim = 256

    grid_size = 32
    num_scenarios = 1000



    batch_size = 128
    n_col = 100
    n_data = 200


    lr = 1e-3
    n_epochs = 40000

    w_pde, w_data, w_border = 5.0, 10.0, 5.0

    torch.manual_seed(0)
    np.random.seed(0)


    # Lambda to tensor
    to_t = lambda a: torch.tensor(a, dtype=torch.float32).to(device)

    # Data generation
    print("Generating data")
    border_data, interior_data = laplace_numerical_solutions(num_scenarios, grid_size)


    # Coords    
    x_coords = np.linspace(0, 1, grid_size)
    y_coords = np.linspace(0, 1, grid_size)
    
    X, Y = np.meshgrid(x_coords, y_coords)
    xy_grid = np.stack([X.ravel(), Y.ravel()], axis = 1)

    
    sol_flat = interior_data.reshape(num_scenarios, -1) # (ARRAY DE DATA) * 500

    # TRAINING
    model = LaplaceEquation(input_branch = 2 * grid_size, input_trunk = 2, out_dim = out_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr = lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = n_epochs, eta_min = 1e-5)


    
    print("\nStarting training")
    for epoch in range(1, n_epochs + 1):

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
        real_dat_np = sol_flat[np.ix_(idx_selected, data_idx)] # QUe escenarios, que puntos quiero


        u_data_np = np.repeat(u_batch, n_data, axis=0)
        y_data_np = np.tile(y_dat_np, (batch_size, 1))

        real_data_np = real_dat_np.ravel()
        real_data_np = real_data_np.reshape(-1, 1)

        u_data_tensor = to_t(u_data_np)
        y_data_tensor = to_t(y_data_np)
        real_data_tensor = to_t(real_data_np)

        # Border conditions
        x_bc = np.linspace(0, 1, grid_size)
        y_bot_np = np.stack([x_bc, np.zeros(grid_size)], axis = 1)
        y_top_np = np.stack([x_bc, np.ones(grid_size)], axis = 1)

        y_bc_np  = np.concatenate([y_bot_np, y_top_np])              

        u_bc_np   = np.repeat(u_batch, 2 * grid_size, axis=0)
        y_bc_tiled = np.tile(y_bc_np, (batch_size, 1))

        
        real_bc_np = u_batch.ravel().reshape(-1, 1)  # (batch * 64, 1)

        u_bc_tensor    = to_t(u_bc_np)
        y_bc_tensor    = to_t(y_bc_tiled)
        real_bc_tensor = to_t(real_bc_np)


        loss = loss_function(model,
                             u_pde_tensor, y_pde_tensor,
                             u_data_tensor, y_data_tensor, real_data_tensor,
                             u_bc_tensor, y_bc_tensor, real_bc_tensor,
                             w_pde, w_data, w_border, epoch)

        optimizer.zero_grad()
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if epoch % 500 == 0:
            print(f"Epoch {epoch}")


    # Evaluation with new scenario
    np.random.seed(99)
    border_test, interior_test = laplace_numerical_solutions(1, grid_size)
    u0_test  = border_test[0]
    sol_test = interior_test[0]

    model.eval()
    with torch.no_grad():
        u0_rep = to_t(np.tile(u0_test, (grid_size ** 2, 1)))
        xy_t   = to_t(xy_grid)
        pred   = model(u0_rep, xy_t).cpu().numpy().reshape(grid_size, grid_size)

    # Plots
    fig, axes = plt.subplots(1, 3, figsize = (18, 4))
 
    vmin = min(sol_test.min(), pred.min())
    vmax = max(sol_test.max(), pred.max())
 
    im0 = axes[0].imshow(sol_test, origin='lower', extent=[0,1,0,1],
                         vmin=vmin, vmax=vmax, cmap='RdBu_r')
    axes[0].set_title("Ground truth")
    plt.colorbar(im0, ax=axes[0])
 
    im1 = axes[1].imshow(pred, origin='lower', extent=[0,1,0,1],
                         vmin=vmin, vmax=vmax, cmap='RdBu_r')
    axes[1].set_title("DeepONet")
    plt.colorbar(im1, ax=axes[1])
 
    err = np.abs(sol_test - pred)
    im2 = axes[2].imshow(err, origin='lower', extent=[0,1,0,1], cmap='hot_r')
    axes[2].set_title(f"Error (max={err.max():.3f})")
    plt.colorbar(im2, ax=axes[2])
 
 
    plt.suptitle("PI-DeepONet — Laplace EQ", fontsize=13)
    plt.tight_layout()
    plt.savefig("laplace_result.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Guardado: laplace_result.png")