import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import binary_erosion



class LaplaceEquation(nn.Module):
    def __init__(self, hidden=128, out_dim=128):
        super().__init__()
        self.branch = nn.Sequential(nn.Linear(3, hidden), nn.Tanh(),
                                     nn.Linear(hidden, hidden), nn.Tanh(),
                                     nn.Linear(hidden, out_dim))
        
        
        self.point_NET = nn.Sequential(nn.Linear(2, hidden), nn.Tanh(),
                                       nn.Linear(hidden, hidden), nn.Tanh(),
                                       nn.Linear(hidden, out_dim))
        self.xy_lift = nn.Linear(2, out_dim)
        
        self.trunk = nn.Sequential(nn.Linear(out_dim, hidden), nn.Tanh(),
                                    nn.Linear(hidden, hidden), nn.Tanh(),
                                    nn.Linear(hidden, out_dim))
        
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, bc_pts, bc_flag, query_xy):
        NUM_QUERY_POINTS = query_xy.shape[1]

        flags_DATA = bc_flag.unsqueeze(-1)
        flags_NUM = bc_flag.sum(1, keepdim=True).unsqueeze(-1)
        
        print("NORMAL")
        print(bc_flag)
        print("UNSQUEEZED")
        print(flags_DATA)
        print("FLAG_NUM")
        print(flags_NUM)
        
        geometry_encoder = self.point_NET(bc_pts[:,:,:2]) # Only coords
        geometry_encoder = (geometry_encoder * flags_DATA).sum(1, keepdim=True) / flags_NUM.clamp(min = 1e-6)

        branch_encoder = self.branch(bc_pts)
        branch_encoder = (branch_encoder * flags_DATA).amax(1, keepdim=True)

        xy_cond = self.xy_lift(query_xy) + geometry_encoder.repeat(1, NUM_QUERY_POINTS, 1)
        trunk_encoder = self.trunk(xy_cond)
        return (branch_encoder * trunk_encoder).sum(-1, keepdim=True) + self.bias


def make_random_blob_mask(grid_size, n_blob_pts=8, with_hole=True):
    x = np.linspace(0, 1, grid_size)
    X, Y = np.meshgrid(x, x)

    def blob(cx, cy, base_r, n_pts):
        angles = np.linspace(0, 2*np.pi, n_pts, endpoint=False)
        radii = np.clip(base_r * (1 + 0.3*np.random.randn(n_pts)), 0.15, 0.45)
        theta = np.arctan2(Y-cy, X-cx) % (2*np.pi)
        r_interp = np.interp(theta.ravel(), np.append(angles, angles[0]+2*np.pi),
                              np.append(radii, radii[0])).reshape(grid_size, grid_size)
        dist = np.sqrt((X-cx)**2 + (Y-cy)**2)
        return dist <= r_interp

    cx, cy = np.random.uniform(0.4, 0.6, 2)
    outer_r = np.random.uniform(0.3, 0.45)
    mask = blob(cx, cy, outer_r, n_blob_pts)

    if with_hole:
        hole_r = outer_r * np.random.uniform(0.25, 0.4)
        hole_cx = cx + np.random.uniform(-0.03, 0.03)
        hole_cy = cy + np.random.uniform(-0.03, 0.03)
        hole = blob(hole_cx, hole_cy, hole_r, n_pts=6)
        mask = mask & (~hole)

    return mask


def laplace_irregular_solutions(n_samples, grid_size, return_grid = False):
    all_masks, all_u_grid = [], []

    x = np.linspace(0, 1, grid_size)
    X, Y = np.meshgrid(x, x)

    all_coors = []
    all_u = []
    all_flag = []

    for _ in range(n_samples):
        mask = make_random_blob_mask(grid_size)
        interior = binary_erosion(mask)
        border = mask & (~interior)

        A, B = np.random.uniform(-1, 1, 2)
        u = np.zeros((grid_size, grid_size))
        u[border] = A*np.sin(np.pi*X[border]) + B*np.cos(np.pi*Y[border])

        for _ in range(3000):
            u_new = u.copy()
            u_new[1:-1,1:-1] = 0.25*(u[1:-1,2:]+u[1:-1,:-2]+u[2:,1:-1]+u[:-2,1:-1])
            u_new[border] = u[border]
            u_new[~mask] = 0.0
            u = u_new

        iy, ix = np.where(interior)
        by, bx = np.where(border)
        
        all_masks.append(mask)
        all_u_grid.append(u.copy())
        
        coors_i = np.concatenate([np.stack([X[iy,ix], Y[iy,ix]], -1),
                                   np.stack([X[by,bx], Y[by,bx]], -1)], 0)
        u_i     = np.concatenate([u[iy,ix], u[by,bx]])[:,None]
        flag_i  = np.concatenate([np.ones(len(iy)), np.zeros(len(by))])

        all_coors.append(coors_i); all_u.append(u_i); all_flag.append(flag_i)

    if return_grid:
        return all_coors, all_u, all_flag, all_masks, all_u_grid
    return all_coors, all_u, all_flag

def pack_dataset(all_coors, all_u, all_flag):
    datasize = len(all_coors)

    max_pde_nodes, max_bc_nodes = 0, 0
    for i in range(datasize):
        max_pde_nodes = max(max_pde_nodes, int(np.sum(all_flag[i]==1)))
        max_bc_nodes  = max(max_bc_nodes,  int(np.sum(all_flag[i]==0)))

    coorT, uT, flagT, parT, par_flagT = [], [], [], [], []
    for i in range(datasize):
        pde_idx = np.where(all_flag[i]==1)[0]
        bc_idx  = np.where(all_flag[i]==0)[0]
        n_pde, n_bc = len(pde_idx), len(bc_idx)

        coor_i = np.concatenate([all_coors[i][pde_idx], np.zeros((max_pde_nodes-n_pde,2)),
                                  all_coors[i][bc_idx],  np.zeros((max_bc_nodes-n_bc,2))], 0)
        u_i    = np.concatenate([all_u[i][pde_idx],      np.zeros((max_pde_nodes-n_pde,1)),
                                  all_u[i][bc_idx],       np.zeros((max_bc_nodes-n_bc,1))], 0)
        flag_i = np.concatenate([np.ones(n_pde), np.zeros(max_pde_nodes-n_pde),
                                  np.ones(n_bc),  np.zeros(max_bc_nodes-n_bc)])

        par_i = np.concatenate([all_coors[i][bc_idx], all_u[i][bc_idx]], -1)
        par_i = np.concatenate([par_i, np.zeros((max_bc_nodes-n_bc, 3))], 0)
        par_flag_i = np.concatenate([np.ones(n_bc), np.zeros(max_bc_nodes-n_bc)])

        coorT.append(coor_i); uT.append(u_i); flagT.append(flag_i)
        parT.append(par_i); par_flagT.append(par_flag_i)

    return (torch.tensor(np.stack(coorT), dtype=torch.float32),
            torch.tensor(np.stack(uT), dtype=torch.float32),
            torch.tensor(np.stack(flagT), dtype=torch.float32),
            torch.tensor(np.stack(parT), dtype=torch.float32),
            torch.tensor(np.stack(par_flagT), dtype=torch.float32),
            max_pde_nodes, max_bc_nodes)

def loss_function(model, bc_pts, bc_flag, query_pde, pde_flag, w_pde, w_bc):
    query_pde = query_pde.requires_grad_(True)
    u_pred = model(bc_pts, bc_flag, query_pde)

    grads = torch.autograd.grad(u_pred, query_pde, torch.ones_like(u_pred), create_graph=True)[0]
    du_dx, du_dy = grads[...,0:1], grads[...,1:2]
    du_dxx = torch.autograd.grad(du_dx, query_pde, torch.ones_like(du_dx), create_graph=True)[0][...,0:1]
    du_dyy = torch.autograd.grad(du_dy, query_pde, torch.ones_like(du_dy), create_graph=True)[0][...,1:2]

    residual = (du_dxx + du_dyy).squeeze(-1)
    l_pde = (residual**2 * pde_flag).sum() / pde_flag.sum().clamp(min=1)

    u_bc_pred = model(bc_pts, bc_flag, bc_pts[:,:,:2]).squeeze(-1)
    l_bc = ((u_bc_pred - bc_pts[:,:,2])**2 * bc_flag).sum() / bc_flag.sum().clamp(min=1)

    return w_pde*l_pde + w_bc*l_bc


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    

    out_dim = 256

    grid_size = 32
    num_scenarios = 1#000



    batch_size = 1#28
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
    all_coors, all_u, all_flag = laplace_irregular_solutions(num_scenarios, grid_size)
    coorT, uT, flagT, parT, par_flagT, max_pde, max_bc = pack_dataset(all_coors, all_u, all_flag)

    model = LaplaceEquation(hidden=128, out_dim=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-5)

    for epoch in range(1, n_epochs + 1):
        idx = np.random.choice(num_scenarios, batch_size, replace=False)

        bc_pts  = parT[idx].to(device)
        bc_flag = par_flagT[idx].to(device)

        query_pde = coorT[idx, :max_pde].to(device)
        pde_flag  = flagT[idx, :max_pde].to(device)

        loss = loss_function(model, bc_pts, bc_flag, query_pde, pde_flag, w_pde, w_border)

        optimizer.zero_grad(); loss.backward(); optimizer.step(); scheduler.step()
        if epoch % 500 == 0:
            print(f"Epoch {epoch}, loss {loss.item():.5f}")


    np.random.seed(99)
    all_coors_t, all_u_t, all_flag_t, all_masks_t, all_ugrid_t = laplace_irregular_solutions(
        1, grid_size, return_grid=True)

    mask = all_masks_t[0]
    sol_test = all_ugrid_t[0]

    coorT_t, uT_t, flagT_t, parT_t, par_flagT_t, max_pde_t, max_bc_t = pack_dataset( all_coors_t, all_u_t, all_flag_t)

    model.eval()
    with torch.no_grad():
        bc_pts_t  = parT_t.to(device)
        bc_flag_t = par_flagT_t.to(device)

        x = np.linspace(0, 1, grid_size)
        X, Y = np.meshgrid(x, x)
        xy_grid = np.stack([X.ravel(), Y.ravel()], -1)
        xy_t = to_t(xy_grid).unsqueeze(0)

        pred = model(bc_pts_t, bc_flag_t, xy_t).cpu().numpy().reshape(grid_size, grid_size)

    sol_masked  = np.where(mask, sol_test, np.nan)
    pred_masked = np.where(mask, pred, np.nan)

    # Plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    vmin = np.nanmin([np.nanmin(sol_masked), np.nanmin(pred_masked)])
    vmax = np.nanmax([np.nanmax(sol_masked), np.nanmax(pred_masked)])

    im0 = axes[0].imshow(sol_masked, origin='lower', extent=[0,1,0,1],
                         vmin=vmin, vmax=vmax, cmap='RdBu_r')
    axes[0].set_title("Ground truth")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(pred_masked, origin='lower', extent=[0,1,0,1],
                         vmin=vmin, vmax=vmax, cmap='RdBu_r')
    axes[1].set_title("DeepONet")
    plt.colorbar(im1, ax=axes[1])

    err = np.abs(sol_masked - pred_masked)
    im2 = axes[2].imshow(err, origin='lower', extent=[0,1,0,1], cmap='hot_r')
    axes[2].set_title(f"Error (max={np.nanmax(err):.3f})")
    plt.colorbar(im2, ax=axes[2])

    plt.suptitle("PI-DeepONet — Laplace EQ (dominio irregular)", fontsize=13)
    plt.tight_layout()
    plt.savefig("laplace_result.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Guardado: laplace_result.png")