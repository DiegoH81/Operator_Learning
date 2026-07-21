import numpy as np

eps = 1e-6

class Equation:
    def __init__ (self, in_function_np, in_function_torch, in_domain, in_uses_scale = False):
        self.f_np = in_function_np
        self.f_torch = in_function_torch
        self.scale = self.compute_scale(in_domain)
        self.uses_scale = in_uses_scale
    
    def compute_scale(self, in_domain):
        f_vals = self.f_np(in_domain[:, 0:1], in_domain[:, 1:2])
        s = np.sqrt(np.mean(f_vals ** 2))
        return s if s > eps else 1.0
    
    def residual(self, du_dxx, du_dyy, y_pde, in_epoch_scale):
        if (self.uses_scale and in_epoch_scale):
            return (du_dxx + du_dyy - self.f_torch(y_pde[:, 0:1], y_pde[:, 1:2])) / self.scale
        else:
            return (du_dxx + du_dyy - self.f_torch(y_pde[:, 0:1], y_pde[:, 1:2]))