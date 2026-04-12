"""PML 复坐标拉伸 (D6)"""
import numpy as np
import torch
import math


class PMLTensors:
    def __init__(self, grid, cfg, omega):
        pw = grid.pml_width
        h = grid.h
        nx_total = grid.nx_total
        ny_total = grid.ny_total
        L_pml = pw * h
        p = cfg.pml.power  # 2
        R0 = cfg.pml.R0
        c_max = 1.0 / (1.0 / cfg.medium.c_background)  # = c_background

        # sigma_max from reflection coefficient formula (D6e)
        sigma_max = -(p + 1) * c_max * math.log(R0) / (2 * L_pml)

        # Build 1D sigma profiles
        sigma_x_1d = np.zeros(nx_total, dtype=np.float64)
        sigma_y_1d = np.zeros(ny_total, dtype=np.float64)
        sigma_prime_x_1d = np.zeros(nx_total, dtype=np.float64)
        sigma_prime_y_1d = np.zeros(ny_total, dtype=np.float64)

        for j in range(pw):
            # Left PML: distance from inner boundary
            d = (pw - j) * h
            sigma_x_1d[j] = sigma_max * (d / L_pml) ** p
            sigma_prime_x_1d[j] = -p * sigma_max * d ** (p - 1) / L_pml ** p
        for j in range(pw + grid.nx, nx_total):
            # Right PML
            d = (j - pw - grid.nx + 1) * h
            sigma_x_1d[j] = sigma_max * (d / L_pml) ** p
            sigma_prime_x_1d[j] = p * sigma_max * d ** (p - 1) / L_pml ** p

        for i in range(pw):
            d = (pw - i) * h
            sigma_y_1d[i] = sigma_max * (d / L_pml) ** p
            sigma_prime_y_1d[i] = -p * sigma_max * d ** (p - 1) / L_pml ** p
        for i in range(pw + grid.ny, ny_total):
            d = (i - pw - grid.ny + 1) * h
            sigma_y_1d[i] = sigma_max * (d / L_pml) ** p
            sigma_prime_y_1d[i] = p * sigma_max * d ** (p - 1) / L_pml ** p

        # Complex stretching factors (D6b)
        gamma_x_1d = 1.0 + 1j * sigma_x_1d / omega
        gamma_y_1d = 1.0 + 1j * sigma_y_1d / omega
        gamma_prime_x_1d = 1j * sigma_prime_x_1d / omega  # analytical!
        gamma_prime_y_1d = 1j * sigma_prime_y_1d / omega

        # Broadcast to 2D: [ny_total, nx_total]
        gamma_x = np.broadcast_to(gamma_x_1d[np.newaxis, :], (ny_total, nx_total)).copy()
        gamma_y = np.broadcast_to(gamma_y_1d[:, np.newaxis], (ny_total, nx_total)).copy()
        gamma_prime_x = np.broadcast_to(gamma_prime_x_1d[np.newaxis, :], (ny_total, nx_total)).copy()
        gamma_prime_y = np.broadcast_to(gamma_prime_y_1d[:, np.newaxis], (ny_total, nx_total)).copy()

        # PML multiplier tensors (D6c): A = 1/gamma^2, B = gamma'/gamma^3
        self.A_x = 1.0 / gamma_x ** 2
        self.A_y = 1.0 / gamma_y ** 2
        self.B_x = gamma_prime_x / gamma_x ** 3
        self.B_y = gamma_prime_y / gamma_y ** 3

        self.gamma_x = gamma_x
        self.gamma_y = gamma_y
        self.sigma_max = sigma_max

        # Torch tensors (complex128)
        self.A_x_t = torch.from_numpy(self.A_x)
        self.A_y_t = torch.from_numpy(self.A_y)
        self.B_x_t = torch.from_numpy(self.B_x)
        self.B_y_t = torch.from_numpy(self.B_y)

    def pml_laplacian(self, U, diff_ops):
        """PML-stretched Laplacian (D6c):
        Laplacian_pml U = Ax * Kxx*U - Bx * Kx*U + Ay * Kyy*U - By * Ky*U
        U: [1,1,H,W] torch tensor (can be complex via dual channel)
        """
        Uxx = diff_ops.diff_xx(U)
        Ux = diff_ops.diff_x(U)
        Uyy = diff_ops.diff_yy(U)
        Uy = diff_ops.diff_y(U)

        Ax = self.A_x_t.to(U.device)
        Bx = self.B_x_t.to(U.device)
        Ay = self.A_y_t.to(U.device)
        By = self.B_y_t.to(U.device)

        return Ax * Uxx.squeeze() + Ay * Uyy.squeeze() - Bx * Ux.squeeze() - By * Uy.squeeze()
