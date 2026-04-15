"""背景场: u0 汉克尔格林函数, tau0 及其解析导数"""
import numpy as np
import torch
from scipy.special import hankel1


class BackgroundField:
    def __init__(self, grid, medium, source, omega):
        s0 = medium.s0
        r_safe = source.safe_distance
        self.tau0 = s0 * source.distance
        self.grad_tau0_x = s0 * (grid.xx - source.x_s) / r_safe
        self.grad_tau0_y = s0 * (grid.yy - source.y_s) / r_safe
        self.lap_tau0 = s0 / r_safe
        k0 = omega * s0
        self.u0 = (1j / 4.0) * hankel1(0, k0 * r_safe)
        self.s0 = s0
        self.omega = omega
        self.k0 = k0
        self.tau0_t = torch.from_numpy(self.tau0)
        self.grad_tau0_x_t = torch.from_numpy(self.grad_tau0_x)
        self.grad_tau0_y_t = torch.from_numpy(self.grad_tau0_y)
        self.lap_tau0_t = torch.from_numpy(self.lap_tau0)
        self.u0_t = torch.from_numpy(np.stack([self.u0.real, self.u0.imag]))
