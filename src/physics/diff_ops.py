"""固定卷积核微分引擎 (D2)"""
import torch
import torch.nn.functional as F


class DiffOps:
    """Fixed-kernel finite difference operators using F.conv2d."""

    def __init__(self, h):
        self.h = h
        # Second order: [1, -2, 1] / h^2
        self.kernel_xx = torch.tensor([[[[1., -2., 1.]]]], dtype=torch.float64) / h**2
        self.kernel_yy = torch.tensor([[[[1.], [-2.], [1.]]]], dtype=torch.float64) / h**2
        # First order: [-1, 0, 1] / (2h)
        self.kernel_x = torch.tensor([[[[-1., 0., 1.]]]], dtype=torch.float64) / (2 * h)
        self.kernel_y = torch.tensor([[[[-1.], [0.], [1.]]]], dtype=torch.float64) / (2 * h)

    def diff_x(self, u):
        """First derivative d/dx. Input [B,1,H,W] or [1,1,H,W]. Reflect pad."""
        u4d = self._ensure_4d(u)
        padded = F.pad(u4d, [1, 1, 0, 0], mode='reflect')
        return F.conv2d(padded, self.kernel_x.to(u4d.device, u4d.dtype))

    def diff_y(self, u):
        """First derivative d/dy. Reflect pad."""
        u4d = self._ensure_4d(u)
        padded = F.pad(u4d, [0, 0, 1, 1], mode='reflect')
        return F.conv2d(padded, self.kernel_y.to(u4d.device, u4d.dtype))

    def diff_xx(self, u):
        """Second derivative d^2/dx^2. Reflect pad."""
        u4d = self._ensure_4d(u)
        padded = F.pad(u4d, [1, 1, 0, 0], mode='reflect')
        return F.conv2d(padded, self.kernel_xx.to(u4d.device, u4d.dtype))

    def diff_yy(self, u):
        """Second derivative d^2/dy^2. Reflect pad."""
        u4d = self._ensure_4d(u)
        padded = F.pad(u4d, [0, 0, 1, 1], mode='reflect')
        return F.conv2d(padded, self.kernel_yy.to(u4d.device, u4d.dtype))

    def laplacian(self, u):
        """Standard Laplacian: d^2u/dx^2 + d^2u/dy^2."""
        return self.diff_xx(u) + self.diff_yy(u)

    def _ensure_4d(self, u):
        if u.dim() == 2:
            return u.unsqueeze(0).unsqueeze(0)
        elif u.dim() == 3:
            return u.unsqueeze(0)
        return u
