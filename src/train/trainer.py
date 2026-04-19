"""
训练循环
=========

单样本 overfit 验证训练管线可行性。
后续扩展：多波数课程学习。
"""

import logging
import math

import torch
import numpy as np

from src.core import Grid2D, Medium2D, PointSource
from src.physics.background import BackgroundField
from src.physics.eikonal import EikonalSolver
from src.physics.tau_ops import TauDerivatives
from src.physics.diff_ops import DiffOps
from src.physics.pml import PMLTensors
from src.physics.rhs import compute_rhs, compute_loss_mask
from src.physics.residual import ResidualComputer
from src.models.nsno import NSNO2D
from src.train.losses import loss_data, loss_total
from src.train.supervision import load_reference_target, resolve_supervision_config

logger = logging.getLogger(__name__)


def resolve_device(requested_device=None):
    """Resolve an execution device while keeping CPU as the backward-compatible default."""
    if requested_device is None:
        return torch.device("cpu")

    if isinstance(requested_device, torch.device):
        device = requested_device
    else:
        requested = str(requested_device).strip().lower()
        if requested == "auto":
            requested = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(requested)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    if device.type == "cuda" and device.index is None:
        device = torch.device(f"cuda:{torch.cuda.current_device()}")

    return device


def build_network_input(grid, medium, source, bg, tau_d, omega):
    """构建网络输入张量 [1, 8, H, W]。

    通道定义 (D10):
        0: s(x) 实部 (慢度)
        1: s(x) 虚部 (=0, 慢度为实数)
        2: u₀ 实部
        3: u₀ 虚部
        4: ∇τ_x
        5: ∇τ_y
        6: Δτ
        7: ω (标量广播)
    """
    H, W = grid.ny_total, grid.nx_total

    channels = []
    # 0-1: 慢度实虚部
    channels.append(torch.from_numpy(medium.slowness).float())
    channels.append(torch.zeros(H, W, dtype=torch.float32))
    # 2-3: u₀ 实虚部
    channels.append(torch.from_numpy(bg.u0.real).float())
    channels.append(torch.from_numpy(bg.u0.imag).float())
    # 4-5: ∇τ
    channels.append(tau_d.grad_tau_x.float())
    channels.append(tau_d.grad_tau_y.float())
    # 6: Δτ
    channels.append(tau_d.lap_tau.float())
    # 7: ω 广播
    channels.append(torch.full((H, W), omega, dtype=torch.float32))

    # 处理可能的 NaN/Inf（来自边界）
    x = torch.stack(channels, dim=0).unsqueeze(0)  # [1, 8, H, W]
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x


class Trainer:
    """最小训练循环。"""

    def __init__(self, cfg, device=None):
        self.cfg = cfg
        omega = cfg.physics.omega
        requested_device = device
        if requested_device is None and hasattr(cfg.training, "device"):
            requested_device = cfg.training.device
        self.device = resolve_device(requested_device)

        # 构建预处理管线
        logger.info("构建预处理管线...")
        self.grid = Grid2D(cfg)
        self.medium = Medium2D(self.grid, cfg)
        self.source = PointSource(self.grid, cfg)
        self.bg = BackgroundField(self.grid, self.medium, self.source, omega)
        self.eik = EikonalSolver(self.grid, self.medium, self.source, self.bg, cfg)
        self.diff_ops = DiffOps(self.grid.h)
        self.tau_d = TauDerivatives(self.bg, self.eik, self.diff_ops)
        self.pml = PMLTensors(self.grid, cfg, omega, s0=self.medium.s0)
        self.rhs = compute_rhs(
            self.grid, self.medium, self.source, self.bg, self.eik, omega, cfg
        )
        self.loss_mask = compute_loss_mask(self.grid, self.source, cfg)
        self.omega = omega

        # 残差计算器
        self.residual_computer = ResidualComputer(
            self.grid, self.pml, self.tau_d, self.rhs,
            self.loss_mask, omega, self.diff_ops,
        ).to(self.device)

        # 网络输入
        self.net_input = build_network_input(
            self.grid, self.medium, self.source, self.bg, self.tau_d, omega
        ).to(self.device)

        # 模型：固定初始化，避免测试依赖外部全局 RNG 状态
        rng_state = torch.random.get_rng_state()
        torch.manual_seed(0)
        self.model = NSNO2D(cfg).to(self.device)
        torch.random.set_rng_state(rng_state)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.training.lr)

        supervision_cfg = resolve_supervision_config(cfg)
        self.lambda_pde = supervision_cfg["lambda_pde"]
        self.lambda_data = supervision_cfg["lambda_data"]
        self.supervision_enabled = False
        self.reference_path = supervision_cfg["reference_path"]
        self.reference_target = None
        self.loss_history_total = []
        self.loss_history_pde = []
        self.loss_history_data = []
        self.last_step_metrics = {}

        if supervision_cfg["active"]:
            self.reference_path, self.reference_target = load_reference_target(
                self.reference_path,
                expected_shape=(self.grid.ny_total, self.grid.nx_total),
                device=self.device,
            )
            self.supervision_enabled = True
        elif supervision_cfg["requested"]:
            logger.warning(
                "training.supervision.enabled=true but lambda_data=0; "
                "training remains PDE-only."
            )

        logger.info(
            "预处理完成。网格: %dx%d, ω=%.1f, device=%s, supervision=%s",
            self.grid.nx_total,
            self.grid.ny_total,
            omega,
            self.device,
            self.supervision_enabled,
        )

    def train(self, epochs=None):
        """执行训练循环。

        Args:
            epochs: 训练轮次，默认使用配置值。

        Returns:
            loss 历史列表。
        """
        if epochs is None:
            epochs = self.cfg.training.epochs

        self.loss_history_total = []
        self.loss_history_pde = []
        self.loss_history_data = []
        mode = "hybrid"
        if self.lambda_data == 0.0:
            mode = "pde-only"
        elif self.lambda_pde == 0.0:
            mode = "data-only"

        logger.info("开始训练: %d epochs, mode=%s", epochs, mode)

        for epoch in range(epochs):
            self.optimizer.zero_grad()

            # 前向
            A_scat = self.model(self.net_input)

            # 残差 + 损失
            result = self.residual_computer.compute(A_scat)
            loss_pde_val = result["loss_pde"]
            loss_data_val = (
                loss_data(A_scat, self.reference_target)
                if self.supervision_enabled
                else None
            )
            total = loss_total(
                loss_pde_val,
                loss_data_val,
                lambda_pde=self.lambda_pde,
                lambda_data=self.lambda_data,
            )

            # 反向
            total.backward()
            self.optimizer.step()

            loss_total_item = float(total.item())
            loss_pde_item = float(loss_pde_val.item())
            loss_data_item = float(loss_data_val.item()) if loss_data_val is not None else 0.0
            self.loss_history_total.append(loss_total_item)
            self.loss_history_pde.append(loss_pde_item)
            self.loss_history_data.append(loss_data_item)
            self.last_step_metrics = {
                "loss_total": loss_total_item,
                "loss_pde": loss_pde_item,
                "loss_data": loss_data_item,
            }

            if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0:
                logger.info(
                    "Epoch %d/%d, loss_total=%.6e, loss_pde=%.6e, loss_data=%.6e",
                    epoch + 1,
                    epochs,
                    loss_total_item,
                    loss_pde_item,
                    loss_data_item,
                )

        logger.info("训练完成。最终 loss_total=%.6e", self.loss_history_total[-1])
        return list(self.loss_history_total)

    def reconstruct_wavefield(self):
        """重构全波场 u_total = u₀ + Â·exp[i(ωτ+π/4)]。

        Returns:
            complex128 numpy [ny_total, nx_total]。
        """
        with torch.no_grad():
            A_scat = self.model(self.net_input)

        A_r = A_scat[0, 0].detach().cpu().numpy().astype(np.float64)
        A_i = A_scat[0, 1].detach().cpu().numpy().astype(np.float64)
        A_complex = A_r + 1j * A_i

        # 安全 tau
        tau_safe = np.where(np.isfinite(self.eik.tau), self.eik.tau, 0.0)
        phase_restorer = np.exp(1j * (self.omega * tau_safe + math.pi / 4.0))

        u_total = self.bg.u0 + A_complex * phase_restorer
        return u_total
