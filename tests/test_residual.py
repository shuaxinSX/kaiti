"""
tests/test_residual.py — PDE 残差前向检查
"""

from pathlib import Path

import torch
import numpy as np
import pytest

from src.config import load_config
from src.core import Grid2D, Medium2D, PointSource
from src.physics.background import BackgroundField
from src.physics.eikonal import EikonalSolver
from src.physics.tau_ops import TauDerivatives
from src.physics.diff_ops import DiffOps
from src.physics.pml import PMLTensors
from src.physics.rhs import compute_rhs, compute_loss_mask
from src.physics.residual import ResidualComputer
from src.train.trainer import Trainer
from src.eval.reference_solver import solve_reference_scattering


@pytest.fixture
def cfg():
    base = Path(__file__).parent.parent / "configs" / "base.yaml"
    debug = Path(__file__).parent.parent / "configs" / "debug.yaml"
    return load_config(base, debug)


@pytest.fixture
def cfg_lens():
    base = Path(__file__).parent.parent / "configs" / "base.yaml"
    debug = Path(__file__).parent.parent / "configs" / "debug.yaml"
    cfg = load_config(base, debug)
    cfg.medium.velocity_model = "smooth_lens"
    return cfg


def _build_pipeline(cfg, lap_tau_mode="mixed_legacy"):
    """构建完整预处理管线。"""
    grid = Grid2D(cfg)
    medium = Medium2D(grid, cfg)
    source = PointSource(grid, cfg)
    omega = cfg.physics.omega
    bg = BackgroundField(grid, medium, source, omega)
    eik = EikonalSolver(grid, medium, source, bg, cfg)
    diff_ops = DiffOps(grid.h)
    tau_d = TauDerivatives(bg, eik, diff_ops)
    pml = PMLTensors(grid, cfg, omega, s0=medium.s0)
    rhs = compute_rhs(grid, medium, source, bg, eik, omega)
    mask = compute_loss_mask(grid, source, cfg)
    rc = ResidualComputer(
        grid, pml, tau_d, rhs, mask, omega, diff_ops, lap_tau_mode=lap_tau_mode
    )
    return rc, grid


def _build_trainer(cfg, lap_tau_mode="mixed_legacy"):
    trainer = Trainer(cfg, device="cpu")
    if lap_tau_mode != "mixed_legacy":
        trainer.residual_computer = ResidualComputer(
            trainer.grid,
            trainer.pml,
            trainer.tau_d,
            trainer.rhs,
            trainer.loss_mask,
            trainer.omega,
            trainer.diff_ops,
            lap_tau_mode=lap_tau_mode,
        ).to(trainer.device)
    return trainer


def _complex_to_dual(arr):
    stacked = np.stack([arr.real, arr.imag], axis=0).astype(np.float32)
    return torch.from_numpy(stacked).unsqueeze(0)


def _rel_l2(a, b, mask):
    diff = (a - b)[mask]
    ref = b[mask]
    numerator = np.linalg.norm(diff.ravel())
    denominator = max(np.linalg.norm(ref.ravel()), 1.0e-12)
    return float(numerator / denominator)


def _loss_on_solution(trainer, env):
    dual = _complex_to_dual(env).to(trainer.device)
    out = trainer.residual_computer.compute(dual)
    return float(out["loss_pde"].item())


@pytest.fixture
def pipeline(cfg):
    return _build_pipeline(cfg)


@pytest.fixture
def pipeline_lens(cfg_lens):
    return _build_pipeline(cfg_lens)


@pytest.fixture
def pipeline_lens_candidate(cfg_lens):
    return _build_pipeline(cfg_lens, lap_tau_mode="stretched_divergence")


class TestResidualComputer:
    def test_invalid_lap_tau_mode_raises(self, cfg):
        """不支持的 lap_tau_mode 应直接报错。"""
        with pytest.raises(ValueError, match="Unsupported lap_tau_mode"):
            _build_pipeline(cfg, lap_tau_mode="invalid_mode")

    def test_forward_runs(self, pipeline, cfg):
        """前向可跑通，无报错。"""
        rc, grid = pipeline
        A_scat = torch.zeros(1, 2, grid.ny_total, grid.nx_total)
        result = rc.compute(A_scat)
        assert 'loss_pde' in result
        assert 'residual_real' in result
        assert 'residual_imag' in result

    def test_no_nan_inf(self, pipeline, cfg):
        """残差不含 NaN/Inf。"""
        rc, grid = pipeline
        A_scat = torch.zeros(1, 2, grid.ny_total, grid.nx_total)
        result = rc.compute(A_scat)
        assert torch.all(torch.isfinite(result['residual_real']))
        assert torch.all(torch.isfinite(result['residual_imag']))
        assert torch.isfinite(result['loss_pde'])

    def test_zero_network_homogeneous(self, pipeline, cfg):
        """零初始化网络 + 均匀介质 → 残差 = -RHS/ω²。
        均匀介质 RHS=0，所以残差应为 0。"""
        rc, grid = pipeline
        A_scat = torch.zeros(1, 2, grid.ny_total, grid.nx_total)
        result = rc.compute(A_scat)
        # 均匀介质 RHS 全零，零网络无散射 → 残差应接近零
        assert result['loss_pde'].item() < 1e-10

    def test_loss_pde_non_negative(self, pipeline, cfg):
        """L_pde 非负。"""
        rc, grid = pipeline
        A_scat = torch.randn(1, 2, grid.ny_total, grid.nx_total) * 0.01
        result = rc.compute(A_scat)
        assert result['loss_pde'].item() >= 0.0

    def test_residual_shape(self, pipeline, cfg):
        """残差 shape 正确。"""
        rc, grid = pipeline
        A_scat = torch.zeros(1, 2, grid.ny_total, grid.nx_total)
        result = rc.compute(A_scat)
        assert result['residual_real'].shape == (grid.ny_total, grid.nx_total)
        assert result['residual_imag'].shape == (grid.ny_total, grid.nx_total)


class TestResidualPMLHeterogeneous:
    """PML 激活 + 非均匀介质下的残差测试。"""

    def test_default_mode_matches_explicit_legacy(self, cfg_lens, pipeline_lens):
        """默认路径应与显式 mixed_legacy 完全一致。"""
        rc_default, grid = pipeline_lens
        rc_legacy, _ = _build_pipeline(cfg_lens, lap_tau_mode="mixed_legacy")
        A_scat = torch.randn(1, 2, grid.ny_total, grid.nx_total) * 0.01
        result_default = rc_default.compute(A_scat)
        result_legacy = rc_legacy.compute(A_scat)
        assert rc_default.lap_tau_mode == "mixed_legacy"
        assert torch.allclose(rc_default.lap_tau_c, rc_legacy.lap_tau_c)
        assert torch.allclose(result_default["residual_real"], result_legacy["residual_real"])
        assert torch.allclose(result_default["residual_imag"], result_legacy["residual_imag"])
        assert result_default["loss_pde"].item() == pytest.approx(result_legacy["loss_pde"].item())

    def test_lens_forward_no_nan(self, pipeline_lens):
        """smooth_lens 介质 + PML 激活：残差不含 NaN/Inf。"""
        rc, grid = pipeline_lens
        A_scat = torch.randn(1, 2, grid.ny_total, grid.nx_total) * 0.01
        result = rc.compute(A_scat)
        assert torch.all(torch.isfinite(result['residual_real']))
        assert torch.all(torch.isfinite(result['residual_imag']))
        assert torch.isfinite(result['loss_pde'])

    def test_lens_nonzero_rhs(self, pipeline_lens):
        """smooth_lens 有非零 RHS（s != s0），零网络残差应非零。"""
        rc, grid = pipeline_lens
        A_scat = torch.zeros(1, 2, grid.ny_total, grid.nx_total)
        result = rc.compute(A_scat)
        # 非均匀介质 RHS 不全为零，所以零网络应产生非零残差
        assert result['loss_pde'].item() > 1e-15

    def test_lens_pml_region_complex_coefficients(self, pipeline_lens):
        """验证 PML 区域系数确实包含虚部。"""
        rc, _ = pipeline_lens
        # A_x 在 PML 区应有非零虚部
        assert rc.A_x.imag.abs().max().item() > 0.0
        assert rc.A_y.imag.abs().max().item() > 0.0

    def test_lens_gradient_flows(self, pipeline_lens):
        """非均匀介质下梯度能正常回传。"""
        rc, grid = pipeline_lens
        A_scat = torch.randn(1, 2, grid.ny_total, grid.nx_total) * 0.01
        A_scat.requires_grad_(True)
        result = rc.compute(A_scat)
        result['loss_pde'].backward()
        assert A_scat.grad is not None
        assert torch.all(torch.isfinite(A_scat.grad))

    def test_candidate_forward_no_nan(self, pipeline_lens_candidate):
        """stretched_divergence 候选模式前向稳定。"""
        rc, grid = pipeline_lens_candidate
        A_scat = torch.randn(1, 2, grid.ny_total, grid.nx_total) * 0.01
        result = rc.compute(A_scat)
        assert rc.lap_tau_mode == "stretched_divergence"
        assert torch.all(torch.isfinite(result["residual_real"]))
        assert torch.all(torch.isfinite(result["residual_imag"]))
        assert torch.isfinite(result["loss_pde"])

    def test_candidate_gradient_flows(self, pipeline_lens_candidate):
        """候选模式下梯度仍能回传。"""
        rc, grid = pipeline_lens_candidate
        A_scat = torch.randn(1, 2, grid.ny_total, grid.nx_total) * 0.01
        A_scat.requires_grad_(True)
        result = rc.compute(A_scat)
        result["loss_pde"].backward()
        assert A_scat.grad is not None
        assert torch.all(torch.isfinite(A_scat.grad))

    def test_candidate_lap_tau_physical_norm_stays_bounded(self, cfg_lens):
        """候选 Δ̃τ 在物理区仍保持与 legacy 同量级。"""
        legacy_rc, grid = _build_pipeline(cfg_lens, lap_tau_mode="mixed_legacy")
        candidate_rc, _ = _build_pipeline(cfg_lens, lap_tau_mode="stretched_divergence")
        phys_y, phys_x = grid.physical_slice()
        legacy_lap = legacy_rc.lap_tau_c.detach().cpu().numpy()[phys_y, phys_x]
        candidate_lap = candidate_rc.lap_tau_c.detach().cpu().numpy()[phys_y, phys_x]
        legacy_norm = np.linalg.norm(legacy_lap.ravel())
        candidate_norm = np.linalg.norm(candidate_lap.ravel())
        norm_ratio = candidate_norm / max(legacy_norm, 1.0e-12)
        assert np.isfinite(norm_ratio)
        assert 0.1 < norm_ratio < 2.0

    def test_reference_audit_legacy_vs_candidate_modes(self, cfg_lens):
        """reference baseline 下两种 mode 都自洽，且候选与 baseline 明显不同。"""
        legacy = _build_trainer(cfg_lens, lap_tau_mode="mixed_legacy")
        candidate = _build_trainer(cfg_lens, lap_tau_mode="stretched_divergence")

        ref_legacy = solve_reference_scattering(legacy)
        ref_candidate = solve_reference_scattering(candidate)

        pml_mask = legacy.grid.pml_mask()
        source_mask = legacy.loss_mask.astype(bool)
        eval_mask = (~pml_mask) & source_mask
        whole_mask = source_mask
        pml_only_mask = pml_mask

        rel_eval = _rel_l2(ref_candidate, ref_legacy, eval_mask)
        rel_whole = _rel_l2(ref_candidate, ref_legacy, whole_mask)
        rel_pml = _rel_l2(ref_candidate, ref_legacy, pml_only_mask)
        self_loss_legacy = _loss_on_solution(legacy, ref_legacy)
        self_loss_candidate = _loss_on_solution(candidate, ref_candidate)
        cross_loss_candidate_on_legacy = _loss_on_solution(candidate, ref_legacy)
        cross_loss_legacy_on_candidate = _loss_on_solution(legacy, ref_candidate)

        assert ref_legacy.shape == ref_candidate.shape
        assert np.all(np.isfinite(ref_legacy.real))
        assert np.all(np.isfinite(ref_legacy.imag))
        assert np.all(np.isfinite(ref_candidate.real))
        assert np.all(np.isfinite(ref_candidate.imag))
        assert np.isfinite(rel_eval)
        assert np.isfinite(rel_whole)
        assert np.isfinite(rel_pml)
        assert rel_eval > 1.0e-3
        assert rel_whole > 1.0e-3
        assert rel_pml > 1.0e-3
        assert self_loss_legacy < 1.0e-10
        assert self_loss_candidate < 1.0e-10
        assert cross_loss_candidate_on_legacy > self_loss_candidate
        assert cross_loss_legacy_on_candidate > self_loss_legacy
