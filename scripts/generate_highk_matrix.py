"""
Generate executable C0-C21 high-wavenumber campaign specs.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import _bootstrap  # noqa: F401
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "configs" / "experiments"
TRAIN_GRID_CAP = 512
REFERENCE_GRID_CAP = 704
DEFAULT_LR = 3.0e-4
DEFAULT_EPOCHS = 5000

BASE_MODEL = {
    "nsno_blocks": 4,
    "nsno_channels": 32,
    "fno_modes": 16,
    "activation": "gelu",
    "pointwise_variant": "linear",
}
DEEP_MODEL = {
    "nsno_blocks": 12,
    "nsno_channels": 48,
    "fno_modes": 16,
    "activation": "gelu",
    "pointwise_variant": "linear",
}
BEST_MODEL = dict(DEEP_MODEL)
QUADRATIC_MODEL = {
    "nsno_blocks": 12,
    "nsno_channels": 48,
    "fno_modes": 16,
    "activation": "gelu",
    "pointwise_variant": "quadratic",
}
STRICT_PML = {
    "power": 2,
    "R0": 1.0e-6,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate C0-C21 matrix YAML specs.")
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="Directory that receives generated C*.yaml files.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing C*.yaml files before writing new ones.",
    )
    return parser.parse_args()


def ppw_to_n(omega, ppw):
    return int(math.ceil(float(ppw) * float(omega) / (2.0 * math.pi)))


def recommended_pml_width(n):
    if n <= 192:
        return 16
    if n <= 320:
        return 24
    if n <= 512:
        return 32
    return 48


def blocked_for_grid(n, entrypoint):
    cap = REFERENCE_GRID_CAP if entrypoint == "reference_only" else TRAIN_GRID_CAP
    if n > cap:
        return True, f"grid {n} exceeds {entrypoint} cap {cap}"
    return False, ""


def ensure_target_kind(training):
    training = dict(training)
    supervision = dict(training.get("supervision", {}))
    supervision.setdefault("target_kind", "scattering_envelope")
    training["supervision"] = supervision
    return training


def training_pde(epochs=DEFAULT_EPOCHS, lr=DEFAULT_LR):
    return ensure_target_kind(
        {
            "epochs": int(epochs),
            "lr": float(lr),
            "lambda_pde": 1.0,
            "lambda_data": 0.0,
            "supervision": {
                "enabled": False,
                "reference_path": None,
            },
        }
    )


def training_supervised(lambda_pde, lambda_data, epochs=DEFAULT_EPOCHS, lr=DEFAULT_LR):
    return ensure_target_kind(
        {
            "epochs": int(epochs),
            "lr": float(lr),
            "lambda_pde": float(lambda_pde),
            "lambda_data": float(lambda_data),
            "supervision": {
                "enabled": True,
                "reference_path": None,
            },
        }
    )


def fixed_train_block(model=None, training=None, medium=None, ppw_default=None):
    payload = {
        "residual": {"lap_tau_mode": "stretched_divergence"},
        "model": dict(model or BEST_MODEL),
        "training": training or training_pde(),
        "logging": {"level": "INFO"},
    }
    if medium is not None:
        payload["medium"] = {"velocity_model": medium}
    if ppw_default is not None:
        payload["ppw_default"] = ppw_default
    return payload


def run_spec(
    run_id,
    overrides,
    *,
    entrypoint=None,
    reference_prep=False,
    reference_label="matched",
    blocked_reason="",
):
    payload = {
        "run_id": run_id,
        "overrides": overrides,
    }
    if entrypoint is not None:
        payload["entrypoint"] = entrypoint
    if reference_prep:
        payload["reference_prep"] = {
            "enabled": True,
            "label_name": reference_label,
        }
    if blocked_reason:
        payload["status"] = "blocked"
        payload["blocked_reason"] = blocked_reason
    return payload


def maybe_blocked(run, n, entrypoint):
    blocked, reason = blocked_for_grid(n, entrypoint)
    if blocked:
        run["status"] = "blocked"
        run["blocked_reason"] = reason
    return run


def batch(batch_id, name, purpose, entrypoint, fixed, runs):
    return {
        "batch_id": batch_id,
        "name": name,
        "status": "active",
        "entrypoint": entrypoint,
        "purpose": purpose,
        "expected_runs": len(runs),
        "fixed": fixed,
        "matrix": runs,
    }


def curriculum_training_base(lambda_pde=1.0, lambda_data=10.0, epochs=1500):
    if lambda_data > 0.0:
        return training_supervised(lambda_pde, lambda_data, epochs=epochs, lr=DEFAULT_LR)
    return training_pde(epochs=epochs, lr=DEFAULT_LR)


def curriculum_stages(target_omega, schedule_name, pointwise_variant="linear"):
    target_omega = int(target_omega)
    if target_omega == 90:
        schedule_map = {
            "direct": [90],
            "octave_2": [45, 90],
            "octave_3": [30, 45, 90],
            "octave_4": [30, 45, 60, 90],
            "dense": [30, 45, 60, 75, 90],
            "grow_depth": [30, 60, 90],
            "no_restart": [30, 45, 60, 90],
        }
        depth_map = {
            "grow_depth": [4, 8, 12],
        }
    elif target_omega == 120:
        schedule_map = {
            "direct": [120],
            "octave_2": [60, 120],
            "octave_3": [30, 60, 120],
            "octave_4": [30, 45, 60, 120],
            "dense": [30, 45, 60, 90, 120],
            "grow_depth": [30, 60, 120],
            "no_restart": [30, 45, 60, 120],
        }
        depth_map = {
            "grow_depth": [4, 8, 12],
        }
    else:
        schedule_map = {
            "direct": [180],
            "octave_2": [90, 180],
            "octave_3": [45, 90, 180],
            "octave_4": [30, 60, 120, 180],
            "dense": [30, 45, 60, 90, 120, 150, 180],
            "grow_depth": [30, 60, 120, 180],
            "no_restart": [30, 60, 120, 180],
        }
        depth_map = {
            "grow_depth": [4, 8, 12, 16],
        }
    omegas = schedule_map[schedule_name]
    blocks_list = depth_map.get(schedule_name, [12] * len(omegas))
    if len(blocks_list) != len(omegas):
        blocks_list = [blocks_list[min(index, len(blocks_list) - 1)] for index in range(len(omegas))]

    total_stages = len(omegas)
    if schedule_name == "direct":
        epoch_budget = [DEFAULT_EPOCHS]
    else:
        base_epochs = max(800, int(math.ceil(DEFAULT_EPOCHS / max(2, total_stages))))
        epoch_budget = [base_epochs] * total_stages
        epoch_budget[-1] = max(epoch_budget[-1], 1200)

    stages = []
    for index, (omega, blocks, stage_epochs) in enumerate(zip(omegas, blocks_list, epoch_budget), start=1):
        n = ppw_to_n(omega, 16)
        stages.append(
            {
                "name": f"omega{omega:03d}_L{blocks:02d}",
                "overrides": {
                    "physics": {"omega": float(omega)},
                    "grid": {"nx": n, "ny": n},
                    "pml": {"width": recommended_pml_width(n)},
                    "model": {
                        "nsno_blocks": int(blocks),
                        "nsno_channels": 48,
                        "fno_modes": 16,
                        "activation": "gelu",
                        "pointwise_variant": pointwise_variant,
                    },
                    "training": {
                        "epochs": int(stage_epochs),
                        "lr": DEFAULT_LR,
                    },
                },
                "warm_start": {
                    "enabled": index > 1,
                    "reset_optimizer": schedule_name not in {"no_restart"},
                    "strict": False,
                },
            }
        )
    return stages


def c0_reference_gate():
    runs = []
    for medium in ("homogeneous", "smooth_lens", "layered"):
        for omega in (30, 60, 90, 120):
            for n in (64, 128, 192):
                for pml_width in (8, 16, 24):
                    run = run_spec(
                        f"{medium}__omega{omega:03d}__N{n:03d}__pml{pml_width:02d}",
                        {
                            "medium": {"velocity_model": medium},
                            "physics": {"omega": float(omega)},
                            "grid": {"nx": n, "ny": n},
                            "pml": {"width": pml_width, **STRICT_PML},
                            "training": {"epochs": 1},
                        },
                    )
                    runs.append(maybe_blocked(run, n, "reference_only"))
    return batch(
        "C0",
        "reference_gate",
        "Reference gate over media, frequency, grid, and PML width.",
        "reference_only",
        {
            "residual": {"lap_tau_mode": "stretched_divergence"},
            "training": {"epochs": 1},
            "logging": {"level": "INFO"},
        },
        runs,
    )


def c1_highk_scaling():
    runs = []
    models = {
        "base": dict(BASE_MODEL),
        "deep": {"nsno_blocks": 8, "nsno_channels": 32, "fno_modes": 16, "activation": "gelu", "pointwise_variant": "linear"},
    }
    for omega in (30, 60, 90, 120, 180, 240, 300):
        for ppw in (8, 12, 16):
            n = ppw_to_n(omega, ppw)
            pml_width = recommended_pml_width(n)
            for medium in ("smooth_lens", "layered"):
                for tier_name, model_cfg in models.items():
                    run_pde = run_spec(
                        f"{medium}__omega{omega:03d}__ppw{ppw:02d}__N{n:03d}__{tier_name}__pde",
                        {
                            "medium": {"velocity_model": medium},
                            "physics": {"omega": float(omega)},
                            "grid": {"nx": n, "ny": n},
                            "pml": {"width": pml_width, **STRICT_PML},
                            "model": model_cfg,
                            "training": training_pde(),
                        },
                    )
                    runs.append(maybe_blocked(run_pde, n, "train"))
                    run_hybrid = run_spec(
                        f"{medium}__omega{omega:03d}__ppw{ppw:02d}__N{n:03d}__{tier_name}__hybrid1x10",
                        {
                            "medium": {"velocity_model": medium},
                            "physics": {"omega": float(omega)},
                            "grid": {"nx": n, "ny": n},
                            "pml": {"width": pml_width, **STRICT_PML},
                            "model": model_cfg,
                            "training": training_supervised(1.0, 10.0),
                        },
                        reference_prep=True,
                    )
                    runs.append(maybe_blocked(run_hybrid, n, "train"))
    return batch(
        "C1",
        "highk_scaling",
        "Frequency scaling under fixed ppw.",
        "train",
        fixed_train_block(),
        runs,
    )


def c2_ppw_resolution():
    runs = []
    for omega in (90, 180, 300):
        for ppw in (6, 8, 10, 12, 16, 24):
            n = ppw_to_n(omega, ppw)
            pml_width = recommended_pml_width(n)
            for medium in ("smooth_lens", "layered"):
                run = run_spec(
                    f"{medium}__omega{omega:03d}__ppw{ppw:02d}__N{n:03d}__best_hybrid",
                    {
                        "medium": {"velocity_model": medium},
                        "physics": {"omega": float(omega)},
                        "grid": {"nx": n, "ny": n},
                        "pml": {"width": pml_width, **STRICT_PML},
                        "model": BEST_MODEL,
                        "training": training_supervised(1.0, 10.0),
                    },
                    reference_prep=True,
                )
                runs.append(maybe_blocked(run, n, "train"))
    return batch(
        "C2",
        "ppw_resolution_law",
        "Resolution law sweep over ppw at representative high frequencies.",
        "train",
        fixed_train_block(model=BEST_MODEL, training=training_supervised(1.0, 10.0)),
        runs,
    )


def c3_grid_pollution():
    runs = []
    for omega in (30, 60, 90, 120, 180):
        for n in (64, 96, 128, 192, 256, 384):
            for medium in ("homogeneous", "smooth_lens", "layered"):
                pml_width = recommended_pml_width(n)
                run = run_spec(
                    f"{medium}__omega{omega:03d}__N{n:03d}__reference",
                    {
                        "medium": {"velocity_model": medium},
                        "physics": {"omega": float(omega)},
                        "grid": {"nx": n, "ny": n},
                        "pml": {"width": pml_width, **STRICT_PML},
                        "training": {"epochs": 1},
                    },
                )
                runs.append(maybe_blocked(run, n, "reference_only"))
    for omega in (60, 120, 180):
        for n in (96, 128, 192, 256):
            for medium in ("smooth_lens", "layered"):
                pml_width = recommended_pml_width(n)
                run = run_spec(
                    f"{medium}__omega{omega:03d}__N{n:03d}__train_subset",
                    {
                        "medium": {"velocity_model": medium},
                        "physics": {"omega": float(omega)},
                        "grid": {"nx": n, "ny": n},
                        "pml": {"width": pml_width, **STRICT_PML},
                        "model": BEST_MODEL,
                        "training": training_pde(),
                    },
                    entrypoint="train",
                )
                runs.append(maybe_blocked(run, n, "train"))
    return batch(
        "C3",
        "grid_pollution_phase_budget",
        "Reference-first grid pollution and phase budget campaign.",
        "reference_only",
        {
            "residual": {"lap_tau_mode": "stretched_divergence"},
            "training": {"epochs": 1},
            "logging": {"level": "INFO"},
        },
        runs,
    )


def c4_highk_pml_stability():
    runs = []
    profiles = {
        "soft": {"power": 1, "R0": 1.0e-4},
        "default": {"power": 2, "R0": 1.0e-6},
        "strong": {"power": 2, "R0": 1.0e-8},
        "steep": {"power": 3, "R0": 1.0e-6},
        "aggressive": {"power": 4, "R0": 1.0e-8},
    }
    for omega in (90, 180, 300):
        n = ppw_to_n(omega, 16)
        for thickness in (0.5, 0.75, 1.0, 1.5, 2.0):
            width = max(4, int(round(thickness * 16)))
            for profile_name, profile_cfg in profiles.items():
                for medium in ("smooth_lens", "layered"):
                    run = run_spec(
                        f"{medium}__omega{omega:03d}__thk{str(thickness).replace('.', 'p')}w__{profile_name}",
                        {
                            "medium": {"velocity_model": medium},
                            "physics": {"omega": float(omega)},
                            "grid": {"nx": n, "ny": n},
                            "pml": {"width": width, **profile_cfg},
                            "model": BEST_MODEL,
                            "training": training_pde(epochs=3000),
                        },
                    )
                    runs.append(maybe_blocked(run, n, "train"))
    return batch(
        "C4",
        "highk_pml_stability",
        "PML profile and thickness stability at fixed ppw=16.",
        "train",
        fixed_train_block(model=BEST_MODEL, training=training_pde(epochs=3000)),
        runs,
    )


def c5_laptau_strictness():
    runs = []
    sources = {
        "center": [0.5, 0.5],
        "near_boundary": [0.15, 0.5],
    }
    for mode in ("mixed_legacy", "stretched_divergence"):
        for omega in (30, 90, 180):
            for pml_width in (8, 16, 24):
                n = ppw_to_n(omega, 16)
                for medium in ("homogeneous", "smooth_lens", "layered"):
                    for source_name, source_pos in sources.items():
                        run = run_spec(
                            f"{medium}__omega{omega:03d}__N{n:03d}__pml{pml_width:02d}__{source_name}__{mode}",
                            {
                                "medium": {"velocity_model": medium},
                                "physics": {"omega": float(omega), "source_pos": source_pos},
                                "grid": {"nx": n, "ny": n},
                                "pml": {"width": pml_width, **STRICT_PML},
                                "residual": {"lap_tau_mode": mode},
                                "model": BEST_MODEL,
                                "training": training_pde(epochs=3000),
                            },
                        )
                        runs.append(maybe_blocked(run, n, "train"))
    return batch(
        "C5",
        "laptau_strictness",
        "Strict vs legacy LapTau residual audit.",
        "train",
        fixed_train_block(model=BEST_MODEL, training=training_pde(epochs=3000)),
        runs,
    )


def c6_medium_complexity():
    runs = []
    for medium in ("homogeneous", "smooth_lens", "layered"):
        for omega in (60, 120, 180, 300):
            for ppw in (12, 16):
                n = ppw_to_n(omega, ppw)
                for depth in (4, 8, 12):
                    run = run_spec(
                        f"{medium}__omega{omega:03d}__ppw{ppw:02d}__N{n:03d}__L{depth:02d}",
                        {
                            "medium": {"velocity_model": medium},
                            "physics": {"omega": float(omega)},
                            "grid": {"nx": n, "ny": n},
                            "pml": {"width": recommended_pml_width(n), **STRICT_PML},
                            "model": {
                                "nsno_blocks": depth,
                                "nsno_channels": 48,
                                "fno_modes": 16,
                                "activation": "gelu",
                                "pointwise_variant": "linear",
                            },
                            "training": training_pde(),
                        },
                    )
                    runs.append(maybe_blocked(run, n, "train"))
    return batch(
        "C6",
        "medium_complexity",
        "Scattering difficulty sweep across media families.",
        "train",
        fixed_train_block(),
        runs,
    )


def c7_source_robustness():
    runs = []
    sources = {
        "center": [0.5, 0.5],
        "x_quarter": [0.25, 0.5],
        "y_quarter": [0.5, 0.25],
        "off_center": [0.35, 0.65],
        "near_left": [0.15, 0.5],
        "near_corner": [0.2, 0.2],
    }
    for source_name, source_pos in sources.items():
        for omega in (60, 120, 180):
            n = ppw_to_n(omega, 16)
            for medium in ("smooth_lens", "layered"):
                run = run_spec(
                    f"{medium}__omega{omega:03d}__N{n:03d}__{source_name}",
                    {
                        "medium": {"velocity_model": medium},
                        "physics": {"omega": float(omega), "source_pos": source_pos},
                        "grid": {"nx": n, "ny": n},
                        "pml": {"width": recommended_pml_width(n), **STRICT_PML},
                        "model": BEST_MODEL,
                        "training": training_supervised(1.0, 10.0),
                    },
                    reference_prep=True,
                )
                runs.append(maybe_blocked(run, n, "train"))
    return batch(
        "C7",
        "source_robustness",
        "Single-source robustness under high frequency.",
        "train",
        fixed_train_block(model=BEST_MODEL, training=training_supervised(1.0, 10.0)),
        runs,
    )


def c8_depth_sweep():
    runs = []
    for depth in (1, 2, 4, 6, 8, 12, 16, 24):
        for omega in (60, 120, 180):
            n = ppw_to_n(omega, 16)
            for medium in ("smooth_lens", "layered"):
                run = run_spec(
                    f"{medium}__omega{omega:03d}__N{n:03d}__L{depth:02d}",
                    {
                        "medium": {"velocity_model": medium},
                        "physics": {"omega": float(omega)},
                        "grid": {"nx": n, "ny": n},
                        "pml": {"width": recommended_pml_width(n), **STRICT_PML},
                        "model": {
                            "nsno_blocks": depth,
                            "nsno_channels": 32,
                            "fno_modes": 16,
                            "activation": "gelu",
                            "pointwise_variant": "linear",
                        },
                        "training": training_pde(),
                    },
                )
                runs.append(maybe_blocked(run, n, "train"))
    return batch(
        "C8",
        "nsno_depth_sweep",
        "Depth sweep for Neumann unrolling capacity.",
        "train",
        fixed_train_block(),
        runs,
    )


def c9_mode_sweep():
    runs = []
    for modes in (4, 8, 12, 16, 24, 32, 48):
        for omega in (120, 180):
            n = ppw_to_n(omega, 16)
            for medium in ("smooth_lens", "layered"):
                run = run_spec(
                    f"{medium}__omega{omega:03d}__N{n:03d}__M{modes:02d}",
                    {
                        "medium": {"velocity_model": medium},
                        "physics": {"omega": float(omega)},
                        "grid": {"nx": n, "ny": n},
                        "pml": {"width": recommended_pml_width(n), **STRICT_PML},
                        "model": {
                            "nsno_blocks": 12,
                            "nsno_channels": 48,
                            "fno_modes": modes,
                            "activation": "gelu",
                            "pointwise_variant": "linear",
                        },
                        "training": training_pde(),
                    },
                )
                runs.append(maybe_blocked(run, n, "train"))
    return batch(
        "C9",
        "fno_mode_bandwidth",
        "FNO mode truncation sweep after phase stripping.",
        "train",
        fixed_train_block(),
        runs,
    )


def c10_channel_sweep():
    runs = []
    omega = 120
    n = ppw_to_n(omega, 16)
    for channels in (16, 32, 48, 64, 96, 128):
        for depth in (8, 12):
            for medium in ("smooth_lens", "layered"):
                run = run_spec(
                    f"{medium}__omega{omega:03d}__N{n:03d}__L{depth:02d}__C{channels:03d}",
                    {
                        "medium": {"velocity_model": medium},
                        "physics": {"omega": float(omega)},
                        "grid": {"nx": n, "ny": n},
                        "pml": {"width": recommended_pml_width(n), **STRICT_PML},
                        "model": {
                            "nsno_blocks": depth,
                            "nsno_channels": channels,
                            "fno_modes": 16,
                            "activation": "gelu",
                            "pointwise_variant": "linear",
                        },
                        "training": training_pde(),
                    },
                )
                runs.append(maybe_blocked(run, n, "train"))
    return batch(
        "C10",
        "channel_capacity",
        "Channel-width capacity sweep.",
        "train",
        fixed_train_block(),
        runs,
    )


def c11_supervision_loss():
    runs = []
    loss_modes = {
        "pde_only": (1.0, 0.0, False),
        "data_only": (0.0, 1.0, True),
        "hybrid_balanced": (1.0, 1.0, True),
        "hybrid_data10": (1.0, 10.0, True),
        "hybrid_pde10": (10.0, 1.0, True),
        "hybrid_weak_pde": (0.1, 1.0, True),
        "hybrid_weak_data": (1.0, 0.1, True),
    }
    for omega in (60, 120, 180):
        n = ppw_to_n(omega, 16)
        for medium in ("smooth_lens", "layered"):
            for mode_name, (lambda_pde, lambda_data, needs_ref) in loss_modes.items():
                training_cfg = (
                    training_supervised(lambda_pde, lambda_data)
                    if needs_ref
                    else training_pde()
                )
                if not needs_ref:
                    training_cfg["lambda_pde"] = float(lambda_pde)
                run = run_spec(
                    f"{medium}__omega{omega:03d}__N{n:03d}__{mode_name}",
                    {
                        "medium": {"velocity_model": medium},
                        "physics": {"omega": float(omega)},
                        "grid": {"nx": n, "ny": n},
                        "pml": {"width": recommended_pml_width(n), **STRICT_PML},
                        "model": BEST_MODEL,
                        "training": training_cfg,
                    },
                    reference_prep=needs_ref,
                )
                runs.append(maybe_blocked(run, n, "train"))
    return batch(
        "C11",
        "supervision_loss_matrix",
        "PDE-only / data-only / hybrid supervision sweep.",
        "train",
        fixed_train_block(model=BEST_MODEL),
        runs,
    )


def c12_optimization_budget():
    runs = []
    for epochs in (1000, 3000, 5000, 10000, 20000, 40000):
        for lr in (1.0e-4, 3.0e-4, 1.0e-3):
            for omega in (120, 180):
                n = ppw_to_n(omega, 16)
                for medium in ("smooth_lens", "layered"):
                    run = run_spec(
                        f"{medium}__omega{omega:03d}__N{n:03d}__lr{str(lr).replace('.', 'p')}__ep{epochs:05d}",
                        {
                            "medium": {"velocity_model": medium},
                            "physics": {"omega": float(omega)},
                            "grid": {"nx": n, "ny": n},
                            "pml": {"width": recommended_pml_width(n), **STRICT_PML},
                            "model": BEST_MODEL,
                            "training": training_supervised(1.0, 10.0, epochs=epochs, lr=lr),
                        },
                        reference_prep=True,
                    )
                    runs.append(maybe_blocked(run, n, "train"))
    return batch(
        "C12",
        "optimization_budget",
        "Learning-rate and epoch-budget sweep.",
        "train",
        fixed_train_block(model=BEST_MODEL, training=training_supervised(1.0, 10.0)),
        runs,
    )


def c13_curriculum():
    runs = []
    for target_omega in (120, 180):
        target_n = ppw_to_n(target_omega, 16)
        for medium in ("smooth_lens", "layered"):
            for schedule_name in ("direct", "octave_2", "octave_3", "octave_4", "dense", "grow_depth", "no_restart"):
                run = run_spec(
                    f"{medium}__omega{target_omega:03d}__N{target_n:03d}__{schedule_name}",
                    {
                        "medium": {"velocity_model": medium},
                        "model": BEST_MODEL,
                        "training": {
                            **curriculum_training_base(),
                            "curriculum": {
                                "enabled": True,
                                "schedule_name": schedule_name,
                                "stages": curriculum_stages(target_omega, schedule_name),
                            },
                        },
                    },
                    reference_prep=True,
                )
                runs.append(maybe_blocked(run, target_n, "train"))
    return batch(
        "C13",
        "curriculum_warm_start",
        "Curriculum and warm-start multi-stage jobs.",
        "train",
        fixed_train_block(model=BEST_MODEL, training=curriculum_training_base()),
        runs,
    )


def c14_phase_pathology():
    runs = []
    supervision_modes = {
        "pde_only": (1.0, 0.0, False),
        "data_only": (0.0, 1.0, True),
        "hybrid": (1.0, 1.0, True),
    }
    model_map = {
        "base": BASE_MODEL,
        "deep": DEEP_MODEL,
    }
    for omega in (60, 120, 180, 300):
        n = ppw_to_n(omega, 16)
        for medium in ("smooth_lens", "layered"):
            for mode_name, (lambda_pde, lambda_data, needs_ref) in supervision_modes.items():
                training_cfg = training_supervised(lambda_pde, lambda_data) if needs_ref else training_pde()
                if not needs_ref:
                    training_cfg["lambda_pde"] = float(lambda_pde)
                for model_name, model_cfg in model_map.items():
                    run = run_spec(
                        f"{medium}__omega{omega:03d}__N{n:03d}__{mode_name}__{model_name}",
                        {
                            "medium": {"velocity_model": medium},
                            "physics": {"omega": float(omega)},
                            "grid": {"nx": n, "ny": n},
                            "pml": {"width": recommended_pml_width(n), **STRICT_PML},
                            "model": model_cfg,
                            "training": training_cfg,
                        },
                        reference_prep=needs_ref,
                    )
                    runs.append(maybe_blocked(run, n, "train"))
    return batch(
        "C14",
        "phase_pathology",
        "Phase-loss pathology matrix.",
        "train",
        fixed_train_block(),
        runs,
    )


def card_specs():
    return {
        "A": {"omega": 30, "n": 256, "medium": "smooth_lens", "source_pos": [0.5, 0.5]},
        "B": {"omega": 120, "n": ppw_to_n(120, 16), "medium": "smooth_lens", "source_pos": [0.5, 0.5]},
        "C": {"omega": 180, "n": ppw_to_n(180, 16), "medium": "layered", "source_pos": [0.5, 0.5]},
        "D": {"omega": 300, "n": ppw_to_n(300, 12), "medium": "smooth_lens", "source_pos": [0.5, 0.5]},
        "E": {"omega": 180, "n": ppw_to_n(180, 8), "medium": "layered", "source_pos": [0.5, 0.5]},
        "F": {"omega": 120, "n": ppw_to_n(120, 16), "medium": "smooth_lens", "source_pos": [0.15, 0.5]},
        "G": {"omega": 180, "n": ppw_to_n(180, 16), "medium": "smooth_lens", "source_pos": [0.5, 0.5]},
    }


def c15_benchmark_cards():
    runs = []
    for card_name, spec in card_specs().items():
        omega = spec["omega"]
        n = spec["n"]
        pml_width = recommended_pml_width(n)
        common = {
            "medium": {"velocity_model": spec["medium"]},
            "physics": {"omega": float(omega), "source_pos": spec["source_pos"]},
            "grid": {"nx": n, "ny": n},
            "pml": {"width": pml_width, **STRICT_PML},
        }
        runs.append(
            maybe_blocked(
                run_spec(
                    f"card{card_name}__reference_only",
                    {
                        **common,
                        "training": {"epochs": 1},
                    },
                    entrypoint="reference_only",
                ),
                n,
                "reference_only",
            )
        )
        runs.append(
            maybe_blocked(
                run_spec(
                    f"card{card_name}__base_nsno",
                    {
                        **common,
                        "model": BASE_MODEL,
                        "training": training_pde(),
                    },
                ),
                n,
                "train",
            )
        )
        runs.append(
            maybe_blocked(
                run_spec(
                    f"card{card_name}__deep_nsno",
                    {
                        **common,
                        "model": DEEP_MODEL,
                        "training": training_pde(),
                    },
                ),
                n,
                "train",
            )
        )
        runs.append(
            maybe_blocked(
                run_spec(
                    f"card{card_name}__best_hybrid_proxy",
                    {
                        **common,
                        "model": BEST_MODEL,
                        "training": training_supervised(1.0, 10.0),
                    },
                    reference_prep=True,
                ),
                n,
                "train",
            )
        )
        curriculum_target = 120 if omega <= 120 else 180
        runs.append(
            maybe_blocked(
                run_spec(
                    f"card{card_name}__best_curriculum_proxy",
                    {
                        **common,
                        "model": BEST_MODEL,
                        "training": {
                            **curriculum_training_base(),
                            "curriculum": {
                                "enabled": True,
                                "schedule_name": "octave_4",
                                "stages": curriculum_stages(curriculum_target, "octave_4"),
                            },
                        },
                    },
                    reference_prep=True,
                ),
                n,
                "train",
            )
        )
    return batch(
        "C15",
        "benchmark_cards",
        "SOTA narrative benchmark cards with proxy best-of variants.",
        "train",
        fixed_train_block(),
        runs,
    )


def c16_neural_pml_window():
    runs = []
    for omega in (60, 120, 180):
        n = ppw_to_n(omega, 16)
        for medium in ("smooth_lens", "layered"):
            for width in (8, 16, 24, 32, 48):
                run = run_spec(
                    f"{medium}__omega{omega:03d}__N{n:03d}__widthA_{width:02d}",
                    {
                        "medium": {"velocity_model": medium},
                        "physics": {"omega": float(omega)},
                        "grid": {"nx": n, "ny": n},
                        "pml": {"width": width, "power": 2, "R0": 1.0e-6},
                        "model": BEST_MODEL,
                        "training": training_supervised(1.0, 10.0),
                    },
                    reference_prep=True,
                )
                runs.append(maybe_blocked(run, n, "train"))
            for r0 in (1.0e-3, 1.0e-4, 1.0e-6, 1.0e-8, 1.0e-10):
                run = run_spec(
                    f"{medium}__omega{omega:03d}__N{n:03d}__dampingB_R0_{str(r0).replace('.', 'p').replace('-', 'm')}",
                    {
                        "medium": {"velocity_model": medium},
                        "physics": {"omega": float(omega)},
                        "grid": {"nx": n, "ny": n},
                        "pml": {"width": 24, "power": 2, "R0": r0},
                        "model": BEST_MODEL,
                        "training": training_supervised(1.0, 10.0),
                    },
                    reference_prep=True,
                )
                runs.append(maybe_blocked(run, n, "train"))
            for power in (1, 2, 3, 4):
                run = run_spec(
                    f"{medium}__omega{omega:03d}__N{n:03d}__profileC_p{power}",
                    {
                        "medium": {"velocity_model": medium},
                        "physics": {"omega": float(omega)},
                        "grid": {"nx": n, "ny": n},
                        "pml": {"width": 24, "power": power, "R0": 1.0e-6},
                        "model": BEST_MODEL,
                        "training": training_supervised(1.0, 10.0),
                    },
                    reference_prep=True,
                )
                runs.append(maybe_blocked(run, n, "train"))
    for omega in (180, 240, 300):
        n = ppw_to_n(omega, 16)
        for label, profile in (
            ("best", {"width": 24, "power": 2, "R0": 1.0e-6}),
            ("worst", {"width": 8, "power": 1, "R0": 1.0e-3}),
        ):
            run = run_spec(
                f"layered__omega{omega:03d}__N{n:03d}__cornerD_{label}",
                {
                    "medium": {"velocity_model": "layered"},
                    "physics": {"omega": float(omega)},
                    "grid": {"nx": n, "ny": n},
                    "pml": profile,
                    "model": BEST_MODEL,
                    "training": training_supervised(1.0, 10.0),
                },
                reference_prep=True,
            )
            runs.append(maybe_blocked(run, n, "train"))
    return batch(
        "C16",
        "neural_pml_damping_window",
        "PML window, damping, and profile diagnostics.",
        "train",
        fixed_train_block(model=BEST_MODEL, training=training_supervised(1.0, 10.0)),
        runs,
    )


def c17_pml_interface_consistency():
    runs = []
    for omega in (120, 180):
        n = ppw_to_n(omega, 16)
        for medium in ("smooth_lens", "layered"):
            for width in (16, 24, 32):
                for power in (1, 2, 3):
                    for mode in ("stretched_divergence", "mixed_legacy"):
                        run = run_spec(
                            f"{medium}__omega{omega:03d}__N{n:03d}__w{width:02d}__p{power}__{mode}",
                            {
                                "medium": {"velocity_model": medium},
                                "physics": {"omega": float(omega)},
                                "grid": {"nx": n, "ny": n},
                                "pml": {"width": width, "power": power, "R0": 1.0e-6},
                                "residual": {"lap_tau_mode": mode},
                                "model": BEST_MODEL,
                                "training": training_supervised(1.0, 10.0),
                            },
                            reference_prep=True,
                        )
                        runs.append(maybe_blocked(run, n, "train"))
    return batch(
        "C17",
        "pml_interface_consistency",
        "Interface consistency proxy sweep over width, power, and LapTau mode.",
        "train",
        fixed_train_block(model=BEST_MODEL, training=training_supervised(1.0, 10.0)),
        runs,
    )


def c18_auxiliary_field_audit():
    runs = []
    for mode in ("stretched_divergence", "mixed_legacy"):
        for omega in (60, 120, 180, 240):
            for ppw in (12, 16):
                n = ppw_to_n(omega, ppw)
                for medium in ("homogeneous", "smooth_lens", "layered"):
                    for width in (16, 24):
                        run = run_spec(
                            f"{medium}__omega{omega:03d}__ppw{ppw:02d}__N{n:03d}__w{width:02d}__{mode}",
                            {
                                "medium": {"velocity_model": medium},
                                "physics": {"omega": float(omega)},
                                "grid": {"nx": n, "ny": n},
                                "pml": {"width": width, **STRICT_PML},
                                "residual": {"lap_tau_mode": mode},
                                "model": DEEP_MODEL,
                                "training": training_pde(epochs=3000),
                            },
                        )
                        runs.append(maybe_blocked(run, n, "train"))
    return batch(
        "C18",
        "auxiliary_field_pml_audit",
        "Auxiliary-field PML audit across frequency, media, and width.",
        "train",
        fixed_train_block(model=DEEP_MODEL, training=training_pde(epochs=3000)),
        runs,
    )


def c19_pml_loss_weighting():
    runs = []
    ratios = (0.01, 0.1, 1.0, 10.0, 100.0)
    pml_fractions = (0.05, 0.15, 0.25, 0.40)
    for ratio in ratios:
        for pml_fraction in pml_fractions:
            for omega in (120, 180):
                n = ppw_to_n(omega, 16)
                for medium in ("smooth_lens", "layered"):
                    run = run_spec(
                        f"{medium}__omega{omega:03d}__N{n:03d}__lpml{str(ratio).replace('.', 'p')}__pf{str(pml_fraction).replace('.', 'p')}",
                        {
                            "medium": {"velocity_model": medium},
                            "physics": {"omega": float(omega)},
                            "grid": {"nx": n, "ny": n},
                            "pml": {"width": recommended_pml_width(n), **STRICT_PML},
                            "model": BEST_MODEL,
                            "training": {
                                **training_supervised(1.0, 10.0),
                                "lambda_physical": 1.0,
                                "lambda_pml": ratio,
                                "lambda_interface": 1.0,
                            },
                            "sampling": {
                                "pml_fraction": pml_fraction,
                                "interface_oversample": 2.0,
                                "interface_band_h": 4,
                            },
                        },
                        reference_prep=True,
                    )
                    runs.append(maybe_blocked(run, n, "train"))
    return batch(
        "C19",
        "pml_loss_weighting",
        "PML residual weighting and collocation balance.",
        "train",
        fixed_train_block(model=BEST_MODEL, training=training_supervised(1.0, 10.0)),
        runs,
    )


def c20_pml_antidote_architectures():
    runs = []
    arch_variants = {
        "base_nsno": {"model": BASE_MODEL, "curriculum": None, "needs_ref": True},
        "deep_nsno": {"model": DEEP_MODEL, "curriculum": None, "needs_ref": True},
        "quadratic_pointwise": {"model": QUADRATIC_MODEL, "curriculum": None, "needs_ref": True},
        "quadratic_curriculum": {"model": QUADRATIC_MODEL, "curriculum": "octave_4", "needs_ref": True},
    }
    for arch_name, arch_spec in arch_variants.items():
        for omega in (90, 120, 180):
            n = ppw_to_n(omega, 16)
            for medium in ("smooth_lens", "layered"):
                for r0 in (1.0e-4, 1.0e-6, 1.0e-8):
                    overrides = {
                        "medium": {"velocity_model": medium},
                        "physics": {"omega": float(omega)},
                        "grid": {"nx": n, "ny": n},
                        "pml": {"width": 24, "power": 2, "R0": r0},
                        "model": arch_spec["model"],
                        "training": training_supervised(1.0, 10.0),
                    }
                    if arch_spec["curriculum"] is not None:
                        overrides["training"] = {
                            **curriculum_training_base(),
                            "curriculum": {
                                "enabled": True,
                                "schedule_name": arch_spec["curriculum"],
                                "stages": curriculum_stages(omega, arch_spec["curriculum"], pointwise_variant="quadratic"),
                            },
                        }
                    run = run_spec(
                        f"{medium}__omega{omega:03d}__N{n:03d}__R0_{str(r0).replace('.', 'p').replace('-', 'm')}__{arch_name}",
                        overrides,
                        reference_prep=True,
                    )
                    runs.append(maybe_blocked(run, n, "train"))
    return batch(
        "C20",
        "pml_antidote_architectures",
        "Repo-feasible architecture antidotes for PML training traps.",
        "train",
        fixed_train_block(),
        runs,
    )


def c21_radiation_baselines_proxy():
    runs = []
    for omega in (90, 120, 180):
        n = ppw_to_n(omega, 16)
        for medium in ("smooth_lens", "layered"):
            for width in (24, 48, 64):
                run = run_spec(
                    f"{medium}__omega{omega:03d}__N{n:03d}__thickPML_w{width:02d}",
                    {
                        "medium": {"velocity_model": medium},
                        "physics": {"omega": float(omega)},
                        "grid": {"nx": n, "ny": n},
                        "pml": {"width": width, **STRICT_PML},
                        "model": BEST_MODEL,
                        "training": training_supervised(1.0, 10.0),
                    },
                    reference_prep=True,
                )
                runs.append(maybe_blocked(run, n, "train"))
    return batch(
        "C21",
        "radiation_baselines_proxy",
        "Thick-PML proxy batch for PML-free radiation baselines.",
        "train",
        fixed_train_block(model=BEST_MODEL, training=training_supervised(1.0, 10.0)),
        runs,
    )


def build_specs():
    return [
        c0_reference_gate(),
        c1_highk_scaling(),
        c2_ppw_resolution(),
        c3_grid_pollution(),
        c4_highk_pml_stability(),
        c5_laptau_strictness(),
        c6_medium_complexity(),
        c7_source_robustness(),
        c8_depth_sweep(),
        c9_mode_sweep(),
        c10_channel_sweep(),
        c11_supervision_loss(),
        c12_optimization_budget(),
        c13_curriculum(),
        c14_phase_pathology(),
        c15_benchmark_cards(),
        c16_neural_pml_window(),
        c17_pml_interface_consistency(),
        c18_auxiliary_field_audit(),
        c19_pml_loss_weighting(),
        c20_pml_antidote_architectures(),
        c21_radiation_baselines_proxy(),
    ]


def write_specs(output_dir: Path, specs):
    output_dir.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        file_path = output_dir / f"{spec['batch_id']}_{spec['name']}.yaml"
        with open(file_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(spec, f, sort_keys=False, allow_unicode=True)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if args.clean:
        for path in output_dir.glob("C*.yaml"):
            path.unlink()
    specs = build_specs()
    write_specs(output_dir, specs)
    active_runs = 0
    blocked_runs = 0
    for spec in specs:
        for run in spec["matrix"]:
            if str(run.get("status", "active")).lower() == "blocked":
                blocked_runs += 1
            else:
                active_runs += 1
    print(
        yaml.safe_dump(
            {
                "output_dir": str(output_dir),
                "batches": len(specs),
                "active_runs": active_runs,
                "blocked_runs": blocked_runs,
            },
            sort_keys=False,
            allow_unicode=True,
        ).strip()
    )


if __name__ == "__main__":
    main()
