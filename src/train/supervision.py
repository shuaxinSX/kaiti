"""
Optional supervision helpers for hybrid training.
"""

from pathlib import Path

import numpy as np
import torch


SUPERVISION_TARGET_KIND = "scattering_envelope"


def _normalize_target_kind(target_kind):
    if target_kind is None or str(target_kind).strip() == "":
        return SUPERVISION_TARGET_KIND
    normalized = str(target_kind).strip().lower().replace("-", "_")
    if normalized != SUPERVISION_TARGET_KIND:
        raise ValueError(
            "training.supervision.target_kind must be "
            f"{SUPERVISION_TARGET_KIND!r}; got {target_kind!r}."
        )
    return normalized


def _reject_known_wavefield_artifact(reference_path):
    if reference_path is None:
        return
    filename = Path(reference_path).name.lower()
    if "wavefield" in filename:
        raise ValueError(
            "training.supervision.reference_path must point to an offline "
            "phase-stripped scattering envelope, not a wavefield artifact. "
            "Use reference_envelope.npy."
        )


def resolve_supervision_config(cfg):
    """Parse supervision-related config with backward-compatible defaults."""
    training_cfg = cfg.training
    supervision_cfg = training_cfg.supervision if hasattr(training_cfg, "supervision") else None

    requested = bool(supervision_cfg.enabled) if supervision_cfg is not None and hasattr(supervision_cfg, "enabled") else False
    reference_path = None
    if supervision_cfg is not None and hasattr(supervision_cfg, "reference_path"):
        reference_path = supervision_cfg.reference_path
    target_kind = _normalize_target_kind(
        supervision_cfg.target_kind
        if supervision_cfg is not None and hasattr(supervision_cfg, "target_kind")
        else None
    )

    lambda_pde = float(training_cfg.lambda_pde)
    lambda_data = float(training_cfg.lambda_data)

    if lambda_data > 0.0:
        if not requested:
            raise ValueError(
                "training.lambda_data > 0 requires training.supervision.enabled = true."
            )
        if reference_path is None or str(reference_path).strip() == "":
            raise ValueError(
                "training.lambda_data > 0 requires training.supervision.reference_path."
            )
        _reject_known_wavefield_artifact(reference_path)

    return {
        "lambda_pde": lambda_pde,
        "lambda_data": lambda_data,
        "requested": requested,
        "active": lambda_data > 0.0 and requested,
        "reference_path": reference_path,
        "target_kind": target_kind,
    }


def complex_array_to_dual_channel_tensor(array):
    """Convert a complex [H, W] ndarray into a float32 [1, 2, H, W] tensor."""
    if not np.iscomplexobj(array):
        raise ValueError("reference_envelope.npy must contain a complex ndarray.")

    dual = np.stack(
        [
            np.asarray(array.real, dtype=np.float32),
            np.asarray(array.imag, dtype=np.float32),
        ],
        axis=0,
    )[None, ...]
    return torch.from_numpy(dual)


def load_reference_target(reference_path, expected_shape, device):
    """Load and validate a reference envelope label from disk."""
    resolved_path = Path(reference_path).expanduser()
    if not resolved_path.is_absolute():
        resolved_path = Path.cwd() / resolved_path

    if not resolved_path.exists():
        raise FileNotFoundError(f"Reference envelope file does not exist: {resolved_path}")

    reference = np.load(resolved_path)
    if reference.shape != tuple(expected_shape):
        raise ValueError(
            "reference_envelope.npy shape mismatch: "
            f"expected {tuple(expected_shape)}, got {reference.shape}."
        )

    target = complex_array_to_dual_channel_tensor(reference).to(device)
    return resolved_path, target
