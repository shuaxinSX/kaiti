"""
Summarize A6 matrix outputs without mutating core training code.

The summarizer only reads persisted run artifacts plus A6 launcher sidecars under
outputs/matrix/. It may derive extra tables from already-saved grids and models,
but it never edits the original training/evaluation scripts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import matplotlib
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.sankey import Sankey

plt.rcParams.update(
    {
        "font.size": 10,
        "figure.dpi": 150,
        "axes.grid": True,
        "grid.alpha": 0.25,
    }
)


REPORT_COLUMNS = [
    "batch_id",
    "run_id",
    "group_tag",
    "output_dir",
    "effective_config_path",
    "git_sha_int",
    "launched_at",
    "machine_host",
    "velocity_model",
    "omega",
    "grid_nx",
    "grid_ny",
    "domain_x_span",
    "domain_y_span",
    "pml_width",
    "pml_power",
    "pml_R0",
    "source_pos_x",
    "source_pos_y",
    "nsno_blocks",
    "nsno_channels",
    "fno_modes",
    "activation",
    "epochs",
    "lr",
    "batch_size",
    "lambda_pde",
    "lambda_data",
    "source_mask_radius",
    "omega2_normalize",
    "pml_rhs_zero",
    "eikonal_precision",
    "lap_tau_mode",
    "supervision_enabled",
    "supervision_reference_path",
    "seed",
    "wavelength_per_cell",
    "ppw",
    "pml_thickness_in_wavelengths",
    "total_params",
    "capacity_tier",
    "supervision_mode",
    "residual_rmse_evaluation",
    "residual_p95_evaluation",
    "residual_max_evaluation",
    "residual_mean_evaluation",
    "final_loss_total",
    "final_loss_pde",
    "final_loss_data",
    "first_loss_total",
    "loss_reduction_ratio",
    "min_loss_total_epoch",
    "reference_available",
    "rel_l2_to_reference",
    "amp_mae_to_reference",
    "phase_mae_to_reference",
    "reference_residual_rmse",
    "reference_residual_max",
    "rmse_gap_to_reference",
    "rmse_gap_ratio",
    "runtime_sec",
    "peak_mem_mb",
    "epochs_completed",
    "converged",
    "nan_detected",
    "early_stopped",
    "ref_solve_runtime_sec",
]

LOSS_HISTORY_COLUMNS = [
    "batch_id",
    "run_id",
    "epoch",
    "loss_total",
    "loss_pde",
    "loss_data",
    "lr_current",
    "loss_components_status",
]

QUANTILE_COLUMNS = [
    "batch_id",
    "run_id",
    "q00",
    "q01",
    "q05",
    "q10",
    "q25",
    "q50",
    "q75",
    "q90",
    "q95",
    "q99",
    "q100",
    "residual_std",
]

CENTERLINE_COLUMNS = [
    "batch_id",
    "run_id",
    "axis",
    "s",
    "pred_re",
    "pred_im",
    "ref_re",
    "ref_im",
    "pred_abs",
    "ref_abs",
    "residual_abs",
    "velocity",
    "loss_mask",
]

PARETO_COLUMNS = [
    "batch_id",
    "run_id",
    "metric_x_name",
    "metric_x",
    "metric_y_name",
    "metric_y",
    "is_pareto_front",
    "pareto_rank",
]

REFERENCE_ONLY_COLUMNS = [
    "run_id",
    "velocity_model",
    "omega",
    "grid_nx",
    "pml_width",
    "eikonal_precision",
    "reference_residual_rmse",
    "reference_residual_max",
    "solve_runtime_sec",
    "peak_mem_mb",
]

FAILURE_COLUMNS = [
    "batch_id",
    "run_id",
    "failure_stage",
    "error_type",
    "error_msg_head",
    "last_logged_epoch",
    "traceback_path",
    "recoverable",
    "retried",
]

PARAM_COUNT_CACHE: Dict[str, Optional[int]] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize A6 matrix outputs.")
    parser.add_argument(
        "--output-root",
        default="outputs/matrix",
        help="Matrix output root produced by scripts/run_benchmark.py.",
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_dict_rows(path: Path, delimiter: str = ",") -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter=delimiter))


def write_table(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            normalized: Dict[str, Any] = {}
            for field in fieldnames:
                value = row.get(field, "")
                if isinstance(value, float):
                    normalized[field] = "" if math.isnan(value) else f"{value:.6e}"
                else:
                    normalized[field] = value
            writer.writerow(normalized)


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def maybe_float(value: Any) -> Optional[float]:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def maybe_bool(value: Any) -> Optional[bool]:
    if value in ("", None):
        return None
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    return None


def is_finite(value: Any) -> bool:
    numeric = maybe_float(value)
    return numeric is not None and math.isfinite(numeric)


def safe_log10(value: Any) -> float:
    numeric = maybe_float(value)
    if numeric is None or numeric <= 0.0:
        return math.nan
    return math.log10(numeric)


def load_batch_specs() -> Dict[str, Dict[str, Any]]:
    specs: Dict[str, Dict[str, Any]] = {}
    for path in sorted((repo_root() / "configs" / "experiments").glob("B*.yaml")):
        spec = load_yaml(path)
        batch_id = str(spec.get("batch_id", path.stem.split("_", 1)[0]))
        spec["__path__"] = str(path)
        specs[batch_id] = spec
    return specs


def load_manifest(meta_root: Path) -> List[Dict[str, str]]:
    return read_dict_rows(meta_root / "manifest.tsv", delimiter="\t")


def collect_run_entries(manifest_rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for row in manifest_rows:
        if row.get("status") == "blocked":
            continue
        run_dir = Path(row["output_dir"])
        if (run_dir / "summary.json").exists() and (run_dir / "config_merged.yaml").exists():
            entries.append({"manifest": row, "run_dir": run_dir})
    return entries


def read_runtime_metadata(path: Path) -> Dict[str, Any]:
    metadata_path = path / "_matrix_runtime.json"
    if metadata_path.exists():
        return load_json(metadata_path)
    return {}


def extract_last_logged_epoch(text: str) -> str:
    matches = re.findall(r"Epoch\s+(\d+)(?:/\d+)?", text, flags=re.IGNORECASE)
    if matches:
        return matches[-1]
    matches = re.findall(r"epoch(?:=|\s+)(\d+)", text, flags=re.IGNORECASE)
    return matches[-1] if matches else ""


def load_csv_array(path: Path, ndmin: int = 1) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    try:
        return np.loadtxt(path, delimiter=",", ndmin=ndmin)
    except Exception:
        return None


def crop_to_physical(array: np.ndarray, cfg: Dict[str, Any], x_len: int, y_len: int) -> np.ndarray:
    if array.shape == (y_len, x_len):
        return array
    pml_width = int((cfg.get("pml", {}) or {}).get("width", 0))
    if pml_width > 0 and array.shape[0] >= y_len + 2 * pml_width and array.shape[1] >= x_len + 2 * pml_width:
        return array[pml_width:-pml_width, pml_width:-pml_width]
    return array[:y_len, :x_len]


def infer_supervision_mode(lambda_pde: Optional[float], lambda_data: Optional[float]) -> str:
    if lambda_pde is None or lambda_data is None:
        return "unknown"
    if lambda_data == 0.0:
        return "pde"
    if lambda_pde == 0.0:
        return "data"
    return "hybrid"


def infer_capacity_tier(nsno_blocks: Any, nsno_channels: Any, fno_modes: Any) -> str:
    triple = (nsno_blocks, nsno_channels, fno_modes)
    mapping = {
        (2, 16, 8): "small",
        (4, 32, 16): "base",
        (6, 64, 24): "large",
        (8, 96, 32): "xlarge",
    }
    return mapping.get(triple, "custom")


def count_params(config_merged_path: Path) -> Optional[int]:
    cache_key = str(config_merged_path.resolve())
    if cache_key in PARAM_COUNT_CACHE:
        return PARAM_COUNT_CACHE[cache_key]

    try:
        from src.config import Config
        from src.models.nsno import NSNO2D

        cfg = Config(load_yaml(config_merged_path))
        model = NSNO2D(cfg)
        PARAM_COUNT_CACHE[cache_key] = int(sum(param.numel() for param in model.parameters()))
    except Exception:
        PARAM_COUNT_CACHE[cache_key] = None

    return PARAM_COUNT_CACHE[cache_key]


def derive_axes(cfg: Dict[str, Any], config_merged_path: Path) -> Dict[str, Any]:
    physics = cfg.get("physics", {})
    grid = cfg.get("grid", {})
    pml = cfg.get("pml", {})
    model = cfg.get("model", {})
    training = cfg.get("training", {})
    loss = cfg.get("loss", {})
    eikonal = cfg.get("eikonal", {})
    medium = cfg.get("medium", {})
    supervision = training.get("supervision", {}) or {}

    domain = grid.get("domain", [0.0, 1.0, 0.0, 1.0])
    x_span = float(domain[1] - domain[0])
    y_span = float(domain[3] - domain[2])
    grid_nx = int(grid.get("nx", 0))
    dx = x_span / max(grid_nx - 1, 1)
    c_background = float(medium.get("c_background", 1.0))
    omega = float(physics.get("omega", 0.0))
    wavelength = 2.0 * math.pi * c_background / omega if omega else math.nan

    pml_width = int(pml.get("width", 0))
    pml_thickness = math.nan
    if wavelength and not math.isnan(wavelength):
        pml_thickness = (pml_width * dx) / wavelength

    return {
        "velocity_model": medium.get("velocity_model"),
        "omega": omega,
        "grid_nx": grid_nx,
        "grid_ny": int(grid.get("ny", 0)),
        "domain_x_span": x_span,
        "domain_y_span": y_span,
        "pml_width": pml_width,
        "pml_power": pml.get("power"),
        "pml_R0": pml.get("R0"),
        "source_pos_x": physics.get("source_pos", [None, None])[0],
        "source_pos_y": physics.get("source_pos", [None, None])[1],
        "nsno_blocks": model.get("nsno_blocks"),
        "nsno_channels": model.get("nsno_channels"),
        "fno_modes": model.get("fno_modes"),
        "activation": model.get("activation"),
        "epochs": training.get("epochs"),
        "lr": training.get("lr"),
        "batch_size": training.get("batch_size"),
        "lambda_pde": training.get("lambda_pde"),
        "lambda_data": training.get("lambda_data"),
        "source_mask_radius": loss.get("source_mask_radius"),
        "omega2_normalize": loss.get("omega2_normalize"),
        "pml_rhs_zero": loss.get("pml_rhs_zero"),
        "eikonal_precision": eikonal.get("precision"),
        "lap_tau_mode": (cfg.get("residual", {}) or {}).get("lap_tau_mode"),
        "supervision_enabled": supervision.get("enabled"),
        "supervision_reference_path": supervision.get("reference_path"),
        "seed": training.get("seed"),
        "wavelength_per_cell": wavelength,
        "ppw": wavelength / dx if dx and not math.isnan(wavelength) else math.nan,
        "pml_thickness_in_wavelengths": pml_thickness,
        "total_params": count_params(config_merged_path),
        "capacity_tier": infer_capacity_tier(
            model.get("nsno_blocks"),
            model.get("nsno_channels"),
            model.get("fno_modes"),
        ),
        "supervision_mode": infer_supervision_mode(
            maybe_float(training.get("lambda_pde")),
            maybe_float(training.get("lambda_data")),
        ),
    }


def read_loss_rows(losses_csv_path: Path) -> List[Dict[str, Any]]:
    rows = read_dict_rows(losses_csv_path)
    payload: List[Dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        payload.append(
            {
                "epoch": int(row.get("epoch", index)),
                "loss_total": maybe_float(row.get("loss_total", row.get("loss"))),
                "loss_pde": maybe_float(row.get("loss_pde")),
                "loss_data": maybe_float(row.get("loss_data")),
            }
        )
    return payload


def compute_final_loss_components(run_dir: Path, cfg: Dict[str, Any]) -> Dict[str, Any]:
    cache_path = run_dir / "_matrix_loss_components.json"
    model_path = run_dir / "model_state.pt"
    config_path = run_dir / "config_merged.yaml"
    if not model_path.exists():
        return {}

    latest_input_mtime = max(model_path.stat().st_mtime, config_path.stat().st_mtime)
    if cache_path.exists() and cache_path.stat().st_mtime >= latest_input_mtime:
        return load_json(cache_path)

    try:
        import torch

        from src.config import Config
        from src.train.losses import loss_data, loss_total
        from src.train.trainer import Trainer

        trainer = Trainer(Config(cfg), device="cpu")
        state_dict = torch.load(model_path, map_location=trainer.device)
        trainer.model.load_state_dict(state_dict)

        with torch.no_grad():
            a_scat = trainer.model(trainer.net_input)
            result = trainer.residual_computer.compute(a_scat)
            loss_pde_value = float(result["loss_pde"].item())
            loss_data_tensor = (
                loss_data(a_scat, trainer.reference_target) if trainer.supervision_enabled else None
            )
            loss_data_value = float(loss_data_tensor.item()) if loss_data_tensor is not None else 0.0
            loss_total_value = float(
                loss_total(
                    result["loss_pde"],
                    loss_data_tensor,
                    lambda_pde=trainer.lambda_pde,
                    lambda_data=trainer.lambda_data,
                ).item()
            )

        payload = {
            "loss_pde": loss_pde_value,
            "loss_data": loss_data_value,
            "loss_total": loss_total_value,
            "status": "computed_from_final_model_state",
        }
    except Exception as exc:
        payload = {
            "loss_pde": None,
            "loss_data": None,
            "loss_total": None,
            "status": f"unavailable:{type(exc).__name__}",
        }

    save_json(cache_path, payload)
    return payload


def build_loss_history_rows(
    batch_id: str,
    run_id: str,
    cfg: Dict[str, Any],
    loss_rows: List[Dict[str, Any]],
    final_components: Dict[str, Any],
) -> List[Dict[str, Any]]:
    lr = ((cfg.get("training", {}) or {}).get("lr"))
    persisted = any(row["loss_pde"] is not None or row["loss_data"] is not None for row in loss_rows)
    rows: List[Dict[str, Any]] = []
    for index, row in enumerate(loss_rows):
        component_status = "persisted" if persisted else "a7_blocked_missing_per_epoch_persistence"
        loss_pde_value = row["loss_pde"]
        loss_data_value = row["loss_data"]

        if not persisted and index == len(loss_rows) - 1 and final_components:
            loss_pde_value = maybe_float(final_components.get("loss_pde"))
            loss_data_value = maybe_float(final_components.get("loss_data"))
            component_status = "final_epoch_only__a7_blocked"

        rows.append(
            {
                "batch_id": batch_id,
                "run_id": run_id,
                "epoch": row["epoch"],
                "loss_total": row["loss_total"],
                "loss_pde": loss_pde_value if loss_pde_value is not None else math.nan,
                "loss_data": loss_data_value if loss_data_value is not None else math.nan,
                "lr_current": lr,
                "loss_components_status": component_status,
            }
        )
    return rows


def compute_quantile_row(batch_id: str, run_id: str, run_dir: Path) -> Dict[str, Any]:
    residual_grid = load_csv_array(run_dir / "residual_magnitude_physical.csv", ndmin=2)
    loss_mask = load_csv_array(run_dir / "loss_mask_physical.csv", ndmin=2)
    payload = {column: "" for column in QUANTILE_COLUMNS}
    payload["batch_id"] = batch_id
    payload["run_id"] = run_id

    if residual_grid is None:
        return payload

    if loss_mask is not None and loss_mask.shape == residual_grid.shape:
        values = residual_grid[loss_mask > 0.5]
    else:
        values = residual_grid.reshape(-1)

    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return payload

    percentiles = np.percentile(values, [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100])
    payload.update(
        {
            "q00": percentiles[0],
            "q01": percentiles[1],
            "q05": percentiles[2],
            "q10": percentiles[3],
            "q25": percentiles[4],
            "q50": percentiles[5],
            "q75": percentiles[6],
            "q90": percentiles[7],
            "q95": percentiles[8],
            "q99": percentiles[9],
            "q100": percentiles[10],
            "residual_std": float(np.std(values)),
        }
    )
    return payload


def build_centerline_rows(batch_id: str, run_id: str, run_dir: Path, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    x_coords = load_csv_array(run_dir / "x_coords_physical.csv")
    y_coords = load_csv_array(run_dir / "y_coords_physical.csv")
    if x_coords is None or y_coords is None:
        return []

    wavefield_path = run_dir / "wavefield.npy"
    if not wavefield_path.exists():
        return []

    wavefield = np.load(wavefield_path)
    reference_wavefield = np.load(run_dir / "reference_wavefield.npy") if (run_dir / "reference_wavefield.npy").exists() else None
    residual_grid = load_csv_array(run_dir / "residual_magnitude_physical.csv", ndmin=2)
    velocity_grid = load_csv_array(run_dir / "velocity_physical.csv", ndmin=2)
    loss_mask = load_csv_array(run_dir / "loss_mask_physical.csv", ndmin=2)

    pred_phys = crop_to_physical(np.asarray(wavefield), cfg, len(x_coords), len(y_coords))
    ref_phys = None
    if reference_wavefield is not None:
        ref_phys = crop_to_physical(np.asarray(reference_wavefield), cfg, len(x_coords), len(y_coords))

    source_pos = (cfg.get("physics", {}) or {}).get("source_pos", [0.5, 0.5])
    x_index = int(np.argmin(np.abs(np.asarray(x_coords) - float(source_pos[0]))))
    y_index = int(np.argmin(np.abs(np.asarray(y_coords) - float(source_pos[1]))))

    rows: List[Dict[str, Any]] = []
    for j, coord in enumerate(np.asarray(x_coords)):
        pred_value = pred_phys[y_index, j]
        ref_value = ref_phys[y_index, j] if ref_phys is not None else None
        residual_value = (
            abs(pred_value - ref_value)
            if ref_value is not None
            else (residual_grid[y_index, j] if residual_grid is not None else math.nan)
        )
        rows.append(
            {
                "batch_id": batch_id,
                "run_id": run_id,
                "axis": "x",
                "s": float(coord),
                "pred_re": float(np.real(pred_value)),
                "pred_im": float(np.imag(pred_value)),
                "ref_re": float(np.real(ref_value)) if ref_value is not None else math.nan,
                "ref_im": float(np.imag(ref_value)) if ref_value is not None else math.nan,
                "pred_abs": float(np.abs(pred_value)),
                "ref_abs": float(np.abs(ref_value)) if ref_value is not None else math.nan,
                "residual_abs": float(residual_value),
                "velocity": float(velocity_grid[y_index, j]) if velocity_grid is not None else math.nan,
                "loss_mask": float(loss_mask[y_index, j]) if loss_mask is not None else math.nan,
            }
        )

    for i, coord in enumerate(np.asarray(y_coords)):
        pred_value = pred_phys[i, x_index]
        ref_value = ref_phys[i, x_index] if ref_phys is not None else None
        residual_value = (
            abs(pred_value - ref_value)
            if ref_value is not None
            else (residual_grid[i, x_index] if residual_grid is not None else math.nan)
        )
        rows.append(
            {
                "batch_id": batch_id,
                "run_id": run_id,
                "axis": "y",
                "s": float(coord),
                "pred_re": float(np.real(pred_value)),
                "pred_im": float(np.imag(pred_value)),
                "ref_re": float(np.real(ref_value)) if ref_value is not None else math.nan,
                "ref_im": float(np.imag(ref_value)) if ref_value is not None else math.nan,
                "pred_abs": float(np.abs(pred_value)),
                "ref_abs": float(np.abs(ref_value)) if ref_value is not None else math.nan,
                "residual_abs": float(residual_value),
                "velocity": float(velocity_grid[i, x_index]) if velocity_grid is not None else math.nan,
                "loss_mask": float(loss_mask[i, x_index]) if loss_mask is not None else math.nan,
            }
        )
    return rows


def normalize_failure_rows(failure_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized = []
    for row in failure_rows:
        payload = {column: row.get(column, "") for column in FAILURE_COLUMNS}
        if not payload["last_logged_epoch"] and payload["traceback_path"]:
            log_path = Path(payload["traceback_path"])
            if log_path.exists():
                with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                    payload["last_logged_epoch"] = extract_last_logged_epoch(f.read())
        normalized.append(payload)
    return normalized


def build_report_row(
    entry: Dict[str, Any],
    cfg: Dict[str, Any],
    summary: Dict[str, Any],
    metrics: Dict[str, Any],
    loss_rows: List[Dict[str, Any]],
    final_components: Dict[str, Any],
) -> Dict[str, Any]:
    manifest_row = entry["manifest"]
    run_dir = entry["run_dir"]
    runtime_metadata = read_runtime_metadata(run_dir)

    row = {column: "" for column in REPORT_COLUMNS}
    row["batch_id"] = manifest_row["batch_id"]
    row["run_id"] = manifest_row["run_id"]
    row["group_tag"] = manifest_row["run_id"]
    row["output_dir"] = str(run_dir)
    row["effective_config_path"] = manifest_row.get("effective_config_path", str(run_dir / "config_merged.yaml"))
    row["git_sha_int"] = manifest_row.get("git_sha_int", "")
    row["launched_at"] = manifest_row.get("launched_at", "")
    row["machine_host"] = manifest_row.get("machine_host", "")
    row.update(derive_axes(cfg, run_dir / "config_merged.yaml"))

    losses = [row_["loss_total"] for row_ in loss_rows if row_["loss_total"] is not None]
    first_loss = losses[0] if losses else maybe_float(summary.get("initial_loss"))
    final_loss = maybe_float(summary.get("final_loss", metrics.get("final_loss")))
    if final_loss is None and losses:
        final_loss = losses[-1]
    min_epoch = int(np.argmin(losses) + 1) if losses else ""

    residual_rmse = maybe_float(summary.get("residual_rmse_evaluation", metrics.get("residual_rmse_evaluation")))
    reference_rmse = maybe_float(summary.get("reference_residual_rmse"))

    runtime_sec = maybe_float(summary.get("runtime_sec"))
    if runtime_sec is None:
        runtime_sec = maybe_float(runtime_metadata.get("train_runtime_sec", runtime_metadata.get("reference_only_runtime_sec")))

    peak_mem_mb = maybe_float(runtime_metadata.get("train_peak_mem_mb"))
    if peak_mem_mb is None:
        peak_mem_mb = maybe_float(runtime_metadata.get("reference_only_peak_mem_mb"))

    ref_runtime = maybe_float(runtime_metadata.get("ref_solve_runtime_sec"))
    if ref_runtime is None and manifest_row.get("reference_output_dir"):
        ref_runtime = maybe_float(
            read_runtime_metadata(Path(manifest_row["reference_output_dir"])).get("reference_runtime_sec")
        )

    nan_detected = any((loss is None or not math.isfinite(loss)) for loss in losses) if losses else False
    tolerance = 1.0e-12
    converged = ""
    if final_loss is not None:
        if final_loss <= tolerance:
            converged = True
        elif losses:
            converged = bool(
                first_loss is not None
                and final_loss < first_loss
                and isinstance(min_epoch, int)
                and min_epoch >= max(1, int(math.ceil(0.8 * len(losses))))
            )
        else:
            converged = False

    row.update(
        {
            "residual_rmse_evaluation": residual_rmse,
            "residual_p95_evaluation": maybe_float(summary.get("residual_p95_evaluation", metrics.get("residual_p95_evaluation"))),
            "residual_max_evaluation": maybe_float(summary.get("residual_max_evaluation", metrics.get("residual_max_evaluation"))),
            "residual_mean_evaluation": maybe_float(summary.get("residual_mean_evaluation", metrics.get("residual_mean_evaluation"))),
            "final_loss_total": final_loss,
            "final_loss_pde": maybe_float(final_components.get("loss_pde")),
            "final_loss_data": maybe_float(final_components.get("loss_data")),
            "first_loss_total": first_loss,
            "loss_reduction_ratio": maybe_float(summary.get("loss_reduction_ratio", metrics.get("loss_reduction_ratio"))),
            "min_loss_total_epoch": min_epoch,
            "reference_available": maybe_bool(summary.get("reference_available")),
            "rel_l2_to_reference": maybe_float(summary.get("rel_l2_to_reference")),
            "amp_mae_to_reference": maybe_float(summary.get("amp_mae_to_reference")),
            "phase_mae_to_reference": maybe_float(summary.get("phase_mae_to_reference")),
            "reference_residual_rmse": reference_rmse,
            "reference_residual_max": maybe_float(summary.get("reference_residual_max")),
            "rmse_gap_to_reference": (
                residual_rmse - reference_rmse
                if residual_rmse is not None and reference_rmse is not None
                else math.nan
            ),
            "rmse_gap_ratio": (
                residual_rmse / reference_rmse
                if residual_rmse is not None and reference_rmse not in (None, 0.0)
                else math.nan
            ),
            "runtime_sec": runtime_sec,
            "peak_mem_mb": peak_mem_mb,
            "epochs_completed": len(losses),
            "converged": converged,
            "nan_detected": nan_detected,
            "early_stopped": bool(losses and len(losses) < int(cfg.get("training", {}).get("epochs", len(losses)))),
            "ref_solve_runtime_sec": ref_runtime,
        }
    )
    return row


def pareto_layers(rows: List[Dict[str, Any]], metric_x: str, metric_y: str) -> Dict[str, int]:
    remaining = [
        row for row in rows
        if is_finite(row.get(metric_x)) and is_finite(row.get(metric_y))
    ]
    ranks: Dict[str, int] = {}
    rank = 1
    while remaining:
        front: List[Dict[str, Any]] = []
        for candidate in remaining:
            x_c = maybe_float(candidate[metric_x])
            y_c = maybe_float(candidate[metric_y])
            dominated = False
            for other in remaining:
                if other is candidate:
                    continue
                x_o = maybe_float(other[metric_x])
                y_o = maybe_float(other[metric_y])
                if x_o is None or y_o is None or x_c is None or y_c is None:
                    continue
                if (x_o <= x_c and y_o <= y_c) and (x_o < x_c or y_o < y_c):
                    dominated = True
                    break
            if not dominated:
                front.append(candidate)
        for row in front:
            ranks[row["run_id"]] = rank
        remaining = [row for row in remaining if row not in front]
        rank += 1
    return ranks


def metric_pairs_for_batch(batch_id: str) -> List[tuple[str, str]]:
    pairs = [("runtime_sec", "residual_rmse_evaluation")]
    if batch_id == "B4":
        pairs.append(("total_params", "residual_rmse_evaluation"))
    elif batch_id == "B6":
        pairs.append(("final_loss_pde", "final_loss_data"))
    elif batch_id == "B10":
        pairs.append(("grid_nx", "reference_residual_rmse"))
    return pairs


def build_pareto_rows(report_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for batch_id in sorted({row["batch_id"] for row in report_rows}):
        batch_rows = [row for row in report_rows if row["batch_id"] == batch_id]
        for metric_x, metric_y in metric_pairs_for_batch(batch_id):
            ranks = pareto_layers(batch_rows, metric_x, metric_y)
            for row in batch_rows:
                if row["run_id"] not in ranks:
                    continue
                rows.append(
                    {
                        "batch_id": batch_id,
                        "run_id": row["run_id"],
                        "metric_x_name": metric_x,
                        "metric_x": row[metric_x],
                        "metric_y_name": metric_y,
                        "metric_y": row[metric_y],
                        "is_pareto_front": ranks[row["run_id"]] == 1,
                        "pareto_rank": ranks[row["run_id"]],
                    }
                )
    return rows


def build_report_assets(output_root: Path) -> Dict[str, Any]:
    meta_root = output_root / "_meta"
    manifest_rows = load_manifest(meta_root)
    batch_specs = load_batch_specs()

    report_rows: List[Dict[str, Any]] = []
    loss_history_rows: List[Dict[str, Any]] = []
    quantile_rows: List[Dict[str, Any]] = []
    centerline_rows: List[Dict[str, Any]] = []
    reference_only_rows: List[Dict[str, Any]] = []
    run_bundles: List[Dict[str, Any]] = []

    for entry in collect_run_entries(manifest_rows):
        run_dir = entry["run_dir"]
        cfg = load_yaml(run_dir / "config_merged.yaml")
        summary = load_json(run_dir / "summary.json")
        metrics_rows = read_dict_rows(run_dir / "metrics_summary.csv")
        metrics = metrics_rows[0] if metrics_rows else {}
        loss_rows = read_loss_rows(run_dir / "losses.csv")
        final_components = compute_final_loss_components(run_dir, cfg)
        report_row = build_report_row(entry, cfg, summary, metrics, loss_rows, final_components)
        report_rows.append(report_row)
        loss_history_rows.extend(
            build_loss_history_rows(entry["manifest"]["batch_id"], entry["manifest"]["run_id"], cfg, loss_rows, final_components)
        )
        quantile_rows.append(compute_quantile_row(entry["manifest"]["batch_id"], entry["manifest"]["run_id"], run_dir))
        centerline_rows.extend(build_centerline_rows(entry["manifest"]["batch_id"], entry["manifest"]["run_id"], run_dir, cfg))
        run_bundles.append(
            {
                "manifest": entry["manifest"],
                "run_dir": run_dir,
                "cfg": cfg,
                "summary": summary,
                "metrics": metrics,
                "loss_rows": loss_rows,
                "final_components": final_components,
                "report_row": report_row,
            }
        )

        if not (run_dir / "model_state.pt").exists():
            reference_only_rows.append(
                {
                    "run_id": entry["manifest"]["run_id"],
                    "velocity_model": report_row["velocity_model"],
                    "omega": report_row["omega"],
                    "grid_nx": report_row["grid_nx"],
                    "pml_width": report_row["pml_width"],
                    "eikonal_precision": report_row["eikonal_precision"],
                    "reference_residual_rmse": report_row["reference_residual_rmse"],
                    "reference_residual_max": report_row["reference_residual_max"],
                    "solve_runtime_sec": report_row["runtime_sec"],
                    "peak_mem_mb": report_row["peak_mem_mb"],
                }
            )

    pareto_rows = build_pareto_rows(report_rows)
    failure_rows = normalize_failure_rows(read_dict_rows(meta_root / "failures.csv"))
    return {
        "batch_specs": batch_specs,
        "manifest_rows": manifest_rows,
        "run_bundles": run_bundles,
        "report_rows": report_rows,
        "loss_history_rows": loss_history_rows,
        "quantile_rows": quantile_rows,
        "centerline_rows": centerline_rows,
        "pareto_rows": pareto_rows,
        "reference_only_rows": reference_only_rows,
        "failure_rows": failure_rows,
    }


def figure_root(output_root: Path) -> Path:
    return output_root / "_reports" / "matrix_figures"


def save_figure(fig: plt.Figure, png_path: Path) -> Path:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    fig.savefig(png_path.with_suffix(".pdf"))
    plt.close(fig)
    return png_path


def report_rows_by_batch(report_rows: List[Dict[str, Any]], batch_id: str) -> List[Dict[str, Any]]:
    return [row for row in report_rows if row["batch_id"] == batch_id]


def bundle_map_by_run(run_bundles: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {bundle["manifest"]["run_id"]: bundle for bundle in run_bundles}


def centerline_map(centerline_rows: List[Dict[str, Any]], batch_id: str, axis: str) -> Dict[str, List[Dict[str, Any]]]:
    mapping: Dict[str, List[Dict[str, Any]]] = {}
    for row in centerline_rows:
        if row["batch_id"] != batch_id or row["axis"] != axis:
            continue
        mapping.setdefault(row["run_id"], []).append(row)
    for rows in mapping.values():
        rows.sort(key=lambda row: maybe_float(row["s"]) or 0.0)
    return mapping


def write_global_heatmap(report_rows: List[Dict[str, Any]], out_dir: Path) -> Optional[Path]:
    batches = sorted({row["batch_id"] for row in report_rows})
    if not batches:
        return None

    rows_by_batch = {batch: sorted(report_rows_by_batch(report_rows, batch), key=lambda row: row["run_id"]) for batch in batches}
    max_runs = max(len(rows) for rows in rows_by_batch.values())
    if max_runs == 0:
        return None

    data = np.full((len(batches), max_runs), np.nan, dtype=np.float64)
    labels = [""] * max_runs
    for batch_index, batch in enumerate(batches):
        for run_index, row in enumerate(rows_by_batch[batch]):
            labels[run_index] = row["run_id"]
            data[batch_index, run_index] = safe_log10(row["residual_rmse_evaluation"])

    fig, ax = plt.subplots(figsize=(max(8, max_runs * 0.7), max(3, len(batches) * 0.65)))
    image = ax.imshow(data, aspect="auto", cmap="viridis")
    ax.set_xticks(range(max_runs))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(batches)))
    ax.set_yticklabels(batches)
    ax.set_title("G1 residual heatmap (log10 residual_rmse_evaluation)")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    return save_figure(fig, out_dir / "global" / "G1__residual_rmse_heatmap.png")


def write_runtime_boxplot(report_rows: List[Dict[str, Any]], out_dir: Path) -> Optional[Path]:
    batches = sorted({row["batch_id"] for row in report_rows})
    data = []
    labels = []
    for batch in batches:
        values = [maybe_float(row["runtime_sec"]) for row in report_rows_by_batch(report_rows, batch)]
        values = [value for value in values if value is not None]
        if values:
            data.append(values)
            labels.append(batch)
    if not data:
        return None

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.7), 4))
    ax.boxplot(data, tick_labels=labels)
    ax.set_ylabel("runtime_sec")
    ax.set_title("G2 runtime distribution by batch")
    return save_figure(fig, out_dir / "global" / "G2__runtime_boxplot.png")


def write_status_sankey(
    report_rows: List[Dict[str, Any]],
    manifest_rows: List[Dict[str, str]],
    failure_rows: List[Dict[str, Any]],
    out_dir: Path,
) -> Optional[Path]:
    active_runs = [row for row in manifest_rows if row.get("status") != "blocked"]
    if not active_runs:
        return None

    completed = len(report_rows)
    failed = len({(row.get("batch_id"), row.get("run_id")) for row in failure_rows})
    nan_runs = sum(1 for row in report_rows if maybe_bool(row.get("nan_detected")))
    converged = sum(1 for row in report_rows if maybe_bool(row.get("converged")))
    other_completed = max(0, completed - converged - nan_runs)
    total = len(active_runs)
    pending = max(0, total - completed - failed)

    fig, ax = plt.subplots(figsize=(7, 5))
    sankey = Sankey(ax=ax, unit=None, format="%.0f")
    sankey.add(
        flows=[total, -converged, -other_completed, -nan_runs, -failed, -pending],
        labels=["launched", "converged", "completed", "nan", "failed", "pending"],
        orientations=[0, 1, 1, -1, -1, 0],
        pathlengths=[0.25] * 6,
    )
    sankey.finish()
    ax.set_title("G3 run health flow")
    return save_figure(fig, out_dir / "global" / "G3__status_sankey.png")


def numeric_matrix(report_rows: List[Dict[str, Any]], columns: Sequence[str]) -> tuple[np.ndarray, List[str], List[int]]:
    kept_columns: List[str] = []
    column_vectors: List[np.ndarray] = []
    for index, row in enumerate(report_rows):
        for column in columns:
            if column in {"pml_R0"}:
                value = maybe_float(row.get(column))
                row[f"__{column}_log10"] = math.log10(value) if value and value > 0.0 else math.nan
    resolved_columns = [f"__{column}_log10" if column == "pml_R0" else column for column in columns]

    for column in resolved_columns:
        values = np.array([maybe_float(row.get(column)) for row in report_rows], dtype=np.float64)
        finite = np.isfinite(values)
        finite_values = values[finite]
        if finite_values.size < 3:
            continue
        if np.allclose(finite_values, finite_values[0]):
            continue
        kept_columns.append(column.replace("__", "").replace("_log10", ""))
        column_vectors.append(values)

    if not kept_columns:
        return np.empty((0, 0)), [], []

    stacked = np.vstack(column_vectors).T
    valid_row_mask = np.all(np.isfinite(stacked), axis=1)
    valid_indices = [index for index, keep in enumerate(valid_row_mask.tolist()) if keep]
    if not valid_indices:
        return np.empty((0, 0)), [], []

    matrix = stacked[valid_indices]
    return matrix, kept_columns, valid_indices


def write_pca_projection(report_rows: List[Dict[str, Any]], out_dir: Path) -> Optional[Path]:
    candidate_columns = [
        "omega",
        "grid_nx",
        "pml_width",
        "pml_power",
        "pml_R0",
        "source_pos_x",
        "source_pos_y",
        "nsno_blocks",
        "nsno_channels",
        "fno_modes",
        "epochs",
        "lr",
        "lambda_pde",
        "lambda_data",
        "source_mask_radius",
    ]
    matrix, labels, indices = numeric_matrix(report_rows, candidate_columns)
    if matrix.shape[0] < 3 or matrix.shape[1] < 2:
        return None

    means = np.nanmean(matrix, axis=0)
    stds = np.nanstd(matrix, axis=0)
    stds[stds == 0.0] = 1.0
    standardized = (matrix - means) / stds
    u, s, _ = np.linalg.svd(standardized, full_matrices=False)
    components = u[:, :2] * s[:2]

    fig, ax = plt.subplots(figsize=(6, 5))
    colors = [safe_log10(report_rows[index]["residual_rmse_evaluation"]) for index in indices]
    scatter = ax.scatter(components[:, 0], components[:, 1], c=colors, cmap="viridis", s=50)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("G4 PCA of numeric config axes")
    fig.colorbar(scatter, ax=ax, label="log10 residual_rmse_evaluation")
    return save_figure(fig, out_dir / "global" / "G4__pca_projection.png")


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def spearman_pair(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 3:
        return math.nan
    xr = rankdata(x[mask])
    yr = rankdata(y[mask])
    return float(np.corrcoef(xr, yr)[0, 1])


def write_spearman_heatmap(report_rows: List[Dict[str, Any]], out_dir: Path) -> Optional[Path]:
    columns = [
        "omega",
        "grid_nx",
        "pml_width",
        "pml_power",
        "pml_R0",
        "source_pos_x",
        "source_pos_y",
        "nsno_blocks",
        "nsno_channels",
        "fno_modes",
        "epochs",
        "lr",
        "lambda_pde",
        "lambda_data",
        "source_mask_radius",
        "residual_rmse_evaluation",
        "residual_p95_evaluation",
        "runtime_sec",
        "rel_l2_to_reference",
        "rmse_gap_ratio",
    ]
    matrix, labels, indices = numeric_matrix(report_rows, columns)
    if matrix.shape[0] < 3 or matrix.shape[1] < 2:
        return None

    corr = np.full((matrix.shape[1], matrix.shape[1]), np.nan, dtype=np.float64)
    for i in range(matrix.shape[1]):
        for j in range(matrix.shape[1]):
            corr[i, j] = spearman_pair(matrix[:, i], matrix[:, j])

    fig, ax = plt.subplots(figsize=(max(8, matrix.shape[1] * 0.5), max(6, matrix.shape[1] * 0.5)))
    image = ax.imshow(corr, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title("G5 Spearman correlation matrix")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    return save_figure(fig, out_dir / "global" / "G5__spearman_heatmap.png")


def write_parallel_coordinates(report_rows: List[Dict[str, Any]], out_dir: Path) -> Optional[Path]:
    columns = [
        "omega",
        "grid_nx",
        "pml_width",
        "nsno_channels",
        "fno_modes",
        "epochs",
        "lambda_data",
        "residual_rmse_evaluation",
        "runtime_sec",
    ]
    finite_rows = [
        row for row in report_rows
        if all(is_finite(row.get(column)) for column in columns)
    ]
    if len(finite_rows) < 3:
        return None

    matrix = np.array([[maybe_float(row[column]) for column in columns] for row in finite_rows], dtype=np.float64)
    mins = np.min(matrix, axis=0)
    maxs = np.max(matrix, axis=0)
    spans = np.where(maxs - mins == 0.0, 1.0, maxs - mins)
    normalized = (matrix - mins) / spans

    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = plt.get_cmap("tab10")
    batches = sorted({row["batch_id"] for row in finite_rows})
    batch_to_color = {batch: cmap(index % 10) for index, batch in enumerate(batches)}
    for index, row in enumerate(finite_rows):
        ax.plot(range(len(columns)), normalized[index], color=batch_to_color[row["batch_id"]], alpha=0.45)
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=30, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("G6 parallel coordinates (normalized)")
    return save_figure(fig, out_dir / "global" / "G6__parallel_coordinates.png")


def write_gap_histogram(report_rows: List[Dict[str, Any]], out_dir: Path) -> Optional[Path]:
    batches = sorted({row["batch_id"] for row in report_rows})
    data = []
    labels = []
    for batch in batches:
        values = [maybe_float(row["rmse_gap_ratio"]) for row in report_rows_by_batch(report_rows, batch)]
        values = [value for value in values if value is not None and math.isfinite(value)]
        if values:
            data.append(values)
            labels.append(batch)
    if not data:
        return None

    fig, ax = plt.subplots(figsize=(8, 4))
    for values, label in zip(data, labels):
        ax.hist(values, bins=20, alpha=0.35, label=label)
    ax.set_xlabel("rmse_gap_ratio")
    ax.set_ylabel("count")
    ax.set_title("G7 rmse gap ratio histogram")
    ax.legend(fontsize=8)
    return save_figure(fig, out_dir / "global" / "G7__gap_histogram.png")


def load_residual_grid(bundle: Dict[str, Any]) -> Optional[np.ndarray]:
    return load_csv_array(bundle["run_dir"] / "residual_magnitude_physical.csv", ndmin=2)


def draw_basic_bar(batch_id: str, rows: List[Dict[str, Any]], out_dir: Path, title: str) -> Optional[Path]:
    if not rows:
        return None
    values = [maybe_float(row["residual_rmse_evaluation"]) for row in rows]
    if not any(value is not None for value in values):
        return None
    fig, ax = plt.subplots(figsize=(max(8, len(rows) * 0.7), 4))
    x = np.arange(len(rows))
    ax.bar(x, [value if value is not None else np.nan for value in values], color="tab:blue")
    ax.set_xticks(x)
    ax.set_xticklabels([row["run_id"] for row in rows], rotation=45, ha="right")
    ax.set_ylabel("residual_rmse_evaluation")
    ax.set_title(title)
    return save_figure(fig, out_dir / batch_id / f"{batch_id}__primary_metric.png")


def plot_batch_b0(rows: List[Dict[str, Any]], batch_spec: Dict[str, Any], out_dir: Path) -> List[Path]:
    rows = sorted(rows, key=lambda row: row["run_id"])
    if not rows:
        return []
    gate_run_id = batch_spec.get("gate_run_id")
    colors = ["tab:red" if row["run_id"] == gate_run_id else "tab:blue" for row in rows]
    fig, ax = plt.subplots(figsize=(max(8, len(rows) * 1.2), 4))
    x = np.arange(len(rows))
    values = [maybe_float(row["residual_rmse_evaluation"]) or 0.0 for row in rows]
    ax.bar(x, values, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels([row["run_id"] for row in rows], rotation=45, ha="right")
    ax.set_ylabel("residual_rmse_evaluation")
    ax.set_title("B0 smoke gate")
    return [save_figure(fig, out_dir / "B0" / "B0__gate_bar.png")]


def plot_batch_b1(rows: List[Dict[str, Any]], bundle_map: Dict[str, Dict[str, Any]], out_dir: Path) -> List[Path]:
    outputs: List[Path] = []
    if not rows:
        return outputs

    media = sorted({row["velocity_model"] for row in rows})
    modes = ["pde", "hybrid"]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(media))
    width = 0.35
    for offset, mode in enumerate(modes):
        values = []
        for medium in media:
            match = next((row for row in rows if row["velocity_model"] == medium and row["supervision_mode"] == mode), None)
            values.append(maybe_float(match["residual_rmse_evaluation"]) if match else math.nan)
        ax.bar(x + (offset - 0.5) * width, values, width=width, label=mode)
    ax.set_xticks(x)
    ax.set_xticklabels(media)
    ax.set_ylabel("residual_rmse_evaluation")
    ax.set_title("B1 media × supervision")
    ax.legend()
    outputs.append(save_figure(fig, out_dir / "B1" / "B1__grouped_bar.png"))

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.reshape(2, 3)
    for ax, row in zip(axes.flat, sorted(rows, key=lambda item: item["run_id"])):
        grid = load_residual_grid(bundle_map[row["run_id"]])
        if grid is None:
            ax.axis("off")
            continue
        image = ax.imshow(np.log10(np.maximum(grid, 1.0e-12)), origin="lower", cmap="inferno")
        ax.set_title(row["run_id"])
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    outputs.append(save_figure(fig, out_dir / "B1" / "B1__residual_collage.png"))
    return outputs


def plot_batch_b2(
    rows: List[Dict[str, Any]],
    centerlines: Dict[str, List[Dict[str, Any]]],
    out_dir: Path,
) -> List[Path]:
    outputs: List[Path] = []
    if not rows:
        return outputs

    grouped = {"pde": [], "hybrid": []}
    for row in rows:
        grouped.setdefault(row["supervision_mode"], []).append(row)

    fig, ax = plt.subplots(figsize=(8, 4))
    for mode, mode_rows in grouped.items():
        mode_rows = sorted(mode_rows, key=lambda row: maybe_float(row["omega"]) or 0.0)
        if not mode_rows:
            continue
        x = [maybe_float(row["omega"]) for row in mode_rows]
        y = [maybe_float(row["residual_rmse_evaluation"]) for row in mode_rows]
        ax.plot(x, y, marker="o", label=mode)
    ax.set_yscale("log")
    ax.set_xlabel("omega")
    ax.set_ylabel("residual_rmse_evaluation")
    ax.set_title("B2 omega sweep")
    ax.legend()
    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    sorted_rows = sorted(rows, key=lambda row: maybe_float(row["omega"]) or 0.0)
    ax_top.set_xticks([maybe_float(row["omega"]) for row in sorted_rows])
    ax_top.set_xticklabels([f"{maybe_float(row['ppw']):.1f}" if is_finite(row["ppw"]) else "NA" for row in sorted_rows], rotation=45, ha="left")
    ax_top.set_xlabel("ppw")
    outputs.append(save_figure(fig, out_dir / "B2" / "B2__omega_line.png"))

    fig, ax = plt.subplots(figsize=(8, 4))
    rows_sorted = sorted(rows, key=lambda row: maybe_float(row["omega"]) or 0.0)
    x = [maybe_float(row["omega"]) for row in rows_sorted]
    amp = [maybe_float(row["amp_mae_to_reference"]) for row in rows_sorted]
    phase = [maybe_float(row["phase_mae_to_reference"]) for row in rows_sorted]
    ax.plot(x, amp, marker="o", color="tab:blue", label="amp_mae")
    ax.set_xlabel("omega")
    ax.set_ylabel("amp_mae_to_reference", color="tab:blue")
    ax.tick_params(axis="y", labelcolor="tab:blue")
    ax2 = ax.twinx()
    ax2.plot(x, phase, marker="s", color="tab:orange", label="phase_mae")
    ax2.set_ylabel("phase_mae_to_reference", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    ax.set_title("B2 amplitude / phase error")
    outputs.append(save_figure(fig, out_dir / "B2" / "B2__amp_phase.png"))

    selected = []
    for target in ("omega30", "omega60", "omega90"):
        match = next((row for row in rows_sorted if target in row["run_id"] and row["supervision_mode"] == "pde"), None)
        if match:
            selected.append(match)
    if selected:
        fig, axes = plt.subplots(len(selected), 1, figsize=(8, 3.2 * len(selected)), sharex=True)
        axes = np.atleast_1d(axes)
        for ax, row in zip(axes, selected):
            line = centerlines.get(row["run_id"], [])
            x_vals = [maybe_float(item["s"]) for item in line]
            pred = [maybe_float(item["pred_abs"]) for item in line]
            ref = [maybe_float(item["ref_abs"]) for item in line]
            ax.plot(x_vals, pred, label="pred")
            if any(value is not None and math.isfinite(value) for value in ref):
                ax.plot(x_vals, ref, label="ref")
            ax.set_title(row["run_id"])
        axes[0].legend()
        outputs.append(save_figure(fig, out_dir / "B2" / "B2__centerline_compare.png"))

    return outputs


def plot_batch_b3(rows: List[Dict[str, Any]], out_dir: Path) -> List[Path]:
    outputs: List[Path] = []
    if not rows:
        return outputs

    n_values = sorted({int(row["grid_nx"]) for row in rows if is_finite(row["grid_nx"])})
    pml_values = sorted({int(row["pml_width"]) for row in rows if is_finite(row["pml_width"])})
    heat = np.full((len(n_values), len(pml_values)), np.nan)
    for row in rows:
        n = int(row["grid_nx"])
        pml = int(row["pml_width"])
        heat[n_values.index(n), pml_values.index(pml)] = safe_log10(row["residual_rmse_evaluation"])
    fig, ax = plt.subplots(figsize=(6, 4))
    image = ax.imshow(heat, origin="lower", cmap="viridis")
    ax.set_xticks(range(len(pml_values)))
    ax.set_xticklabels(pml_values)
    ax.set_yticks(range(len(n_values)))
    ax.set_yticklabels(n_values)
    ax.set_xlabel("pml_width")
    ax.set_ylabel("grid_nx")
    ax.set_title("B3 log10 residual heatmap")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    outputs.append(save_figure(fig, out_dir / "B3" / "B3__grid_pml_heatmap.png"))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    slice_rows = [row for row in rows if int(row["pml_width"]) == 16]
    slice_rows = sorted(slice_rows, key=lambda row: int(row["grid_nx"]))
    axes[0].plot([int(row["grid_nx"]) for row in slice_rows], [maybe_float(row["residual_rmse_evaluation"]) for row in slice_rows], marker="o")
    axes[0].plot([int(row["grid_nx"]) for row in slice_rows], [maybe_float(row["reference_residual_rmse"]) for row in slice_rows], marker="s")
    axes[0].set_xlabel("grid_nx")
    axes[0].set_ylabel("rmse")
    axes[0].set_title("fixed pml=16 slice")
    fixed_n_rows = [row for row in rows if int(row["grid_nx"]) == 128]
    fixed_n_rows = sorted(fixed_n_rows, key=lambda row: int(row["pml_width"]))
    axes[1].scatter(
        [maybe_float(row["pml_thickness_in_wavelengths"]) for row in fixed_n_rows],
        [maybe_float(row["residual_max_evaluation"]) for row in fixed_n_rows],
        s=60,
    )
    axes[1].set_xlabel("pml_thickness_in_wavelengths")
    axes[1].set_ylabel("residual_max_evaluation")
    axes[1].set_title("fixed N=128")
    outputs.append(save_figure(fig, out_dir / "B3" / "B3__slices.png"))
    return outputs


def plot_batch_b4(rows: List[Dict[str, Any]], pareto_rows: List[Dict[str, Any]], out_dir: Path) -> List[Path]:
    outputs: List[Path] = []
    if not rows:
        return outputs

    pareto_lookup = {
        row["run_id"]: row for row in pareto_rows
        if row["batch_id"] == "B4" and row["metric_x_name"] == "runtime_sec"
    }
    fig, ax = plt.subplots(figsize=(6, 5))
    for row in rows:
        runtime = maybe_float(row["runtime_sec"])
        rmse = maybe_float(row["residual_rmse_evaluation"])
        if runtime is None or rmse is None:
            continue
        color = "tab:red" if pareto_lookup.get(row["run_id"], {}).get("is_pareto_front") else "tab:blue"
        ax.scatter(runtime, rmse, color=color, s=70)
        ax.annotate(row["run_id"], (runtime, rmse), fontsize=7)
    ax.set_xlabel("runtime_sec")
    ax.set_ylabel("residual_rmse_evaluation")
    ax.set_title("B4 Pareto runtime vs accuracy")
    outputs.append(save_figure(fig, out_dir / "B4" / "B4__pareto.png"))

    fig, ax = plt.subplots(figsize=(6, 4))
    sorted_rows = sorted(rows, key=lambda row: maybe_float(row["total_params"]) or math.inf)
    ax.plot([maybe_float(row["total_params"]) for row in sorted_rows], [maybe_float(row["residual_rmse_evaluation"]) for row in sorted_rows], marker="o")
    ax.set_xscale("log")
    ax.set_xlabel("total_params")
    ax.set_ylabel("residual_rmse_evaluation")
    ax.set_title("B4 parameter count vs accuracy")
    outputs.append(save_figure(fig, out_dir / "B4" / "B4__params_curve.png"))
    return outputs


def interpolate_loss_curve(loss_rows: List[Dict[str, Any]], steps: int = 100) -> np.ndarray:
    values = np.asarray([maybe_float(row["loss_total"]) for row in loss_rows], dtype=np.float64)
    if values.size == 0:
        return np.full(steps, np.nan)
    if values.size == 1:
        return np.full(steps, values[0])
    source_x = np.linspace(0.0, 1.0, values.size)
    target_x = np.linspace(0.0, 1.0, steps)
    return np.interp(target_x, source_x, values)


def plot_batch_b5(rows: List[Dict[str, Any]], run_bundles: List[Dict[str, Any]], out_dir: Path) -> List[Path]:
    outputs: List[Path] = []
    if not rows:
        return outputs

    lr_values = sorted({maybe_float(row["lr"]) for row in rows if is_finite(row["lr"])})
    epoch_values = sorted({int(row["epochs"]) for row in rows if is_finite(row["epochs"])})
    heat = np.full((len(lr_values), len(epoch_values)), np.nan)
    for row in rows:
        lr = maybe_float(row["lr"])
        epochs = int(row["epochs"])
        heat[lr_values.index(lr), epoch_values.index(epochs)] = safe_log10(row["final_loss_total"])
    fig, ax = plt.subplots(figsize=(7, 4))
    image = ax.imshow(heat, origin="lower", cmap="magma")
    ax.set_xticks(range(len(epoch_values)))
    ax.set_xticklabels(epoch_values)
    ax.set_yticks(range(len(lr_values)))
    ax.set_yticklabels([f"{value:.0e}" for value in lr_values])
    ax.set_xlabel("epochs")
    ax.set_ylabel("lr")
    ax.set_title("B5 log10 final_loss_total")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    outputs.append(save_figure(fig, out_dir / "B5" / "B5__optim_heatmap.png"))

    bundles_by_run = bundle_map_by_run(run_bundles)
    rows_sorted = sorted(rows, key=lambda row: (maybe_float(row["lr"]) or 0.0, int(row["epochs"])))
    n_cols = 4
    n_rows = math.ceil(len(rows_sorted) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3.2 * n_rows))
    axes = np.atleast_1d(axes).reshape(n_rows, n_cols)
    for ax, row in zip(axes.flat, rows_sorted):
        loss_rows = bundles_by_run[row["run_id"]]["loss_rows"]
        epochs = [item["epoch"] for item in loss_rows]
        values = [item["loss_total"] for item in loss_rows]
        ax.plot(epochs, values)
        ax.set_yscale("log")
        ax.set_title(row["run_id"])
    for ax in axes.flat[len(rows_sorted):]:
        ax.axis("off")
    outputs.append(save_figure(fig, out_dir / "B5" / "B5__loss_panels.png"))

    grouped_by_lr: Dict[float, List[np.ndarray]] = {}
    for row in rows:
        lr = maybe_float(row["lr"])
        grouped_by_lr.setdefault(lr, []).append(interpolate_loss_curve(bundles_by_run[row["run_id"]]["loss_rows"]))
    best_lr = min(
        grouped_by_lr,
        key=lambda lr: np.nanmedian([maybe_float(row["final_loss_total"]) for row in rows if maybe_float(row["lr"]) == lr]),
    )
    curves = np.vstack(grouped_by_lr[best_lr])
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.linspace(0.0, 1.0, curves.shape[1])
    ax.fill_between(x, np.nanpercentile(curves, 10, axis=0), np.nanpercentile(curves, 90, axis=0), alpha=0.3)
    ax.plot(x, np.nanpercentile(curves, 50, axis=0), linewidth=2)
    ax.set_yscale("log")
    ax.set_xlabel("normalized training progress")
    ax.set_ylabel("loss_total")
    ax.set_title(f"B5 best-lr envelope ({best_lr:.0e})")
    outputs.append(save_figure(fig, out_dir / "B5" / "B5__loss_envelope.png"))
    return outputs


def plot_batch_b6(rows: List[Dict[str, Any]], out_dir: Path) -> List[Path]:
    outputs: List[Path] = []
    if not rows:
        return outputs

    fig, ax = plt.subplots(figsize=(7, 4))
    plotted = False
    for row in rows:
        lambda_pde = maybe_float(row["lambda_pde"])
        lambda_data = maybe_float(row["lambda_data"])
        if lambda_pde is None or lambda_data is None or lambda_data <= 0.0:
            continue
        ratio = math.log10(lambda_pde / lambda_data)
        rmse = maybe_float(row["residual_rmse_evaluation"])
        if rmse is None:
            continue
        point_size = 60.0
        rel_l2 = maybe_float(row["rel_l2_to_reference"])
        if rel_l2 is not None and math.isfinite(rel_l2):
            point_size = max(30.0, 30.0 + 200.0 * rel_l2)
        ax.scatter(ratio, rmse, s=point_size, label=row["run_id"])
        plotted = True
    ax.set_xlabel("log10(lambda_pde / lambda_data)")
    ax.set_ylabel("residual_rmse_evaluation")
    ax.set_title("B6 hybrid weight sweep")
    if plotted:
        outputs.append(save_figure(fig, out_dir / "B6" / "B6__lambda_scatter.png"))
    else:
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(8, len(rows) * 0.7), 4))
    x = np.arange(len(rows))
    ax.bar(x - 0.18, [maybe_float(row["final_loss_pde"]) or 0.0 for row in rows], width=0.36, label="final_loss_pde")
    ax.bar(x + 0.18, [maybe_float(row["final_loss_data"]) or 0.0 for row in rows], width=0.36, label="final_loss_data")
    ax.set_xticks(x)
    ax.set_xticklabels([row["run_id"] for row in rows], rotation=45, ha="right")
    ax.set_yscale("log")
    ax.set_title("B6 final loss decomposition")
    ax.legend()
    outputs.append(save_figure(fig, out_dir / "B6" / "B6__loss_decomposition.png"))
    return outputs


def plot_batch_b7(rows: List[Dict[str, Any]], centerlines: Dict[str, List[Dict[str, Any]]], out_dir: Path) -> List[Path]:
    outputs: List[Path] = []
    if not rows:
        return outputs

    fig, ax = plt.subplots(figsize=(5, 5))
    colors = [maybe_float(row["residual_rmse_evaluation"]) for row in rows]
    scatter = ax.scatter(
        [maybe_float(row["source_pos_x"]) for row in rows],
        [maybe_float(row["source_pos_y"]) for row in rows],
        c=colors,
        cmap="viridis",
        s=90,
    )
    ax.add_patch(Rectangle((0.0, 0.0), 1.0, 1.0, fill=False, linewidth=1.5, color="black"))
    for row in rows:
        ax.annotate(row["run_id"], (maybe_float(row["source_pos_x"]), maybe_float(row["source_pos_y"])), fontsize=7)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("B7 source position map")
    fig.colorbar(scatter, ax=ax, label="residual_rmse_evaluation")
    outputs.append(save_figure(fig, out_dir / "B7" / "B7__source_map.png"))

    fig, ax = plt.subplots(figsize=(6, 4))
    distances = []
    residuals = []
    for row in rows:
        x = maybe_float(row["source_pos_x"]) or 0.0
        y = maybe_float(row["source_pos_y"]) or 0.0
        distances.append(min(x, y, 1.0 - x, 1.0 - y))
        residuals.append(maybe_float(row["residual_max_evaluation"]))
    ax.scatter(distances, residuals, s=70)
    ax.set_xlabel("distance to nearest boundary")
    ax.set_ylabel("residual_max_evaluation")
    ax.set_title("B7 boundary proximity")
    outputs.append(save_figure(fig, out_dir / "B7" / "B7__boundary_distance.png"))

    center_run = next((row["run_id"] for row in rows if row["run_id"] == "center"), None)
    corner_run = next((row["run_id"] for row in rows if "corner" in row["run_id"]), None)
    if center_run and corner_run:
        fig, ax = plt.subplots(figsize=(7, 4))
        for run_id, label in [(center_run, "center"), (corner_run, "corner")]:
            line = centerlines.get(run_id, [])
            ax.plot([maybe_float(item["s"]) for item in line], [maybe_float(item["pred_abs"]) for item in line], label=label)
        ax.set_title("B7 centerline compare")
        ax.legend()
        outputs.append(save_figure(fig, out_dir / "B7" / "B7__centerline_compare.png"))
    return outputs


def plot_batch_b9(rows: List[Dict[str, Any]], out_dir: Path) -> List[Path]:
    outputs: List[Path] = []
    if not rows:
        return outputs

    modes = {}
    for row in rows:
        key = row["velocity_model"]
        modes.setdefault(key, {})[row["lap_tau_mode"]] = row
    fig, ax = plt.subplots(figsize=(6, 4))
    for medium, mapping in modes.items():
        legacy = mapping.get("mixed_legacy")
        stretched = mapping.get("stretched_divergence")
        if legacy is None or stretched is None:
            continue
        x_values = [0, 1]
        y_values = [
            maybe_float(legacy["residual_rmse_evaluation"]),
            maybe_float(stretched["residual_rmse_evaluation"]),
        ]
        ax.plot(x_values, y_values, marker="o", label=medium)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["mixed_legacy", "stretched_divergence"])
    ax.set_ylabel("residual_rmse_evaluation")
    ax.set_title("B9 paired mode comparison")
    ax.legend()
    outputs.append(save_figure(fig, out_dir / "B9" / "B9__paired_scatter.png"))
    return outputs


def plot_batch_b10(rows: List[Dict[str, Any]], out_dir: Path) -> List[Path]:
    outputs: List[Path] = []
    if not rows:
        return outputs

    fig, ax = plt.subplots(figsize=(7, 4))
    omegas = sorted({maybe_float(row["omega"]) for row in rows if is_finite(row["omega"])})
    for omega in omegas:
        omega_rows = sorted([row for row in rows if maybe_float(row["omega"]) == omega], key=lambda row: int(row["grid_nx"]))
        if not omega_rows:
            continue
        y_values = [maybe_float(row["reference_residual_rmse"]) for row in omega_rows]
        if not any(value is not None for value in y_values):
            continue
        ax.plot([int(row["grid_nx"]) for row in omega_rows], y_values, marker="o", label=f"omega={omega:g}")
    ax.set_xlabel("grid_nx")
    ax.set_ylabel("reference_residual_rmse")
    ax.set_title("B10 reference floor vs grid")
    ax.legend(fontsize=8)
    outputs.append(save_figure(fig, out_dir / "B10" / "B10__reference_vs_grid.png"))

    fig, ax = plt.subplots(figsize=(6, 4))
    x = [int(row["grid_nx"]) ** 2 for row in rows if is_finite(row["runtime_sec"])]
    y = [maybe_float(row["runtime_sec"]) for row in rows if is_finite(row["runtime_sec"])]
    if x and y:
        ax.scatter(x, y, s=60)
        ax.set_xlabel("grid_nx^2")
        ax.set_ylabel("solve_runtime_sec")
        ax.set_title("B10 runtime scaling")
        outputs.append(save_figure(fig, out_dir / "B10" / "B10__runtime_scaling.png"))
    else:
        plt.close(fig)
    return outputs


def plot_batch_b11(rows: List[Dict[str, Any]], out_dir: Path) -> List[Path]:
    outputs: List[Path] = []
    if not rows:
        return outputs

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True)
    for medium in sorted({row["velocity_model"] for row in rows}):
        medium_rows = sorted([row for row in rows if row["velocity_model"] == medium], key=lambda row: maybe_float(row["source_mask_radius"]) or 0.0)
        radii = [maybe_float(row["source_mask_radius"]) for row in medium_rows]
        axes[0].plot(radii, [maybe_float(row["residual_rmse_evaluation"]) for row in medium_rows], marker="o", label=medium)
        axes[1].plot(radii, [maybe_float(row["residual_max_evaluation"]) for row in medium_rows], marker="o", label=medium)
        axes[2].plot(radii, [maybe_float(row["final_loss_pde"]) for row in medium_rows], marker="o", label=medium)
    axes[0].set_title("residual_rmse")
    axes[1].set_title("residual_max")
    axes[2].set_title("final_loss_pde")
    axes[0].legend()
    outputs.append(save_figure(fig, out_dir / "B11" / "B11__radius_sweep.png"))
    return outputs


def plot_batch_b12(rows: List[Dict[str, Any]], out_dir: Path) -> List[Path]:
    outputs: List[Path] = []
    if not rows:
        return outputs
    powers = sorted({int(row["pml_power"]) for row in rows if is_finite(row["pml_power"])})
    r0_values = sorted({safe_log10(row["pml_R0"]) for row in rows if is_finite(row["pml_R0"])})
    heat = np.full((len(powers), len(r0_values)), np.nan)
    for row in rows:
        p = int(row["pml_power"])
        r0 = safe_log10(row["pml_R0"])
        heat[powers.index(p), r0_values.index(r0)] = safe_log10(row["residual_max_evaluation"])
    fig, ax = plt.subplots(figsize=(6, 4))
    image = ax.imshow(heat, origin="lower", cmap="plasma")
    ax.set_xticks(range(len(r0_values)))
    ax.set_xticklabels([f"{value:.0f}" for value in r0_values])
    ax.set_yticks(range(len(powers)))
    ax.set_yticklabels(powers)
    ax.set_xlabel("log10(R0)")
    ax.set_ylabel("pml_power")
    ax.set_title("B12 log10 residual_max")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    outputs.append(save_figure(fig, out_dir / "B12" / "B12__pml_heatmap.png"))
    return outputs


def load_label_reference_line(bundle: Dict[str, Any], axis: str = "x") -> Optional[List[complex]]:
    reference_dir = bundle["manifest"].get("reference_output_dir")
    if not reference_dir:
        return None
    wavefield_path = Path(reference_dir) / "reference_wavefield.npy"
    if not wavefield_path.exists():
        return None

    cfg = bundle["cfg"]
    x_coords = load_csv_array(bundle["run_dir"] / "x_coords_physical.csv")
    y_coords = load_csv_array(bundle["run_dir"] / "y_coords_physical.csv")
    if x_coords is None or y_coords is None:
        return None

    wavefield = np.load(wavefield_path)
    physical = crop_to_physical(wavefield, cfg, len(x_coords), len(y_coords))
    source_pos = (cfg.get("physics", {}) or {}).get("source_pos", [0.5, 0.5])
    x_index = int(np.argmin(np.abs(np.asarray(x_coords) - float(source_pos[0]))))
    y_index = int(np.argmin(np.abs(np.asarray(y_coords) - float(source_pos[1]))))
    if axis == "x":
        return [physical[y_index, j] for j in range(physical.shape[1])]
    return [physical[i, x_index] for i in range(physical.shape[0])]


def plot_batch_b14(
    rows: List[Dict[str, Any]],
    centerlines: Dict[str, List[Dict[str, Any]]],
    run_bundles: List[Dict[str, Any]],
    out_dir: Path,
) -> List[Path]:
    outputs: List[Path] = []
    if not rows:
        return outputs
    bundles = bundle_map_by_run(run_bundles)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    x = np.arange(len(rows))
    axes[0].bar(x, [maybe_float(row["residual_rmse_evaluation"]) or 0.0 for row in rows], color="tab:blue")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([row["run_id"] for row in rows], rotation=45, ha="right")
    axes[0].set_title("B14 primary metric")
    axes[1].bar(x - 0.18, [maybe_float(row["final_loss_pde"]) or 0.0 for row in rows], width=0.36, label="loss_pde")
    axes[1].bar(x + 0.18, [maybe_float(row["final_loss_data"]) or 0.0 for row in rows], width=0.36, label="loss_data")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([row["run_id"] for row in rows], rotation=45, ha="right")
    axes[1].set_yscale("log")
    axes[1].set_title("B14 final loss decomposition")
    axes[1].legend()
    outputs.append(save_figure(fig, out_dir / "B14" / "B14__bars.png"))

    fig, axes = plt.subplots(len(rows), 1, figsize=(8, 3.2 * len(rows)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, row in zip(axes, sorted(rows, key=lambda item: item["run_id"])):
        line = centerlines.get(row["run_id"], [])
        x_vals = [maybe_float(item["s"]) for item in line]
        pred_complex = np.array(
            [
                complex(maybe_float(item["pred_re"]) or 0.0, maybe_float(item["pred_im"]) or 0.0)
                for item in line
            ],
            dtype=np.complex128,
        )
        ref_complex = np.array(
            [
                complex(maybe_float(item["ref_re"]) or 0.0, maybe_float(item["ref_im"]) or 0.0)
                for item in line
            ],
            dtype=np.complex128,
        )
        label_line = load_label_reference_line(bundles[row["run_id"]], axis="x")
        ax.plot(x_vals, np.unwrap(np.angle(pred_complex)), label="pred")
        if np.any(np.isfinite(np.real(ref_complex))) and np.any(np.isfinite(np.imag(ref_complex))):
            ax.plot(x_vals, np.unwrap(np.angle(ref_complex)), label="true_ref")
        if label_line is not None:
            ax.plot(x_vals, np.unwrap(np.angle(np.asarray(label_line))), label="label")
        ax.set_title(row["run_id"])
    axes[0].legend()
    outputs.append(save_figure(fig, out_dir / "B14" / "B14__phase_profiles.png"))
    return outputs


def render_batch_figures(assets: Dict[str, Any], out_dir: Path) -> List[Path]:
    report_rows = assets["report_rows"]
    run_bundles = assets["run_bundles"]
    pareto_rows = assets["pareto_rows"]
    centerline_rows = assets["centerline_rows"]
    batch_specs = assets["batch_specs"]

    outputs: List[Path] = []
    bundle_lookup = bundle_map_by_run(run_bundles)
    centerline_x_by_batch = {
        batch_id: centerline_map(centerline_rows, batch_id, "x")
        for batch_id in sorted({row["batch_id"] for row in report_rows})
    }

    for batch_id in sorted({row["batch_id"] for row in report_rows}):
        rows = sorted(report_rows_by_batch(report_rows, batch_id), key=lambda row: row["run_id"])
        spec = batch_specs.get(batch_id, {})

        if batch_id == "B0":
            outputs.extend(plot_batch_b0(rows, spec, out_dir))
        elif batch_id == "B1":
            outputs.extend(plot_batch_b1(rows, bundle_lookup, out_dir))
        elif batch_id == "B2":
            outputs.extend(plot_batch_b2(rows, centerline_x_by_batch.get(batch_id, {}), out_dir))
        elif batch_id == "B3":
            outputs.extend(plot_batch_b3(rows, out_dir))
        elif batch_id == "B4":
            outputs.extend(plot_batch_b4(rows, pareto_rows, out_dir))
        elif batch_id == "B5":
            outputs.extend(plot_batch_b5(rows, run_bundles, out_dir))
        elif batch_id == "B6":
            outputs.extend(plot_batch_b6(rows, out_dir))
        elif batch_id == "B7":
            outputs.extend(plot_batch_b7(rows, centerline_x_by_batch.get(batch_id, {}), out_dir))
        elif batch_id == "B9":
            outputs.extend(plot_batch_b9(rows, out_dir))
        elif batch_id == "B10":
            outputs.extend(plot_batch_b10(rows, out_dir))
        elif batch_id == "B11":
            outputs.extend(plot_batch_b11(rows, out_dir))
        elif batch_id == "B12":
            outputs.extend(plot_batch_b12(rows, out_dir))
        elif batch_id == "B14":
            outputs.extend(plot_batch_b14(rows, centerline_x_by_batch.get(batch_id, {}), run_bundles, out_dir))
        else:
            fallback = draw_basic_bar(batch_id, rows, out_dir, f"{batch_id} primary metric")
            if fallback is not None:
                outputs.append(fallback)

    return outputs


def write_markdown_report(
    output_root: Path,
    assets: Dict[str, Any],
    figure_paths: List[Path],
) -> Path:
    report_path = output_root / "_reports" / "matrix_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report_rows = assets["report_rows"]
    failure_rows = assets["failure_rows"]
    batch_specs = assets["batch_specs"]
    manifest_rows = assets["manifest_rows"]
    figure_root_path = report_path.parent

    blocked_specs = [
        spec for spec in batch_specs.values()
        if str(spec.get("status", "active")).lower() == "blocked"
    ]

    lines = [
        "# Matrix Report",
        "",
        f"- completed runs: {len(report_rows)}",
        f"- failed runs: {len(failure_rows)}",
        f"- blocked batches: {len(blocked_specs)}",
        "- loss component history: per-epoch `loss_pde` / `loss_data` remain an A7 blocker until runner persists multi-column loss history; L1 now reports final-state components and L2 marks the final epoch explicitly.",
        "",
    ]

    global_figures = [path for path in figure_paths if "/global/" in str(path)]
    if global_figures:
        lines.append("## Global Figures")
        lines.append("")
        for path in global_figures:
            lines.append(f"- `{path.relative_to(figure_root_path)}`")
        lines.append("")

    for batch_id in sorted({row["batch_id"] for row in report_rows} | set(batch_specs.keys())):
        spec = batch_specs.get(batch_id, {})
        lines.append(f"## {batch_id}")
        if spec.get("purpose"):
            lines.append(f"- purpose: {spec['purpose']}")
        if spec.get("status") == "blocked":
            lines.append(f"- status: blocked")
            lines.append(f"- blocker: {spec.get('blocked_reason', 'unspecified')}")
            lines.append("")
            continue
        lines.append("")

        batch_rows = sorted(
            report_rows_by_batch(report_rows, batch_id),
            key=lambda row: (
                math.inf if maybe_float(row["residual_rmse_evaluation"]) is None else maybe_float(row["residual_rmse_evaluation"]),
                row["run_id"],
            ),
        )
        if not batch_rows:
            lines.append("- no completed runs yet")
            lines.append("")
            continue

        lines.append("| run_id | primary | secondary | runtime_sec | supervision_mode |")
        lines.append("|---|---:|---:|---:|---|")
        for row in batch_rows[:12]:
            lines.append(
                "| {run_id} | {primary} | {secondary} | {runtime} | {mode} |".format(
                    run_id=row["run_id"],
                    primary=row["residual_rmse_evaluation"] if row["residual_rmse_evaluation"] != "" else "NA",
                    secondary=row["residual_p95_evaluation"] if row["residual_p95_evaluation"] != "" else "NA",
                    runtime=row["runtime_sec"] if row["runtime_sec"] != "" else "NA",
                    mode=row["supervision_mode"],
                )
            )
        lines.append("")

        batch_figures = [
            path.relative_to(figure_root_path)
            for path in figure_paths
            if f"/{batch_id}/" in str(path)
        ]
        if batch_figures:
            lines.append("Figures:")
            for path in batch_figures:
                lines.append(f"- `{path}`")
            lines.append("")

    if failure_rows:
        lines.append("## Failures")
        lines.append("")
        lines.append("| batch_id | run_id | stage | error | last_epoch | log |")
        lines.append("|---|---|---|---|---:|---|")
        for row in failure_rows:
            lines.append(
                "| {batch} | {run} | {stage} | {error} | {last_epoch} | `{log}` |".format(
                    batch=row.get("batch_id", ""),
                    run=row.get("run_id", ""),
                    stage=row.get("failure_stage", ""),
                    error=row.get("error_type", ""),
                    last_epoch=row.get("last_logged_epoch", ""),
                    log=row.get("traceback_path", ""),
                )
            )
        lines.append("")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return report_path


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    assets = build_report_assets(output_root)
    reports_root = output_root / "_reports"

    write_table(reports_root / "matrix_report.csv", REPORT_COLUMNS, assets["report_rows"])
    write_table(reports_root / "matrix_loss_history.csv", LOSS_HISTORY_COLUMNS, assets["loss_history_rows"])
    write_table(reports_root / "matrix_quantiles.csv", QUANTILE_COLUMNS, assets["quantile_rows"])
    write_table(reports_root / "matrix_centerline.csv", CENTERLINE_COLUMNS, assets["centerline_rows"])
    write_table(reports_root / "matrix_pareto.csv", PARETO_COLUMNS, assets["pareto_rows"])
    write_table(reports_root / "matrix_reference_only.csv", REFERENCE_ONLY_COLUMNS, assets["reference_only_rows"])
    write_table(reports_root / "matrix_failures.csv", FAILURE_COLUMNS, assets["failure_rows"])

    figures: List[Path] = []
    for maybe_path in [
        write_global_heatmap(assets["report_rows"], figure_root(output_root)),
        write_runtime_boxplot(assets["report_rows"], figure_root(output_root)),
        write_status_sankey(assets["report_rows"], assets["manifest_rows"], assets["failure_rows"], figure_root(output_root)),
        write_pca_projection(assets["report_rows"], figure_root(output_root)),
        write_spearman_heatmap(assets["report_rows"], figure_root(output_root)),
        write_parallel_coordinates(assets["report_rows"], figure_root(output_root)),
        write_gap_histogram(assets["report_rows"], figure_root(output_root)),
    ]:
        if maybe_path is not None:
            figures.append(maybe_path)

    figures.extend(render_batch_figures(assets, figure_root(output_root)))
    report_path = write_markdown_report(output_root, assets, figures)

    payload = {
        "report_csv": str(reports_root / "matrix_report.csv"),
        "report_md": str(report_path),
        "figure_count": len(figures),
        "run_count": len(assets["report_rows"]),
        "failure_count": len(assets["failure_rows"]),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
