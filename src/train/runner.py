"""
训练运行器
==========

提供可复现的训练入口，并将结果保存到项目输出目录。
"""

import argparse
import csv
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import torch
import yaml

from src.config import load_config
from src.train.trainer import Trainer

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def configure_logging(level="INFO"):
    """Configure root logging once for CLI execution."""
    resolved_level = getattr(logging, str(level).upper(), logging.INFO)
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=resolved_level,
            format="%(levelname)s:%(name)s:%(message)s",
        )
    root_logger.setLevel(resolved_level)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def resolve_output_dir(save_dir, output_dir=None):
    """Resolve the final output directory under the current project folder."""
    if output_dir is not None:
        final_dir = Path(output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_dir = Path(save_dir) / f"run_{timestamp}"

    if not final_dir.is_absolute():
        final_dir = Path.cwd() / final_dir

    final_dir.mkdir(parents=True, exist_ok=False)
    return final_dir


def save_losses(losses, output_dir):
    """Persist loss history as both numeric data and a plot."""
    losses_array = np.asarray(losses, dtype=np.float64)
    np.save(output_dir / "losses.npy", losses_array)

    with open(output_dir / "losses.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss"])
        for epoch, loss in enumerate(losses, start=1):
            writer.writerow([epoch, f"{loss:.16e}"])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(1, len(losses) + 1), losses_array, linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "loss_curve.png", dpi=160)
    plt.close(fig)


def save_wavefield(u_total, output_dir):
    """Save the reconstructed wavefield and a quick-look visualization."""
    np.save(output_dir / "wavefield.npy", u_total)

    magnitude = np.abs(u_total)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    plot_specs = (
        ("Real", u_total.real),
        ("Imag", u_total.imag),
        ("Magnitude", magnitude),
    )
    for ax, (title, values) in zip(axes, plot_specs):
        image = ax.imshow(values, origin="lower", cmap="viridis")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(output_dir / "wavefield.png", dpi=160)
    plt.close(fig)


def save_model(model, output_dir):
    """Save a CPU copy of the trained model weights."""
    state_dict = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save(state_dict, output_dir / "model_state.pt")


def save_config_snapshot(cfg, output_dir):
    """Save the merged runtime configuration."""
    with open(output_dir / "config_merged.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False, allow_unicode=True)


def save_summary(summary, output_dir):
    """Write the run summary as JSON."""
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def run_training(
    base_config,
    overlay_config=None,
    device="auto",
    epochs=None,
    output_dir=None,
    velocity_model=None,
):
    """Run one end-to-end training job and persist artifacts."""
    cfg = load_config(base_config, overlay_config)
    if velocity_model is not None:
        cfg.medium.velocity_model = velocity_model
    if epochs is not None:
        cfg.training.epochs = int(epochs)
    configure_logging(cfg.logging.level)
    final_output_dir = resolve_output_dir(cfg.logging.save_dir, output_dir)
    save_config_snapshot(cfg, final_output_dir)

    trainer = Trainer(cfg, device=device)
    actual_epochs = cfg.training.epochs

    logger.info("运行输出目录: %s", final_output_dir)
    start_time = time.perf_counter()
    losses = trainer.train(epochs=actual_epochs)
    runtime_sec = time.perf_counter() - start_time
    u_total = trainer.reconstruct_wavefield()

    save_losses(losses, final_output_dir)
    save_wavefield(u_total, final_output_dir)
    save_model(trainer.model, final_output_dir)

    summary = {
        "output_dir": str(final_output_dir),
        "device": str(trainer.device),
        "epochs": int(actual_epochs),
        "final_loss": float(losses[-1]),
        "runtime_sec": runtime_sec,
        "velocity_model": cfg.medium.velocity_model,
        "grid_shape": [int(trainer.grid.ny_total), int(trainer.grid.nx_total)],
    }
    save_summary(summary, final_output_dir)

    return summary


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Run one training job and save artifacts.")
    parser.add_argument(
        "--config",
        default="configs/base.yaml",
        help="Base config path.",
    )
    parser.add_argument(
        "--overlay",
        default=None,
        help="Optional overlay config path, e.g. configs/debug.yaml.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Execution device: auto, cpu, cuda, cuda:0, ...",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional epoch override.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Defaults to logging.save_dir/run_TIMESTAMP.",
    )
    parser.add_argument(
        "--velocity-model",
        default=None,
        help="Optional medium override: homogeneous, smooth_lens, layered.",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()
    summary = run_training(
        base_config=args.config,
        overlay_config=args.overlay,
        device=args.device,
        epochs=args.epochs,
        output_dir=args.output_dir,
        velocity_model=args.velocity_model,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
