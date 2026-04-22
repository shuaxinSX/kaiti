"""
Launch the large A6 matrix from local batch specs.

This launcher stays on the integrated branch and reads committed matrix YAML
definitions from `configs/experiments/B*.yaml`. It does not rely on the
branch-specific Unix launcher from the experimental worktree.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import _bootstrap  # noqa: F401
import yaml


MANIFEST_COLUMNS = [
    "batch_id",
    "batch_name",
    "batch_file",
    "status",
    "entrypoint",
    "run_id",
    "output_dir",
    "effective_config_path",
    "reference_config_path",
    "reference_output_dir",
    "reference_cache_key",
    "git_sha",
    "spec_branch",
    "launched_at",
    "blocked_reason",
    "epochs_override",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full matrix from local batch specs.")
    parser.add_argument(
        "--base-config",
        default="configs/base.yaml",
        help="Base config from the current workspace.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/matrix",
        help="Output root for matrix runs.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Execution device passed to downstream scripts.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only materialize configs and manifest; do not launch runs.",
    )
    parser.add_argument(
        "--include-blocked",
        action="store_true",
        help="Include blocked batches/runs in the manifest output.",
    )
    parser.add_argument(
        "--batch-filter",
        default=None,
        help="Keep only batches whose id or filename contains this substring.",
    )
    parser.add_argument(
        "--run-filter",
        default=None,
        help="Keep only runs whose run_id contains this substring.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on launched active runs after filtering.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record failures and continue with later runs.",
    )
    parser.add_argument(
        "--epochs-override",
        type=int,
        default=None,
        help="Optional global training.epochs override applied after all batch/run overlays.",
    )
    return parser.parse_args()


def git_cmd(repo_root: Path, *args: str) -> List[str]:
    safe_repo = repo_root.resolve().as_posix()
    return ["git", "-c", f"safe.directory={safe_repo}", "-C", str(repo_root), *args]


def git_output(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        git_cmd(repo_root, *args),
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return completed.stdout


def deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    result = deepcopy(base)
    for key, value in overlay.items():
        if isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_yaml_file(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def dump_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)


def stable_config_hash(payload: Dict[str, Any]) -> str:
    dumped = yaml.safe_dump(payload, sort_keys=True, allow_unicode=True)
    import hashlib

    return hashlib.sha256(dumped.encode("utf-8")).hexdigest()[:12]


def discover_batch_specs(repo_root: Path) -> List[Dict[str, Any]]:
    paths = sorted((repo_root / "configs" / "experiments").glob("B*.yaml"))
    specs: List[Dict[str, Any]] = []
    for path in paths:
        spec = load_yaml_file(path)
        spec["__file__"] = str(path.relative_to(repo_root).as_posix())
        specs.append(spec)
    return specs


def is_blocked(batch_spec: Dict[str, Any], run_spec: Dict[str, Any]) -> tuple[bool, str]:
    batch_status = str(batch_spec.get("status", "active")).lower()
    if batch_status == "blocked":
        return True, str(batch_spec.get("blocked_reason", "batch is blocked"))

    run_status = str(run_spec.get("status", "active")).lower()
    if run_status == "blocked":
        return True, str(run_spec.get("blocked_reason", "run is blocked"))

    return False, ""


def build_reference_plan(
    batch_spec: Dict[str, Any],
    run_spec: Dict[str, Any],
    train_config: Dict[str, Any],
    meta_root: Path,
    output_root: Path,
) -> Optional[Dict[str, Any]]:
    reference_plan = run_spec.get("reference_prep")
    if not reference_plan or not reference_plan.get("enabled", False):
        return None

    batch_slug = f"{batch_spec['batch_id']}_{batch_spec.get('name', 'batch')}"
    label_name = str(reference_plan.get("label_name", "reference_label"))
    reference_overrides = reference_plan.get("config_overrides", {})
    reference_config = deep_merge(train_config, reference_overrides)

    training_cfg = reference_config.setdefault("training", {})
    supervision_cfg = training_cfg.setdefault("supervision", {})
    training_cfg["lambda_data"] = 0.0
    supervision_cfg["enabled"] = False
    supervision_cfg["reference_path"] = None

    cache_key = stable_config_hash(reference_config)
    cfg_path = meta_root / "generated_configs" / batch_slug / f"{run_spec['run_id']}__reference.yaml"
    out_dir = output_root / "_reference_cache" / batch_slug / run_spec["run_id"] / f"{label_name}__{cache_key}"
    dump_yaml(cfg_path, reference_config)
    return {
        "config_path": cfg_path,
        "output_dir": out_dir,
        "reference_path": out_dir / "reference_envelope.npy",
        "cache_key": cache_key,
    }


def build_manifest_rows(
    repo_root: Path,
    args: argparse.Namespace,
    output_root: Path,
    meta_root: Path,
) -> List[Dict[str, Any]]:
    base_cfg = load_yaml_file(repo_root / args.base_config)
    batch_specs = discover_batch_specs(repo_root)
    git_sha = git_output(repo_root, "rev-parse", "HEAD").strip()
    launched_at = datetime.now().isoformat(timespec="seconds")

    rows: List[Dict[str, Any]] = []
    active_count = 0
    for batch_spec in batch_specs:
        batch_id = str(batch_spec.get("batch_id", Path(batch_spec["__file__"]).stem))
        batch_name = str(batch_spec.get("name", batch_id))
        if args.batch_filter and args.batch_filter not in batch_id and args.batch_filter not in batch_spec["__file__"]:
            continue

        fixed = batch_spec.get("fixed", {})
        entrypoint = str(batch_spec.get("entrypoint", "train"))
        batch_slug = f"{batch_id}_{batch_name}"

        for run_spec in batch_spec.get("matrix", []):
            run_id = str(run_spec["run_id"])
            if args.run_filter and args.run_filter not in run_id:
                continue

            blocked, blocked_reason = is_blocked(batch_spec, run_spec)
            if blocked and not args.include_blocked:
                continue

            merged = deep_merge(base_cfg, fixed)
            merged = deep_merge(merged, run_spec.get("overrides", {}))
            if args.epochs_override is not None:
                merged.setdefault("training", {})
                merged["training"]["epochs"] = int(args.epochs_override)

            reference = build_reference_plan(batch_spec, run_spec, merged, meta_root, output_root)
            if reference is not None:
                merged.setdefault("training", {})
                merged["training"].setdefault("supervision", {})
                merged["training"]["supervision"]["enabled"] = True
                merged["training"]["supervision"]["reference_path"] = str(reference["reference_path"])

            cfg_path = meta_root / "generated_configs" / batch_slug / f"{run_id}.yaml"
            dump_yaml(cfg_path, merged)

            rows.append(
                {
                    "batch_id": batch_id,
                    "batch_name": batch_name,
                    "batch_file": batch_spec["__file__"],
                    "status": "blocked" if blocked else "active",
                    "entrypoint": str(run_spec.get("entrypoint", entrypoint)),
                    "run_id": run_id,
                    "output_dir": str(output_root / batch_slug / run_id),
                    "effective_config_path": str(cfg_path),
                    "reference_config_path": str(reference["config_path"]) if reference else "",
                    "reference_output_dir": str(reference["output_dir"]) if reference else "",
                    "reference_cache_key": str(reference["cache_key"]) if reference else "",
                    "git_sha": git_sha,
                    "spec_branch": "local-configs",
                    "launched_at": launched_at,
                    "blocked_reason": blocked_reason,
                    "epochs_override": str(args.epochs_override) if args.epochs_override is not None else "",
                }
            )

            if not blocked:
                active_count += 1
                if args.limit is not None and active_count >= args.limit:
                    return rows

    return rows


def write_manifest(meta_root: Path, rows: List[Dict[str, Any]]) -> Path:
    manifest_path = meta_root / "manifest.tsv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


def run_logged_command(cmd: List[str], cwd: Path, log_path: Path) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("$ " + " ".join(cmd) + "\n")
        if completed.stdout:
            f.write(completed.stdout)
            if not completed.stdout.endswith("\n"):
                f.write("\n")
        if completed.stderr:
            f.write(completed.stderr)
            if not completed.stderr.endswith("\n"):
                f.write("\n")
    return completed


def launch_reference_if_needed(repo_root: Path, row: Dict[str, Any], device: str, meta_root: Path) -> None:
    if not row["reference_config_path"]:
        return

    reference_output_dir = Path(row["reference_output_dir"])
    reference_path = reference_output_dir / "reference_envelope.npy"
    if reference_path.exists():
        return

    cmd = [
        sys.executable,
        "scripts/solve_reference.py",
        "--config",
        row["reference_config_path"],
        "--device",
        device,
        "--output-dir",
        row["reference_output_dir"],
    ]
    completed = run_logged_command(
        cmd,
        repo_root,
        meta_root / "logs" / row["batch_id"] / f"{row['run_id']}__reference.log",
    )
    if completed.returncode != 0:
        raise RuntimeError(f"reference prep failed for {row['run_id']}")


def ensure_reference_only_config_snapshot(row: Dict[str, Any]) -> None:
    output_dir = Path(row["output_dir"])
    cfg_path = output_dir / "config_merged.yaml"
    if cfg_path.exists():
        return
    dump_yaml(cfg_path, load_yaml_file(Path(row["effective_config_path"])))


def backfill_reference_only_summary(row: Dict[str, Any], device: str) -> None:
    output_dir = Path(row["output_dir"])
    ensure_reference_only_config_snapshot(row)
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        return

    payload: Dict[str, Any] = {}
    reference_metrics_path = output_dir / "reference_metrics.json"
    if reference_metrics_path.exists():
        with open(reference_metrics_path, "r", encoding="utf-8") as f:
            payload.update(json.load(f))

    cfg = load_yaml_file(Path(row["effective_config_path"]))
    training_cfg = cfg.get("training", {})
    medium_cfg = cfg.get("medium", {})
    payload.update(
        {
            "output_dir": str(output_dir),
            "device": device,
            "entrypoint": "reference_only",
            "epochs": int(training_cfg["epochs"]) if training_cfg.get("epochs") is not None else None,
            "velocity_model": medium_cfg.get("velocity_model"),
        }
    )

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def launch_run(repo_root: Path, row: Dict[str, Any], device: str, meta_root: Path) -> None:
    output_dir = Path(row["output_dir"])
    if output_dir.exists():
        if row["entrypoint"] == "reference_only":
            backfill_reference_only_summary(row, device)
        return

    launch_reference_if_needed(repo_root, row, device, meta_root)

    if row["entrypoint"] == "reference_only":
        cmd = [
            sys.executable,
            "scripts/solve_reference.py",
            "--config",
            row["effective_config_path"],
            "--device",
            device,
            "--output-dir",
            row["output_dir"],
        ]
        completed = run_logged_command(
            cmd,
            repo_root,
            meta_root / "logs" / row["batch_id"] / f"{row['run_id']}__launch.log",
        )
        if completed.returncode != 0:
            raise RuntimeError(f"reference-only run failed for {row['run_id']}")
        backfill_reference_only_summary(row, device)
        return

    train_cmd = [
        sys.executable,
        "scripts/run_train.py",
        "--config",
        row["effective_config_path"],
        "--device",
        device,
        "--output-dir",
        row["output_dir"],
    ]
    completed = run_logged_command(
        train_cmd,
        repo_root,
        meta_root / "logs" / row["batch_id"] / f"{row['run_id']}__train.log",
    )
    if completed.returncode != 0:
        raise RuntimeError(f"train failed for {row['run_id']}")

    eval_cmd = [
        sys.executable,
        "scripts/evaluate_run.py",
        "--run-dir",
        row["output_dir"],
        "--device",
        device,
    ]
    completed = run_logged_command(
        eval_cmd,
        repo_root,
        meta_root / "logs" / row["batch_id"] / f"{row['run_id']}__eval.log",
    )
    if completed.returncode != 0:
        raise RuntimeError(f"evaluate_run failed for {row['run_id']}")


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    output_root = (repo_root / args.output_root).resolve()
    meta_root = output_root / "_meta"
    rows = build_manifest_rows(repo_root, args, output_root, meta_root)
    manifest_path = write_manifest(meta_root, rows)

    launched = 0
    blocked = 0
    failed = 0
    start_time = time.perf_counter()

    for index, row in enumerate(rows, start=1):
        if row["status"] == "blocked":
            blocked += 1
            continue
        if args.dry_run:
            continue

        print(f"[{index}/{len(rows)}] {row['batch_id']} / {row['run_id']}", flush=True)
        try:
            launch_run(repo_root, row, args.device, meta_root)
            launched += 1
        except Exception as exc:
            failed += 1
            failure_path = meta_root / "failures.jsonl"
            failure_path.parent.mkdir(parents=True, exist_ok=True)
            with open(failure_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"run_id": row["run_id"], "batch_id": row["batch_id"], "error": str(exc)}, ensure_ascii=False) + "\n")
            if not args.continue_on_error:
                raise

    payload = {
        "manifest": str(manifest_path),
        "output_root": str(output_root),
        "rows": len(rows),
        "active_rows": sum(1 for row in rows if row["status"] != "blocked"),
        "blocked_rows": blocked,
        "launched": launched,
        "failed": failed,
        "dry_run": bool(args.dry_run),
        "epochs_override": args.epochs_override,
        "runtime_sec": time.perf_counter() - start_time,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
