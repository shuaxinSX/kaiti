"""
Serial A6 batch launcher.

The launcher never mutates core training code. It materializes per-run configs
from batch YAML files, calls the existing training/reference entrypoints, and
writes launch metadata under outputs/matrix/_meta.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import resource
import socket
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml


META_MANIFEST_COLUMNS = [
    "batch_id",
    "batch_name",
    "batch_slug",
    "status",
    "entrypoint",
    "run_id",
    "output_dir",
    "effective_config_path",
    "reference_config_path",
    "reference_output_dir",
    "reference_cache_key",
    "git_sha_int",
    "launched_at",
    "machine_host",
    "blocked_reason",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch A6 matrix batches serially.")
    parser.add_argument(
        "batch_configs",
        nargs="*",
        help="Explicit batch YAML paths. Defaults to every B*.yaml under configs/experiments.",
    )
    parser.add_argument(
        "--all-batches",
        action="store_true",
        help="Launch every B*.yaml under configs/experiments.",
    )
    parser.add_argument(
        "--base-config",
        default="configs/base.yaml",
        help="Base config used to materialize each run.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/matrix",
        help="Ignored-by-git output root for matrix runs.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Execution device passed to downstream scripts.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Materialize configs and manifest without launching runs.",
    )
    parser.add_argument(
        "--include-blocked",
        action="store_true",
        help="Include blocked batches/runs in the manifest output.",
    )
    parser.add_argument(
        "--run-filter",
        default=None,
        help="Only launch runs whose run_id contains this substring.",
    )
    parser.add_argument(
        "--batch-filter",
        default=None,
        help="Only launch batches whose batch_id or filename contains this substring.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of launched runs after filtering.",
    )
    parser.add_argument(
        "--summarize",
        action="store_true",
        help="Run scripts/summarize_matrix.py after the launcher finishes.",
    )
    return parser.parse_args()


def deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    result = deepcopy(base)
    for key, value in overlay.items():
        if isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def dump_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)


def stable_config_hash(payload: Dict[str, Any]) -> str:
    dumped = yaml.safe_dump(payload, sort_keys=True, allow_unicode=True)
    return hashlib.sha256(dumped.encode("utf-8")).hexdigest()[:12]


def discover_batch_paths(repo_root: Path, args: argparse.Namespace) -> List[Path]:
    if args.batch_configs:
        return [Path(path).resolve() for path in args.batch_configs]

    experiments_dir = repo_root / "configs" / "experiments"
    discovered = sorted(experiments_dir.glob("B*.yaml"))
    if args.all_batches or not args.batch_configs:
        return discovered
    return discovered


def read_git_sha(repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def status_is_blocked(batch_spec: Dict[str, Any], run_spec: Dict[str, Any]) -> tuple[bool, str]:
    if str(batch_spec.get("status", "active")).lower() == "blocked":
        return True, str(batch_spec.get("blocked_reason", "batch is blocked"))
    if str(run_spec.get("status", "active")).lower() == "blocked":
        return True, str(run_spec.get("blocked_reason", "run is blocked"))
    return False, ""


def derive_batch_slug(batch_spec: Dict[str, Any], batch_path: Path) -> str:
    if batch_spec.get("batch_id") and batch_spec.get("name"):
        return f"{batch_spec['batch_id']}_{batch_spec['name']}"
    return batch_path.stem


def build_reference_plan(
    base_config: Dict[str, Any],
    batch_spec: Dict[str, Any],
    run_spec: Dict[str, Any],
    train_config: Dict[str, Any],
    meta_root: Path,
    output_root: Path,
) -> Optional[Dict[str, Any]]:
    reference_plan = run_spec.get("reference_prep")
    if not reference_plan or not reference_plan.get("enabled", False):
        return None

    batch_slug = derive_batch_slug(batch_spec, Path("unused"))
    label_name = str(reference_plan.get("label_name", "reference_label"))
    reference_overrides = reference_plan.get("config_overrides", {})
    # Intentionally start from the fully materialized train config so the label run
    # inherits the exact same grid / PML / model-independent physics context unless
    # the batch explicitly overrides it.
    reference_config = deep_merge(train_config, reference_overrides)
    training_cfg = reference_config.setdefault("training", {})
    supervision_cfg = training_cfg.setdefault("supervision", {})
    # Reference generation is a physics solve, not a supervised training run.
    # Hybrid settings from the originating train config would trip A3 fail-fast
    # checks because no reference_path exists yet for the label-generation job.
    training_cfg["lambda_data"] = 0.0
    supervision_cfg["enabled"] = False
    supervision_cfg["reference_path"] = None
    cache_key = stable_config_hash(reference_config)

    reference_cfg_path = meta_root / "generated_configs" / batch_slug / f"{run_spec['run_id']}__reference.yaml"
    reference_output_dir = output_root / "_reference_cache" / batch_slug / run_spec["run_id"] / f"{label_name}__{cache_key}"
    dump_yaml(reference_cfg_path, reference_config)

    return {
        "config": reference_config,
        "config_path": reference_cfg_path,
        "output_dir": reference_output_dir,
        "reference_path": reference_output_dir / "reference_envelope.npy",
        "cache_key": cache_key,
    }


def build_manifest_rows(
    repo_root: Path,
    batch_paths: Iterable[Path],
    base_config_path: Path,
    output_root: Path,
    meta_root: Path,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    base_config = load_yaml(base_config_path)
    git_sha = read_git_sha(repo_root)
    launched_at = datetime.now().isoformat(timespec="seconds")
    machine_host = socket.gethostname()

    rows: List[Dict[str, Any]] = []
    launched = 0

    for batch_path in batch_paths:
        batch_spec = load_yaml(batch_path)
        batch_id = str(batch_spec.get("batch_id", batch_path.stem))
        batch_name = str(batch_spec.get("name", batch_path.stem.replace(f"{batch_id}_", "")))
        batch_slug = derive_batch_slug(batch_spec, batch_path)

        if args.batch_filter and args.batch_filter not in batch_id and args.batch_filter not in batch_path.name:
            continue

        fixed = batch_spec.get("fixed", {})
        entrypoint = str(batch_spec.get("entrypoint", "train"))

        for run_spec in batch_spec.get("matrix", []):
            run_id = str(run_spec["run_id"])
            if args.run_filter and args.run_filter not in run_id:
                continue

            blocked, blocked_reason = status_is_blocked(batch_spec, run_spec)
            if blocked and not args.include_blocked:
                continue

            merged = deep_merge(base_config, fixed)
            merged = deep_merge(merged, run_spec.get("overrides", {}))

            reference_plan = build_reference_plan(
                base_config,
                batch_spec,
                run_spec,
                merged,
                meta_root,
                output_root,
            )
            if reference_plan is not None:
                merged.setdefault("training", {})
                merged["training"].setdefault("supervision", {})
                merged["training"]["supervision"]["enabled"] = True
                merged["training"]["supervision"]["reference_path"] = str(reference_plan["reference_path"])

            effective_cfg_path = meta_root / "generated_configs" / batch_slug / f"{run_id}.yaml"
            dump_yaml(effective_cfg_path, merged)

            row = {
                "batch_id": batch_id,
                "batch_name": batch_name,
                "batch_slug": batch_slug,
                "status": "blocked" if blocked else "active",
                "entrypoint": str(run_spec.get("entrypoint", entrypoint)),
                "run_id": run_id,
                "output_dir": str(output_root / batch_slug / run_id),
                "effective_config_path": str(effective_cfg_path),
                "reference_config_path": str(reference_plan["config_path"]) if reference_plan else "",
                "reference_output_dir": str(reference_plan["output_dir"]) if reference_plan else "",
                "reference_cache_key": str(reference_plan["cache_key"]) if reference_plan else "",
                "git_sha_int": git_sha,
                "launched_at": launched_at,
                "machine_host": machine_host,
                "blocked_reason": blocked_reason,
            }
            rows.append(row)
            launched += 0 if blocked else 1
            if args.limit is not None and launched >= args.limit:
                return rows

    return rows


def write_manifest(rows: List[Dict[str, Any]], meta_root: Path) -> Path:
    manifest_path = meta_root / "manifest.tsv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=META_MANIFEST_COLUMNS, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return manifest_path


def append_failure_row(meta_root: Path, row: Dict[str, Any]) -> None:
    failures_path = meta_root / "failures.csv"
    failures_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not failures_path.exists()
    with open(failures_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FAILURE_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def append_resume_row(meta_root: Path, row: Dict[str, Any]) -> None:
    resume_path = meta_root / "resume.tsv"
    resume_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not resume_path.exists()
    with open(resume_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["batch_id", "run_id", "output_dir", "reason"],
            delimiter="\t",
        )
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def read_tail(log_path: Path, n_lines: int = 10) -> str:
    if not log_path.exists():
        return ""
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    return "".join(lines[-n_lines:]).strip()


def extract_last_logged_epoch(text: str) -> str:
    matches = re.findall(r"Epoch\s+(\d+)(?:/\d+)?", text, flags=re.IGNORECASE)
    if matches:
        return matches[-1]
    matches = re.findall(r"epoch(?:=|\s+)(\d+)", text, flags=re.IGNORECASE)
    return matches[-1] if matches else ""


def parse_peak_mem_mb(stderr: str) -> Optional[float]:
    match = re.search(r"(\d+)\s+maximum resident set size", stderr)
    if match is None:
        return None
    rss_bytes = float(match.group(1))
    return rss_bytes / (1024.0 * 1024.0)


def write_runtime_metadata(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def poll_process_rss_mb(pid: int) -> Optional[float]:
    try:
        completed = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)],
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    value = completed.stdout.strip()
    if not value:
        return None
    try:
        return float(value) / 1024.0
    except ValueError:
        return None


def run_logged_command(cmd: List[str], cwd: Path, log_path: Path) -> Dict[str, Any]:
    before_rusage = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    start_time = time.perf_counter()
    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    peak_mem_mb = None
    while True:
        try:
            stdout, stderr = process.communicate(timeout=0.1)
            break
        except subprocess.TimeoutExpired:
            rss_mb = poll_process_rss_mb(process.pid)
            if rss_mb is not None:
                peak_mem_mb = rss_mb if peak_mem_mb is None else max(peak_mem_mb, rss_mb)

    runtime_sec = time.perf_counter() - start_time
    completed = subprocess.CompletedProcess(cmd, process.returncode, stdout, stderr)
    after_rusage = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    if peak_mem_mb is None and after_rusage:
        if sys.platform == "darwin":
            peak_mem_mb = float(after_rusage) / (1024.0 * 1024.0)
        else:
            peak_mem_mb = float(after_rusage) / 1024.0
    if peak_mem_mb is None and before_rusage and after_rusage >= before_rusage:
        delta = after_rusage - before_rusage
        if delta > 0:
            if sys.platform == "darwin":
                peak_mem_mb = float(delta) / (1024.0 * 1024.0)
            else:
                peak_mem_mb = float(delta) / 1024.0
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
    return {
        "completed": completed,
        "runtime_sec": runtime_sec,
        "peak_mem_mb": peak_mem_mb,
        "last_logged_epoch": extract_last_logged_epoch(f"{completed.stdout}\n{completed.stderr}"),
    }


def launch_reference_if_needed(repo_root: Path, row: Dict[str, Any], device: str, meta_root: Path) -> bool:
    reference_cfg_path = row["reference_config_path"]
    reference_output_dir = row["reference_output_dir"]
    if not reference_cfg_path:
        return True

    reference_path = Path(reference_output_dir) / "reference_envelope.npy"
    if reference_path.exists():
        return True

    reference_log = meta_root / "logs" / row["batch_id"] / f"{row['run_id']}__reference.log"
    cmd = [
        sys.executable,
        "scripts/solve_reference.py",
        "--config",
        reference_cfg_path,
        "--device",
        device,
        "--output-dir",
        reference_output_dir,
    ]
    result = run_logged_command(cmd, repo_root, reference_log)
    completed = result["completed"]
    if completed.returncode == 0:
        write_runtime_metadata(
            Path(reference_output_dir) / "_matrix_runtime.json",
            {
                "reference_runtime_sec": result["runtime_sec"],
                "reference_peak_mem_mb": result["peak_mem_mb"],
                "reference_cache_key": row.get("reference_cache_key", ""),
            },
        )
        return True

    append_failure_row(
        meta_root,
        {
            "batch_id": row["batch_id"],
            "run_id": row["run_id"],
            "failure_stage": "reference",
            "error_type": f"returncode:{completed.returncode}",
            "error_msg_head": read_tail(reference_log, n_lines=8)[:300],
            "last_logged_epoch": result["last_logged_epoch"],
            "traceback_path": str(reference_log),
            "recoverable": True,
            "retried": False,
        },
    )
    return False


def solve_reference_for_run(repo_root: Path, row: Dict[str, Any], device: str, meta_root: Path) -> Optional[Dict[str, Any]]:
    """Materialize evaluation reference artifacts in the run directory itself.

    Training runs otherwise rely on evaluate_run.py to synthesize these
    artifacts implicitly, which makes ref_solve_runtime_sec impossible to
    measure separately.
    """
    reference_path = Path(row["output_dir"]) / "reference_envelope.npy"
    if reference_path.exists():
        return None

    reference_log = meta_root / "logs" / row["batch_id"] / f"{row['run_id']}__reference_eval.log"
    cmd = [
        sys.executable,
        "scripts/solve_reference.py",
        "--run-dir",
        row["output_dir"],
        "--device",
        device,
    ]
    result = run_logged_command(cmd, repo_root, reference_log)
    completed = result["completed"]
    if completed.returncode == 0:
        return result

    append_failure_row(
        meta_root,
        {
            "batch_id": row["batch_id"],
            "run_id": row["run_id"],
            "failure_stage": "reference",
            "error_type": f"returncode:{completed.returncode}",
            "error_msg_head": read_tail(reference_log, n_lines=8)[:300],
            "last_logged_epoch": result["last_logged_epoch"],
            "traceback_path": str(reference_log),
            "recoverable": True,
            "retried": False,
        },
    )
    return result


def launch_run(repo_root: Path, row: Dict[str, Any], device: str, meta_root: Path) -> bool:
    output_dir = Path(row["output_dir"])
    if output_dir.exists():
        append_resume_row(
            meta_root,
            {
                "batch_id": row["batch_id"],
                "run_id": row["run_id"],
                "output_dir": row["output_dir"],
                "reason": "output directory already exists",
            },
        )
        return True

    if not launch_reference_if_needed(repo_root, row, device, meta_root):
        return False

    log_root = meta_root / "logs" / row["batch_id"]
    entrypoint = row["entrypoint"]
    if entrypoint == "reference_only":
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
    else:
        cmd = [
            sys.executable,
            "scripts/run_train.py",
            "--config",
            row["effective_config_path"],
            "--device",
            device,
            "--output-dir",
            row["output_dir"],
        ]

    launch_log = log_root / f"{row['run_id']}__launch.log"
    launch_result = run_logged_command(cmd, repo_root, launch_log)
    completed = launch_result["completed"]
    if completed.returncode != 0:
        append_failure_row(
            meta_root,
            {
                "batch_id": row["batch_id"],
                "run_id": row["run_id"],
                "failure_stage": entrypoint,
                "error_type": f"returncode:{completed.returncode}",
                "error_msg_head": read_tail(launch_log, n_lines=8)[:300],
                "last_logged_epoch": launch_result["last_logged_epoch"],
                "traceback_path": str(launch_log),
                "recoverable": True,
                "retried": False,
            },
        )
        return False

    runtime_metadata = {
        "train_runtime_sec": launch_result["runtime_sec"] if entrypoint == "train" else None,
        "train_peak_mem_mb": launch_result["peak_mem_mb"] if entrypoint == "train" else None,
        "reference_only_runtime_sec": launch_result["runtime_sec"] if entrypoint == "reference_only" else None,
        "reference_only_peak_mem_mb": launch_result["peak_mem_mb"] if entrypoint == "reference_only" else None,
        "ref_solve_runtime_sec": None,
        "ref_solve_peak_mem_mb": None,
        "eval_runtime_sec": None,
        "eval_peak_mem_mb": None,
    }

    if entrypoint == "reference_only":
        effective_cfg = load_yaml(Path(row["effective_config_path"]))
        dump_yaml(Path(row["output_dir"]) / "config_merged.yaml", effective_cfg)
        reference_metrics_path = Path(row["output_dir"]) / "reference_metrics.json"
        if reference_metrics_path.exists():
            with open(reference_metrics_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            summary["output_dir"] = row["output_dir"]
            summary["device"] = device
            write_runtime_metadata(Path(row["output_dir"]) / "summary.json", summary)

    if entrypoint == "train":
        reference_result = solve_reference_for_run(repo_root, row, device, meta_root)
        if reference_result is not None:
            if reference_result["completed"].returncode != 0:
                return False
            runtime_metadata["ref_solve_runtime_sec"] = reference_result["runtime_sec"]
            runtime_metadata["ref_solve_peak_mem_mb"] = reference_result["peak_mem_mb"]

        eval_log = log_root / f"{row['run_id']}__evaluate.log"
        eval_cmd = [
            sys.executable,
            "scripts/evaluate_run.py",
            "--run-dir",
            row["output_dir"],
            "--device",
            device,
        ]
        eval_result = run_logged_command(eval_cmd, repo_root, eval_log)
        eval_completed = eval_result["completed"]
        if eval_completed.returncode != 0:
            append_failure_row(
                meta_root,
                {
                    "batch_id": row["batch_id"],
                    "run_id": row["run_id"],
                    "failure_stage": "eval",
                    "error_type": f"returncode:{eval_completed.returncode}",
                    "error_msg_head": read_tail(eval_log, n_lines=8)[:300],
                    "last_logged_epoch": eval_result["last_logged_epoch"],
                    "traceback_path": str(eval_log),
                    "recoverable": True,
                    "retried": False,
                },
            )
            return False
        runtime_metadata["eval_runtime_sec"] = eval_result["runtime_sec"]
        runtime_metadata["eval_peak_mem_mb"] = eval_result["peak_mem_mb"]

    if row["reference_output_dir"]:
        reference_runtime_path = Path(row["reference_output_dir"]) / "_matrix_runtime.json"
        if reference_runtime_path.exists():
            with open(reference_runtime_path, "r", encoding="utf-8") as f:
                reference_runtime = json.load(f)
            runtime_metadata["ref_solve_runtime_sec"] = reference_runtime.get("reference_runtime_sec")
            runtime_metadata["ref_solve_peak_mem_mb"] = reference_runtime.get("reference_peak_mem_mb")

    write_runtime_metadata(Path(row["output_dir"]) / "_matrix_runtime.json", runtime_metadata)

    return True


def maybe_summarize(repo_root: Path, output_root: Path) -> None:
    cmd = [
        sys.executable,
        "scripts/summarize_matrix.py",
        "--output-root",
        str(output_root),
    ]
    subprocess.run(cmd, cwd=repo_root, check=True)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    base_config_path = (repo_root / args.base_config).resolve()
    output_root = (repo_root / args.output_root).resolve()
    meta_root = output_root / "_meta"

    batch_paths = discover_batch_paths(repo_root, args)
    rows = build_manifest_rows(
        repo_root=repo_root,
        batch_paths=batch_paths,
        base_config_path=base_config_path,
        output_root=output_root,
        meta_root=meta_root,
        args=args,
    )
    manifest_path = write_manifest(rows, meta_root)

    launched = 0
    skipped = 0
    blocked = 0

    for row in rows:
        if row["status"] == "blocked":
            blocked += 1
            continue
        if args.dry_run:
            skipped += 1
            continue

        ok = launch_run(repo_root, row, args.device, meta_root)
        launched += 1 if ok else 0
        skipped += 0 if ok else 1

    if args.summarize:
        maybe_summarize(repo_root, output_root)

    payload = {
        "manifest": str(manifest_path),
        "output_root": str(output_root),
        "rows": len(rows),
        "launched": launched,
        "blocked": blocked,
        "skipped_or_failed": skipped,
        "dry_run": bool(args.dry_run),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
