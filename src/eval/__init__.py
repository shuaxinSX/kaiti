"""
eval 子包 — 评估与参考解
=========================

- reference_solver.py : 基于离散残差算子的稀疏参考解
- reference_eval.py   : reference 指标、产物导出与 run-dir/config 工作流
"""

from src.eval.reference_eval import (
    REFERENCE_SUMMARY_KEYS,
    compute_prediction_envelope,
    export_reference_artifacts,
    solve_reference_from_config,
    solve_reference_from_run_dir,
)
from src.eval.reference_solver import assemble_reference_operator, solve_reference_scattering

__all__ = [
    "REFERENCE_SUMMARY_KEYS",
    "assemble_reference_operator",
    "compute_prediction_envelope",
    "export_reference_artifacts",
    "solve_reference_from_config",
    "solve_reference_from_run_dir",
    "solve_reference_scattering",
]
