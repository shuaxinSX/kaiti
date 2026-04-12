"""
core 子包 — 基础数据结构
========================

- grid.py          : Grid2D — 规则网格、坐标系、距离场
- medium.py        : Medium2D — 介质慢度场、PML 内退化
- source.py        : PointSource — 源坐标、索引、邻域
- complex_utils.py : 实虚拆分、合并、复数乘法辅助
"""

from .grid import Grid2D
from .medium import Medium2D
from .source import PointSource
from .complex_utils import complex_to_dual, dual_to_complex, complex_mul_dual, to_network_input
