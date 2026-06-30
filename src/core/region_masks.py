"""
Grid-aligned region masks shared by training and evaluation utilities.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def _normalize_band(band_h: int, upper_bound: int) -> int:
    return max(1, min(int(band_h), max(1, int(upper_bound))))


def build_interface_band_mask(grid, band_h: int) -> np.ndarray:
    """Return a ring mask around the physical/PML interface on both sides."""
    band = _normalize_band(band_h, grid.pml_width + max(grid.nx, grid.ny))
    mask = np.zeros((grid.ny_total, grid.nx_total), dtype=bool)

    top = max(0, grid.pml_width - band)
    bottom = min(grid.ny_total, grid.pml_width + grid.ny + band)
    left = max(0, grid.pml_width - band)
    right = min(grid.nx_total, grid.pml_width + grid.nx + band)
    mask[top:bottom, left:right] = True

    inner_top = grid.pml_width + band
    inner_bottom = grid.pml_width + grid.ny - band
    inner_left = grid.pml_width + band
    inner_right = grid.pml_width + grid.nx - band
    if inner_top < inner_bottom and inner_left < inner_right:
        mask[inner_top:inner_bottom, inner_left:inner_right] = False

    return mask


def build_outer_boundary_band_mask(grid, band_h: int) -> np.ndarray:
    """Return a band mask adjacent to the outer computational boundary."""
    band = _normalize_band(band_h, min(grid.nx_total, grid.ny_total))
    mask = np.zeros((grid.ny_total, grid.nx_total), dtype=bool)
    mask[:band, :] = True
    mask[-band:, :] = True
    mask[:, :band] = True
    mask[:, -band:] = True
    return mask


def build_region_masks(
    grid,
    loss_mask: Optional[np.ndarray] = None,
    interface_band_h: int = 4,
    pml_band_h: int = 4,
    outer_boundary_h: int = 1,
) -> Dict[str, np.ndarray]:
    """Build reusable region masks for diagnostics and weighted PDE losses."""
    pml_mask = grid.pml_mask()
    physical_mask = ~pml_mask
    active_mask = np.ones_like(pml_mask, dtype=bool) if loss_mask is None else np.asarray(loss_mask, dtype=bool)

    interface_1h = build_interface_band_mask(grid, 1)
    interface_2h = build_interface_band_mask(grid, 2)
    interface_4h = build_interface_band_mask(grid, 4)
    interface_custom = build_interface_band_mask(grid, interface_band_h)
    outer_boundary_band = build_outer_boundary_band_mask(grid, outer_boundary_h)
    pml_inner_band = pml_mask & build_interface_band_mask(grid, pml_band_h)
    pml_outer_band = pml_mask & build_outer_boundary_band_mask(grid, pml_band_h)

    return {
        "active_mask": active_mask,
        "pml_mask": pml_mask,
        "physical_mask": physical_mask,
        "pml_active_mask": pml_mask & active_mask,
        "physical_active_mask": physical_mask & active_mask,
        "evaluation_mask": physical_mask & active_mask,
        "interface_1h": interface_1h,
        "interface_2h": interface_2h,
        "interface_4h": interface_4h,
        "interface_custom": interface_custom,
        "interface_1h_active": interface_1h & active_mask,
        "interface_2h_active": interface_2h & active_mask,
        "interface_4h_active": interface_4h & active_mask,
        "interface_custom_active": interface_custom & active_mask,
        "physical_bulk_active": physical_mask & active_mask & ~interface_custom,
        "pml_bulk_active": pml_mask & active_mask & ~interface_custom,
        "physical_interface_active": physical_mask & active_mask & interface_custom,
        "pml_interface_active": pml_mask & active_mask & interface_custom,
        "pml_inner_band": pml_inner_band,
        "pml_outer_band": pml_outer_band,
        "outer_boundary_band": outer_boundary_band,
        "outer_boundary_pml_band": pml_mask & outer_boundary_band,
    }
