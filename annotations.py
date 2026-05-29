"""
annotations.py
==============
From an "isolated-component" Q-scan pixel grid (e.g. the chirp by itself, or
the glitch by itself), produce:

  * a binary mask of the bright structure,
  * one or more polygons enclosing each connected component, and
  * a single YOLOv8 segmentation label line per polygon, normalised to [0, 1].

YOLOv8 seg label format (one per line):

    <class_id> x1 y1 x2 y2 ... xn yn

with all coords divided by image width/height. The image origin in YOLO is
top-left, so we flip y after computing the mask (which has origin bottom-left).
"""
from __future__ import annotations

import numpy as np
from typing import List, Tuple
from skimage import measure, morphology

from config import ImageCfg, QScanCfg, CLASS_ID
from skimage import morphology


def mask_from_energy(
    energy_grid: np.ndarray,
    threshold: float,
    min_pixels: int,
) -> np.ndarray:
    """Threshold and clean up a normalised energy grid into a binary mask."""
    if threshold < 0:
        percentile = -threshold
        thresh_val = np.percentile(energy_grid, percentile)
        m = energy_grid > thresh_val
    else:
        m = energy_grid > threshold

    if not m.any():
        return m

    labelled = measure.label(m, connectivity=2)
    keep = np.zeros_like(m, dtype=bool)
    for region in measure.regionprops(labelled):
        if region.area >= min_pixels:
            keep[labelled == region.label] = True

    # ========= 添加膨胀操作 =========
    radius = 20  # 膨胀半径（像素），可根据需要调整（1,2,3）
    selem = morphology.disk(radius)
    keep = morphology.binary_dilation(keep, selem)
    # ================================

    return keep


def mask_to_polygons(
    mask: np.ndarray,
    image_cfg: ImageCfg,
    min_pixels: int = 25,
) -> List[np.ndarray]:
    """Each connected component -> one simplified polygon (Nx2 in YOLO coords).

    Returned coordinates are normalised to [0, 1] with origin at top-left,
    matching YOLOv8 segmentation expectations.
    """
    H, W = mask.shape
    if H != image_cfg.height or W != image_cfg.width:
        raise ValueError("mask shape must match image grid")

    polygons: List[np.ndarray] = []
    labelled = measure.label(mask, connectivity=2)
    # Sort by area descending and take only the largest component
    regions = sorted(measure.regionprops(labelled), key=lambda r: r.area, reverse=True)
    for region in regions[:1]:  # Only keep the biggest component
        if region.area < min_pixels:
            continue
        # Outer contour at level 0.5
        contours = measure.find_contours(
            (labelled == region.label).astype(float), 0.5
        )
        if not contours:
            continue
        # Take the longest contour
        contour = max(contours, key=len)
        # Simplify so YOLO label files don't explode
        contour = _decimate_contour(contour, max_points=64)
        # contour is (row, col) = (y, x) with origin bottom-left of the array
        # YOLO wants (x, y) normalised with origin top-left.
        ys = contour[:, 0]
        xs = contour[:, 1]
        x_norm = xs / W
        y_norm = 1.0 - ys / H                  # flip vertically
        poly = np.stack([x_norm, y_norm], axis=1)
        poly = np.clip(poly, 0.0, 1.0)
        polygons.append(poly)
    return polygons


def _decimate_contour(contour: np.ndarray, max_points: int) -> np.ndarray:
    if len(contour) <= max_points:
        return contour
    step = len(contour) // max_points
    return contour[::step]


def polygons_to_yolo_lines(
    polygons: List[np.ndarray],
    class_name: str,
) -> List[str]:
    """Convert polygons to YOLOv8-seg label lines."""
    cid = CLASS_ID[class_name]
    lines = []
    for poly in polygons:
        flat = poly.reshape(-1)
        # Need at least 3 points for a valid polygon
        if len(flat) < 6:
            continue
        line = f"{cid} " + " ".join(f"{v:.6f}" for v in flat)
        lines.append(line)
    return lines


def build_label_lines(
    isolated_components: dict,
    image_cfg: ImageCfg,
    qscan_cfg: QScanCfg,
) -> List[str]:
    """Given a dict {class_name: pixel_grid}, produce all YOLO label lines.

    Example:
        isolated_components = {"chirp": chirp_grid, "noise": glitch_grid}

    Each pixel_grid has shape (H, W). The 'flip-y' for YOLO is done in
    mask_to_polygons; the pixel grid is expected with origin bottom-left
    (i.e. low frequencies at row 0), which is what energy_to_pixel_grid
    returns.
    """
    all_lines: List[str] = []
    thresholds = {
        "chirp": qscan_cfg.chirp_mask_threshold,
        "noise": qscan_cfg.noise_mask_threshold,
    }
    for class_name, grid in isolated_components.items():
        if grid is None:
            continue
        mask = mask_from_energy(grid, thresholds[class_name],
                                qscan_cfg.min_blob_pixels)
        if not mask.any():
            continue
        polys = mask_to_polygons(mask, image_cfg, qscan_cfg.min_blob_pixels)
        all_lines.extend(polygons_to_yolo_lines(polys, class_name))
    return all_lines


def save_yolo_label(lines: List[str], path: str) -> None:
    with open(path, "w") as f:
        f.write("\n".join(lines))
        if lines:
            f.write("\n")
