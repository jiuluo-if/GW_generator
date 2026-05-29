"""
pipeline.py
===========
The data-generation pipeline.

For each requested scene type we:

  1. Pick a target SNR (uniformly within a configured band) and parameters.
  2. Generate the chirp strain (if BBH / BNS scene) at that SNR.
  3. Generate the glitch strain (if glitch scene) at its own SNR.
  4. Add coloured background noise.
  5. Compute three Q-scans:
        - combined (the image YOLO trains on)
        - chirp only (to derive the chirp mask)
        - glitch only (to derive the noise mask)
  6. Render the combined Q-scan to PNG.
  7. Threshold the per-component grids to get pixel masks, convert to
     YOLO seg polygons, and write the label file.

This decoupled-masking trick is the one alluded to in the paper: the
authors only had to "annotate" because they had the isolated components
available too.
"""
from __future__ import annotations
from waveforms import generate_bbh, generate_bns, RealChirpLoader

import os
import json
import math
import shutil
from dataclasses import asdict
from typing import Optional, Tuple, List, Dict
import numpy as np
from tqdm import tqdm

from config import Config, CLASS_NAMES
from waveforms import generate_bbh, generate_bns
from glitches import build_glitch_source
from noise import generate_background
from spectrograms import (
    compute_qscan, render_qscan_png, energy_to_pixel_grid,
)
from annotations import build_label_lines, save_yolo_label


SceneType = str  # "bbh", "bns", "bbh_glitch", "bns_glitch", "glitch", "background"


# =============================================================================
# A single scene
# =============================================================================
def _draw_snr(snr_bins: Tuple[float, ...], rng: np.random.Generator) -> float:
    """Uniform within one of the configured SNR bins (paper-style banding)."""
    bin_idx = int(rng.integers(0, len(snr_bins)))
    lo = snr_bins[bin_idx]
    hi = snr_bins[bin_idx + 1] if bin_idx + 1 < len(snr_bins) else lo + 3.0
    return float(rng.uniform(lo, hi))


def generate_sample(
    scene: SceneType,   # 现在固定为 "mixed"
    cfg: Config,
    glitch_source,
    rng: np.random.Generator,
) -> Dict:
    """Generate one synthetic sample containing background, chirp, and glitch(es)."""
    strain_cfg = cfg.strain
    meta: Dict = {"scene": scene}
    isolated_grids: Dict[str, np.ndarray] = {}

    # ---- 1. 背景 ----
    background = generate_background(strain_cfg, rng)

    # ---- 2. chirp（随机选择 BBH 或 BNS 或真实） ----
    chirp_strain = None
    if cfg.strain.use_real_chirp:
        from waveforms import RealChirpLoader
        loader = RealChirpLoader(cfg.strain.real_chirp_dir, strain_cfg)
        chirp_strain, gmeta = loader.sample(rng)
        t_m = 0.0
        params = {"type": "real", "file": gmeta["file"]}
        meta.update(chirp_snr=None, chirp_t_merger=t_m, chirp_params=params)
    else:
        # 随机选择 BBH 或 BNS
        if rng.random() < 0.5:
            snr = _draw_snr(cfg.bbh.snr_bins, rng)
            chirp_strain, t_m, params = generate_bbh(strain_cfg, cfg.bbh, snr, rng)
        else:
            snr = _draw_snr(cfg.bns.snr_bins, rng)
            chirp_strain, t_m, params = generate_bns(strain_cfg, cfg.bns, snr, rng)
        meta.update(chirp_snr=snr, chirp_t_merger=t_m, chirp_params=params)

    # ---- 3. glitch（多个） ----
    glitch_strain_total = None
    noise_grids = []
    glitch_metas = []
    num_glitches = rng.integers(cfg.glitch.num_glitches_range[0],
                                cfg.glitch.num_glitches_range[1] + 1)
    for _ in range(num_glitches):
        g_strain, g_meta = glitch_source.sample(rng)
        if glitch_strain_total is None:
            glitch_strain_total = g_strain
        else:
            glitch_strain_total += g_strain
        glitch_metas.append(g_meta)
        e_glitch, _, f_glitch = compute_qscan(g_strain + 0.05 * background,
                                              strain_cfg, cfg.qscan)
        grid = energy_to_pixel_grid(e_glitch, f_glitch, cfg.image)
        noise_grids.append(grid)
    meta["glitch"] = glitch_metas
    if noise_grids:
        isolated_grids["noise"] = np.maximum.reduce(noise_grids)

    # ---- 4. 合成 strain ----
    combined = background.copy()
    if chirp_strain is not None:
        combined += chirp_strain
    if glitch_strain_total is not None:
        combined += glitch_strain_total

    # ---- 5. 综合 Q-scan 和 chirp 孤立网格 ----
    energy_combined, t_q, f_q = compute_qscan(combined, strain_cfg, cfg.qscan)
    if chirp_strain is not None:
        e_chirp, _, f_chirp = compute_qscan(chirp_strain + 0.0 * background,
                                            strain_cfg, cfg.qscan)
        # 使用固定低阈值确保 chirp 总是被标注（只要存在任何能量）
        chirp_grid = energy_to_pixel_grid(e_chirp, f_chirp, cfg.image)
        # 强制采用非常低的阈值（捕获几乎所有非零能量）
        low_thresh = 0.01
        chirp_mask = chirp_grid > low_thresh
        # 将 mask 作为能量网格传给 build_label_lines（需要非零值）
        isolated_grids["chirp"] = chirp_mask.astype(float) * 25.0

        # ========== 增强 chirp 在综合图像中的可见性（亮度映射到随机范围） ==========
        if e_chirp.max() > 0:
            # 取 chirp 能量大于其最大值的 30% 的区域
            threshold = 0.3 * e_chirp.max()
            chirp_mask_energy = e_chirp > threshold
            if chirp_mask_energy.any():
                # 随机选择亮度范围
                ranges = [(12, 18), (15, 22), (18, 25), (10, 15), (20, 25)]
                LOW, HIGH = rng.choice(ranges)
                original_vals = e_chirp[chirp_mask_energy]
                min_orig = original_vals.min()
                max_orig = original_vals.max()
                if max_orig > min_orig:
                    mapped = LOW + (original_vals - min_orig) / (max_orig - min_orig) * (HIGH - LOW)
                else:
                    mapped = (LOW + HIGH) / 2
                energy_combined[chirp_mask_energy] = mapped
        # =================================================

    return {
        "meta": meta,
        "energy_combined": energy_combined,
        "times": t_q,
        "freqs": f_q,
        "isolated_grids": isolated_grids,
    }
# =============================================================================
# Disk writer
# =============================================================================
def write_sample(
    sample: Dict,
    image_path: str,
    label_path: str,
    cfg: Config,
) -> None:
    render_qscan_png(
        sample["energy_combined"],
        sample["times"],
        sample["freqs"],
        image_path,
        cfg.image,
    )
    lines = build_label_lines(sample["isolated_grids"], cfg.image, cfg.qscan)
    save_yolo_label(lines, label_path)


# =============================================================================
# Dataset assembly
# =============================================================================
def _build_layout(root: str) -> None:
    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(root, "images", split), exist_ok=True)
        os.makedirs(os.path.join(root, "labels", split), exist_ok=True)


def _split_index(idx: int, total: int, splits: Tuple[float, float, float]) -> str:
    cum = np.cumsum(splits)
    p = idx / max(total, 1)
    if p < cum[0]:
        return "train"
    if p < cum[1]:
        return "val"
    return "test"


def _scene_plan(cfg: Config) -> List[SceneType]:
    """All samples are mixed (background + chirp + glitch)."""
    return ["mixed"] * cfg.dataset.total_samples


def write_data_yaml(cfg: Config) -> None:
    root = os.path.abspath(cfg.dataset.output_root)
    content = (
        f"path: {root}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: {CLASS_NAMES}\n"
    )
    with open(os.path.join(root, "data.yaml"), "w") as f:
        f.write(content)


def run_serial(cfg: Config) -> None:
    """Single-process generation. Use for debugging; prefer run_parallel."""
    rng = np.random.default_rng(cfg.dataset.seed)
    _build_layout(cfg.dataset.output_root)
    glitch_source = build_glitch_source(cfg.strain, cfg.glitch)

    plan = _scene_plan(cfg)
    rng.shuffle(plan)

    metadata = []
    for i, scene in enumerate(tqdm(plan, desc="generating")):
        split = _split_index(i, len(plan), cfg.dataset.split)
        stem = f"{i:07d}_{scene}"
        img_path = os.path.join(cfg.dataset.output_root, "images", split,
                                stem + ".png")
        lbl_path = os.path.join(cfg.dataset.output_root, "labels", split,
                                stem + ".txt")
        try:
            sample = generate_sample(scene, cfg, glitch_source, rng)
            write_sample(sample, img_path, lbl_path, cfg)
            sample["meta"]["split"] = split
            sample["meta"]["stem"] = stem
            metadata.append(sample["meta"])
        except Exception as e:                                      # noqa: BLE001
            print(f"[warn] sample {i} ({scene}) failed: {e}")

    write_data_yaml(cfg)
    with open(os.path.join(cfg.dataset.output_root, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, default=str)


# =============================================================================
# Parallel generation (worker pool)
# =============================================================================
def _worker(args):
    (idx, scene, split, stem, cfg, child_seed) = args
    rng = np.random.default_rng(child_seed)
    glitch_source = build_glitch_source(cfg.strain, cfg.glitch)
    img_path = os.path.join(cfg.dataset.output_root, "images", split,
                            stem + ".png")
    lbl_path = os.path.join(cfg.dataset.output_root, "labels", split,
                            stem + ".txt")
    try:
        sample = generate_sample(scene, cfg, glitch_source, rng)
        write_sample(sample, img_path, lbl_path, cfg)
        meta = sample["meta"]
        meta["split"] = split
        meta["stem"] = stem
        return meta
    except Exception as e:                                          # noqa: BLE001
        return {"scene": scene, "stem": stem, "error": str(e),
                "split": split}


def run_parallel(cfg: Config) -> None:
    import multiprocessing as mp

    _build_layout(cfg.dataset.output_root)
    plan = _scene_plan(cfg)
    rng = np.random.default_rng(cfg.dataset.seed)
    rng.shuffle(plan)

    seeds = rng.integers(0, 2**31 - 1, size=len(plan))
    tasks = [
        (i, scene, _split_index(i, len(plan), cfg.dataset.split),
         f"{i:07d}_{scene}", cfg, int(seeds[i]))
        for i, scene in enumerate(plan)
    ]

    metadata = []
    with mp.Pool(cfg.dataset.n_workers) as pool:
        for meta in tqdm(pool.imap_unordered(_worker, tasks, chunksize=4),
                         total=len(tasks), desc="generating"):
            metadata.append(meta)

    write_data_yaml(cfg)
    with open(os.path.join(cfg.dataset.output_root, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, default=str)
