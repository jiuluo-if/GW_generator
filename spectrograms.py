"""
spectrograms.py
===============
Q-transform of a timeseries -> 2D normalised-energy array on a fixed
(time x frequency) grid, plus PNG renderer.

Uses GWpy's TimeSeries.q_transform — the same code path that GravitySpy
uses to generate its training images — so the morphologies look exactly
like what the LIGO detector characterisation team sees.
"""
from __future__ import annotations

import io
import numpy as np
from typing import Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import StrainCfg, QScanCfg, ImageCfg


def _import_gwpy():
    from gwpy.timeseries import TimeSeries
    return TimeSeries


# -----------------------------------------------------------------------------
# Core Q-transform
# -----------------------------------------------------------------------------
def compute_qscan(
    strain: np.ndarray,
    strain_cfg: StrainCfg,
    qscan_cfg: QScanCfg,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (energy[freq, time], times, freqs) at a fixed resolution.
    
    Output energy is normalized to [0, 25] range.
    """
    TimeSeries = _import_gwpy()
    
    # Helper to create zero grid on failure (minimum dimension = 2 for interpolation)
    def zero_grid():
        n_t = max(2, int(strain_cfg.duration / qscan_cfg.tres) + 1)
        n_f = max(2, int((qscan_cfg.frange[1] - qscan_cfg.frange[0]) / qscan_cfg.fres) + 1)
        times = np.linspace(0, strain_cfg.duration, n_t)
        freqs = np.linspace(qscan_cfg.frange[0], qscan_cfg.frange[1], n_f)
        return np.zeros((n_f, n_t)), times, freqs

    # Check input validity
    if not np.all(np.isfinite(strain)):
        return zero_grid()

    try:
        ts = TimeSeries(strain, sample_rate=strain_cfg.sample_rate, epoch=0)
        
        if qscan_cfg.whiten:
            try:
                ts = ts.whiten(fduration=1.0)
                if not np.all(np.isfinite(ts.value)):
                    ts = TimeSeries(strain, sample_rate=strain_cfg.sample_rate, epoch=0)
            except Exception:
                ts = TimeSeries(strain, sample_rate=strain_cfg.sample_rate, epoch=0)
        
        q = ts.q_transform(
            qrange=qscan_cfg.qrange,
            frange=qscan_cfg.frange,
            tres=qscan_cfg.tres,
            fres=qscan_cfg.fres,
            outseg=(0, strain_cfg.duration),
            whiten=False,
        )
        energy = np.asarray(q.value).T   # shape (n_freq, n_time)
        times = np.asarray(q.times.value) - q.t0.value
        freqs = np.asarray(q.frequencies.value)
        
        # ---- NORMALIZE energy to [0, 25] ----
        max_energy = np.percentile(energy, 99.9)
        if max_energy > 0:
            energy = np.clip(energy / max_energy * 25.0, 0, 25.0)
        else:
            energy = np.zeros_like(energy)
        # -------------------------------------
        
        return energy, times, freqs
    except Exception:
        return zero_grid()


# -----------------------------------------------------------------------------
# Image rendering
# -----------------------------------------------------------------------------
def render_qscan_png(
    energy: np.ndarray,
    times: np.ndarray,
    freqs: np.ndarray,
    out_path: str,
    image_cfg: ImageCfg,
) -> None:
    """Render the Q-scan to a fixed-size PNG, exactly the format used for
    YOLO training (no axes, no margins)."""
    fig = plt.figure(
        figsize=(image_cfg.width / image_cfg.dpi,
                 image_cfg.height / image_cfg.dpi),
        dpi=image_cfg.dpi,
    )
    ax = fig.add_axes([0, 0, 1, 1])       # full-bleed, no padding
    ax.set_axis_off()

    # Log-y if requested: resample energy to a log-frequency grid so YOLO
    # sees the same axis the human reader does.
    if image_cfg.log_freq:
        log_f = np.logspace(np.log10(freqs[0]), np.log10(freqs[-1]),
                            image_cfg.height)
        # 1-D interp on each time column
        from scipy.interpolate import interp1d
        interp = interp1d(freqs, energy, axis=0, bounds_error=False,
                          fill_value=0.0)
        energy_disp = interp(log_f)
    else:
        energy_disp = energy

    ax.imshow(
        energy_disp,
        origin="lower",
        aspect="auto",
        extent=(times[0], times[-1], 0, image_cfg.height),
        vmin=image_cfg.vmin,
        vmax=image_cfg.vmax,
        cmap=image_cfg.cmap,
        interpolation="nearest",
    )
    fig.savefig(out_path, dpi=image_cfg.dpi, pad_inches=0)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Pixel-coordinate utilities
# -----------------------------------------------------------------------------
def energy_to_pixel_grid(
    energy: np.ndarray,
    freqs: np.ndarray,
    image_cfg: ImageCfg,
) -> np.ndarray:
    """Resample energy onto exactly the pixel grid used in the rendered PNG.

    Uses nearest-neighbor interpolation to prevent energy smearing.
    """
    from scipy.interpolate import interp1d

    # Validate inputs
    if energy.size == 0 or freqs.size < 2 or energy.shape[0] != freqs.size:
        return np.zeros((image_cfg.height, image_cfg.width))

    # Clean NaNs/Infs
    if not np.all(np.isfinite(energy)):
        energy = np.nan_to_num(energy, nan=0.0, posinf=0.0, neginf=0.0)

    try:
        if image_cfg.log_freq:
            f_min = max(freqs[0], 1e-10)
            f_max = max(freqs[-1], f_min + 1e-10)
            new_f = np.logspace(np.log10(f_min), np.log10(f_max), image_cfg.height)
        else:
            new_f = np.linspace(freqs[0], freqs[-1], image_cfg.height)

        interp_f = interp1d(freqs, energy, axis=0, kind='nearest',
                            bounds_error=False, fill_value=0.0)
        e_f = interp_f(new_f)

        t_old = np.linspace(0, 1, e_f.shape[1])
        t_new = np.linspace(0, 1, image_cfg.width)
        interp_t = interp1d(t_old, e_f, axis=1, kind='nearest',
                            bounds_error=False, fill_value=0.0)
        grid = interp_t(t_new)
        return grid
    except Exception:
        return np.zeros((image_cfg.height, image_cfg.width))
