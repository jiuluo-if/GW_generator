"""
glitches.py
===========
Transient noise (glitch) timeseries supply.

Two backends:

1. ``GravitySpyLoader`` — reads precomputed glitch timeseries from a directory
   (recommended: the public GravitySpy O3 Zenodo dataset, or LIGO-internal
   strain pulled around Omicron triggers).

2. ``SimulatedGlitchGenerator`` — analytic models of the most common LIGO
   glitch families (blip, tomte, koi-fish, scattered-light, low-frequency
   burst, whistle, extremely-loud). Lets you generate massive training data
   without needing access to detector data.

Both return a numpy array of length ``sample_rate * duration``.
"""
from __future__ import annotations

import os
import glob
import numpy as np
from typing import Tuple, List

from config import StrainCfg, GlitchCfg


# =============================================================================
# Real-data loader
# =============================================================================
class GravitySpyLoader:
    """Load real glitch timeseries dumped as .npy or .txt files."""

    def __init__(self, directory: str, strain_cfg: StrainCfg):
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"glitch dir not found: {directory}")
        self.files: List[str] = sorted(
            glob.glob(os.path.join(directory, "**/*.npy"), recursive=True) +
            glob.glob(os.path.join(directory, "**/*.txt"), recursive=True)
        )
        if not self.files:
            raise RuntimeError(f"no .npy/.txt files in {directory}")
        self.strain_cfg = strain_cfg

    def sample(self, rng: np.random.Generator) -> Tuple[np.ndarray, dict]:
        path = self.files[int(rng.integers(0, len(self.files)))]
        if path.endswith(".npy"):
            ts = np.load(path)
        else:
            ts = np.loadtxt(path)
        # Ensure 1D array (some saved data may be 2D)
        if ts.ndim == 2:
            ts = ts[:, 0]
        ts = ts.flatten()
        ts = self._fit_to_window(ts, rng)
        # Extract family label from parent directory name
        family_label = os.path.basename(os.path.dirname(path))
        return ts, {"family": family_label, "source_file": os.path.basename(path)}

    def _fit_to_window(self, ts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        N = int(self.strain_cfg.sample_rate * self.strain_cfg.duration)
        if len(ts) >= N:
            # 随机截取一段连续的 N 个样本（保证截取片段完整）
            max_start = len(ts) - N
            start = rng.integers(0, max_start + 1)
            return ts[start:start + N]
        else:
            # 长度小于窗口，随机放置在窗口内（不超出边界）
            out = np.zeros(N)
            max_offset = N - len(ts)
            offset = rng.integers(0, max_offset + 1)
            out[offset:offset + len(ts)] = ts
            return out


# =============================================================================
# Simulated glitches
# =============================================================================
class SimulatedGlitchGenerator:
    """Parametric glitch simulator.

    The waveforms here are *not* physical models of the underlying noise
    coupling; they are chosen to *reproduce the morphology* of each
    GravitySpy class in time-frequency, which is what the YOLO model
    actually sees.
    """

    def __init__(self, strain_cfg: StrainCfg, glitch_cfg: GlitchCfg):
        self.strain_cfg = strain_cfg
        self.glitch_cfg = glitch_cfg
        self.fs = strain_cfg.sample_rate
        self.N = int(self.fs * strain_cfg.duration)
        self.t = np.arange(self.N) / self.fs

    # --------------- public ---------------------------------------------------
    def sample(self, rng: np.random.Generator) -> Tuple[np.ndarray, dict]:
        family = rng.choice(self.glitch_cfg.families)
        snr_target = rng.uniform(*self.glitch_cfg.snr_range)
        h = self._dispatch(family, rng)
        h = self._normalise_snr(h, snr_target)
        # Guard against NaN/Inf from degenerate waveforms
        if not np.all(np.isfinite(h)):
            h = np.zeros_like(h)
        return h, {"family": str(family), "snr": float(snr_target)}

    def _dispatch(self, family: str, rng: np.random.Generator) -> np.ndarray:
        fn = {
            "blip": self._blip,
            "tomte": self._tomte,
            "koi_fish": self._koi_fish,
            "scattered_light": self._scattered_light,
            "low_freq_burst": self._low_freq_burst,
            "whistle": self._whistle,
            "extremely_loud": self._extremely_loud,
        }.get(family)
        if fn is None:
            raise ValueError(f"unknown glitch family {family}")
        return fn(rng)

    # --------------- helpers --------------------------------------------------
    def _centre_time(self, rng: np.random.Generator,
                     pad: float = 0.4) -> float:
        """Pick a random central time, keeping the glitch inside the window."""
        return rng.uniform(pad, self.strain_cfg.duration - pad)

    def _normalise_snr(self, h: np.ndarray, target_snr: float) -> np.ndarray:
        # Approximate SNR against a white-noise reference. Good enough for
        # generating training morphologies; the real PSD whitening happens
        # later in the Q-scan step.
        rms = np.sqrt(np.mean(h ** 2)) + 1e-30
        return h * (target_snr / (rms * np.sqrt(self.N)))

    # =====================  glitch families  =================================
    def _blip(self, rng):
        """Short broadband ~ms pulse, ~30–300 Hz."""
        t0 = self._centre_time(rng)
        sigma = rng.uniform(0.003, 0.012)
        f0 = rng.uniform(40, 250)
        env = np.exp(-((self.t - t0) ** 2) / (2 * sigma ** 2))
        return env * np.cos(2 * np.pi * f0 * (self.t - t0))

    def _tomte(self, rng):
        """Like blip but lower frequency, longer."""
        t0 = self._centre_time(rng)
        sigma = rng.uniform(0.020, 0.060)
        f0 = rng.uniform(20, 90)
        env = np.exp(-((self.t - t0) ** 2) / (2 * sigma ** 2))
        return env * np.cos(2 * np.pi * f0 * (self.t - t0))

    def _koi_fish(self, rng):
        """Loud broadband with a downward chirp."""
        t0 = self._centre_time(rng)
        sigma = rng.uniform(0.04, 0.10)
        f_start = rng.uniform(150, 300)
        f_end = rng.uniform(20, 80)
        tt = self.t - t0
        env = np.exp(-(tt ** 2) / (2 * sigma ** 2))
        phase = 2 * np.pi * (f_start * tt +
                             0.5 * (f_end - f_start) / sigma * tt ** 2)
        return env * np.cos(phase)

    def _scattered_light(self, rng):
        """Stacked low-frequency arches, classic for LIGO."""
        n_arches = int(rng.integers(2, 6))
        t0 = self._centre_time(rng, pad=0.7)
        h = np.zeros_like(self.t)
        for k in range(n_arches):
            f_peak = rng.uniform(15, 45) * (1 + 0.2 * k)
            sigma = rng.uniform(0.20, 0.45)
            tt = self.t - t0
            env = np.exp(-(tt ** 2) / (2 * sigma ** 2))
            phase = 2 * np.pi * f_peak * tt * (1 - tt / (2 * sigma + 1e-3))
            h += env * np.cos(phase) * (1.0 / (k + 1))
        return h

    def _low_freq_burst(self, rng):
        """Smooth low-frequency bump < 30 Hz."""
        t0 = self._centre_time(rng)
        sigma = rng.uniform(0.05, 0.20)
        f0 = rng.uniform(10, 28)
        env = np.exp(-((self.t - t0) ** 2) / (2 * sigma ** 2))
        return env * np.cos(2 * np.pi * f0 * (self.t - t0))

    def _whistle(self, rng):
        """Long, high-Q upward sweep — looks like a thin sloped line."""
        t0 = self._centre_time(rng, pad=0.6)
        duration = rng.uniform(0.2, 0.8)
        f0 = rng.uniform(80, 200)
        f1 = f0 + rng.uniform(50, 250)
        tt = self.t - t0
        env = np.where(np.abs(tt) < duration / 2,
                       np.cos(np.pi * tt / duration) ** 2, 0.0)
        phase = 2 * np.pi * (f0 * tt + 0.5 * (f1 - f0) / duration * tt ** 2)
        return env * np.cos(phase)

    def _extremely_loud(self, rng):
        """Saturated broadband: stack blips + low-freq energy."""
        h = self._blip(rng) * 5
        h += self._low_freq_burst(rng) * 3
        h += self._koi_fish(rng) * 2
        return h


# =============================================================================
# Factory
# =============================================================================
def build_glitch_source(strain_cfg: StrainCfg, glitch_cfg: GlitchCfg):
    if glitch_cfg.source == "gravityspy":
        return GravitySpyLoader(glitch_cfg.gravityspy_dir, strain_cfg)
    if glitch_cfg.source == "simulated":
        return SimulatedGlitchGenerator(strain_cfg, glitch_cfg)
    raise ValueError(f"unknown glitch source {glitch_cfg.source}")
