"""
waveforms.py
============
Compact-binary-coalescence waveform generation, matching the paper's recipe:

  * BBH: m1, m2 > 5 M_sun, IMRPhenomD or similar, sampled to match O1-O3 events
  * BNS: m1, m2 < 2.5 M_sun, TaylorF2 (fast, fine for BNS inspiral)

The signal is generated in the detector frame at requested SNR by rescaling
against a chosen aLIGO PSD. The returned object is a numpy array of length
duration * sample_rate, centred (more or less) on the merger.
"""
from __future__ import annotations
import os

import numpy as np
from typing import Tuple

from config import StrainCfg, BBHCfg, BNSCfg


# Lazy imports — only fail at call time so the module imports cleanly even
# when PyCBC isn't installed yet (e.g. during code review).
def _import_pycbc():
    from pycbc.waveform import get_td_waveform
    from pycbc.psd.analytical import from_string as psd_from_string
    from pycbc.filter import sigma
    from pycbc.types import TimeSeries
    return get_td_waveform, psd_from_string, sigma, TimeSeries


# -----------------------------------------------------------------------------
# Parameter sampling
# -----------------------------------------------------------------------------
def sample_bbh_params(rng: np.random.Generator, cfg: BBHCfg) -> dict:
    m1 = rng.uniform(*cfg.m1_range)
    m2 = rng.uniform(*cfg.m2_range)
    if m2 > m1:                                   # convention: m1 >= m2
        m1, m2 = m2, m1
    s1z = rng.uniform(*cfg.spin_range)
    s2z = rng.uniform(*cfg.spin_range)
    distance = rng.uniform(*cfg.distance_range)
    inclination = np.arccos(rng.uniform(-1, 1))   # isotropic
    return dict(mass1=m1, mass2=m2, spin1z=s1z, spin2z=s2z,
                distance=distance, inclination=inclination,
                approximant=cfg.waveform_approximant)


def sample_bns_params(rng: np.random.Generator, cfg: BNSCfg) -> dict:
    m1 = rng.uniform(*cfg.m1_range)
    m2 = rng.uniform(*cfg.m2_range)
    if m2 > m1:
        m1, m2 = m2, m1
    s1z = rng.uniform(*cfg.spin_range)
    s2z = rng.uniform(*cfg.spin_range)
    distance = rng.uniform(*cfg.distance_range)
    inclination = np.arccos(rng.uniform(-1, 1))
    return dict(mass1=m1, mass2=m2, spin1z=s1z, spin2z=s2z,
                distance=distance, inclination=inclination,
                approximant=cfg.waveform_approximant)


# -----------------------------------------------------------------------------
# Core waveform generation
# -----------------------------------------------------------------------------
def generate_cbc_strain(
    params: dict,
    target_snr: float,
    strain_cfg: StrainCfg,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, float]:
    """
    Generate a CBC time-domain strain, rescale to target_snr against the PSD,
    and embed it into a fixed-length zero array of `strain_cfg.duration`.

    Returns
    -------
    h : np.ndarray
        Strain timeseries of length sample_rate * duration.
    t_merger : float
        Merger time relative to the start of the window (seconds).
    """
    get_td_waveform, psd_from_string, sigma, TimeSeries = _import_pycbc()

    delta_t = 1.0 / strain_cfg.sample_rate
    hp, _ = get_td_waveform(
        delta_t=delta_t,
        f_lower=strain_cfg.f_lower,
        **params,
    )

    # PSD for normalisation
    flen = int(strain_cfg.sample_rate * strain_cfg.duration) // 2 + 1
    delta_f = 1.0 / strain_cfg.duration
    psd = psd_from_string(strain_cfg.detector_psd, flen, delta_f,
                          strain_cfg.f_lower)

    target_len = int(strain_cfg.sample_rate * strain_cfg.duration)
    # Work with numpy array to avoid TimeSeries length confusion
    hp_np = hp.numpy()

    # Truncate if too long (keep last part, containing merger)
    if len(hp_np) > target_len:
        hp_np = hp_np[-target_len:]

    # Now len(hp_np) <= target_len, pad at start
    pad_len = target_len - len(hp_np)
    hp_padded = TimeSeries(
        np.concatenate([np.zeros(pad_len), hp_np]),
        delta_t=delta_t,
    )

    # Current SNR against the PSD
    current_snr = sigma(hp_padded, psd=psd, low_frequency_cutoff=strain_cfg.f_lower)
    if current_snr <= 0 or not np.isfinite(current_snr):
        raise RuntimeError("Could not compute waveform SNR (degenerate parameters).")
    scale = target_snr / current_snr
    h = hp_padded.numpy() * scale

    # Random temporal jitter so the merger isn't always at the same time bin
    max_shift = int(0.5 * strain_cfg.sample_rate)          # +/- 0.5 s
    shift = rng.integers(-max_shift, max_shift + 1)
    h = np.roll(h, shift)

    # Approximate merger time: largest |h|
    t_merger = float(np.argmax(np.abs(h))) / strain_cfg.sample_rate
    return h, t_merger


# Convenience wrappers
def generate_bbh(strain_cfg: StrainCfg, bbh_cfg: BBHCfg,
                 target_snr: float, rng: np.random.Generator):
    p = sample_bbh_params(rng, bbh_cfg)
    h, t_merger = generate_cbc_strain(p, target_snr, strain_cfg, rng)
    return h, t_merger, p


def generate_bns(strain_cfg: StrainCfg, bns_cfg: BNSCfg,
                 target_snr: float, rng: np.random.Generator):
    p = sample_bns_params(rng, bns_cfg)
    h, t_merger = generate_cbc_strain(p, target_snr, strain_cfg, rng)
    return h, t_merger, p
    

# ========== 真实引力波加载器 ==========
class RealChirpLoader:
    """从预下载且经过 SNR 筛选的 .npy 文件中加载真实 Chirp 信号"""
    def __init__(self, data_dir: str, strain_cfg: StrainCfg):
        import json
        self.data_dir = data_dir
        self.metadata_file = os.path.join(data_dir, "chirp_metadata.json")
        if not os.path.exists(self.metadata_file):
            raise FileNotFoundError(f"Metadata file {self.metadata_file} not found. Run prepare_chirp_metadata.py first.")
        with open(self.metadata_file, 'r') as f:
            self.metadata = json.load(f)
        self.valid_files = list(self.metadata.keys())
        if not self.valid_files:
            raise RuntimeError(f"No valid chirp files found in {data_dir} after SNR screening.")
        self.N = int(strain_cfg.sample_rate * strain_cfg.duration)
        self.sample_rate = strain_cfg.sample_rate

    def sample(self, rng: np.random.Generator):
        filename = rng.choice(self.valid_files)
        filepath = os.path.join(self.data_dir, filename)
        h = np.load(filepath).flatten()
        # 确保长度正好为 N（截取中心段）
        if len(h) >= self.N:
            start = (len(h) - self.N) // 2
            h = h[start:start+self.N]
        else:
            pad = self.N - len(h)
            h = np.pad(h, (pad//2, pad - pad//2), constant_values=0)
        # 随机时间偏移
        max_shift = int(0.5 * self.sample_rate)
        shift = rng.integers(-max_shift, max_shift + 1)
        h = np.roll(h, shift)
        # 随机振幅缩放（避免过弱）
        scale = rng.uniform(0.8, 2.0)
        h = h * scale
        return h, {
            "source": "real_gw",
            "file": filename,
            "scale": scale,
            "snr": self.metadata[filename].get("snr", None)
        }
