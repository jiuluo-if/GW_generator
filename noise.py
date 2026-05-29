"""
noise.py
========
Coloured Gaussian background noise (LIGO-like) with additional realistic disturbances.
"""
from __future__ import annotations

import numpy as np
from scipy import signal
from config import StrainCfg


def _import_pycbc_psd():
    from pycbc.psd.analytical import from_string as psd_from_string
    from pycbc.noise import noise_from_psd
    return psd_from_string, noise_from_psd


def generate_background(strain_cfg: StrainCfg,
                        rng: np.random.Generator) -> np.ndarray:
    """Generate colored Gaussian noise with additional spikes, low-frequency drift,
    and narrow-band disturbances."""
    psd_from_string, noise_from_psd = _import_pycbc_psd()
    flen = int(strain_cfg.sample_rate * strain_cfg.duration) // 2 + 1
    delta_f = 1.0 / strain_cfg.duration
    psd = psd_from_string(strain_cfg.detector_psd, flen, delta_f,
                          strain_cfg.f_lower)
    n_samples = int(strain_cfg.sample_rate * strain_cfg.duration)
    delta_t = 1.0 / strain_cfg.sample_rate
    seed = int(rng.integers(0, 2**31 - 1))
    ts = noise_from_psd(n_samples, delta_t, psd, seed=seed)
    noise = ts.numpy()

    # 1. 随机尖峰 (已有的)
    if strain_cfg.background_spike_rate > 0:
        num_spikes = rng.poisson(strain_cfg.background_spike_rate * n_samples)
        if num_spikes > 0:
            spike_positions = rng.choice(n_samples, size=num_spikes, replace=False)
            std_noise = np.std(noise)
            spike_amps = rng.normal(0, strain_cfg.background_spike_amplitude * std_noise,
                                    size=num_spikes)
            noise[spike_positions] += spike_amps

    # 2. 低频漂移 (0.1 - 5 Hz)
    if rng.random() < 0.5:  # 50% 的概率添加
        t = np.arange(n_samples) / strain_cfg.sample_rate
        f_drift = rng.uniform(0.1, 5.0)
        amplitude_drift = rng.uniform(0.1, 0.5) * np.std(noise)
        drift = amplitude_drift * np.sin(2 * np.pi * f_drift * t)
        noise += drift

    # 3. 窄带噪声 (如60Hz谐波等)
    if rng.random() < 0.3:
        num_tones = rng.integers(1, 4)
        for _ in range(num_tones):
            f_center = rng.uniform(20, 500)  # 频率中心
            q = rng.uniform(10, 50)          # 品质因数
            # 生成窄带噪声：高斯白噪声带通滤波
            duration = strain_cfg.duration
            bw = f_center / q
            sos = signal.butter(4, [f_center - bw/2, f_center + bw/2], btype='band', fs=strain_cfg.sample_rate, output='sos')
            white = rng.normal(0, 1, n_samples)
            narrow = signal.sosfilt(sos, white)
            narrow = narrow / np.std(narrow) * 0.3 * np.std(noise)
            noise += narrow

    # 4. 随机突发噪声 (短时脉冲簇)
    if rng.random() < 0.4:
        num_bursts = rng.integers(1, 4)
        for _ in range(num_bursts):
            burst_len = rng.integers(int(0.01 * strain_cfg.sample_rate), int(0.1 * strain_cfg.sample_rate))
            start = rng.integers(0, n_samples - burst_len)
            burst_amp = rng.uniform(1.0, 3.0) * np.std(noise)
            burst = rng.normal(0, burst_amp, burst_len)
            noise[start:start+burst_len] += burst

    return noise
