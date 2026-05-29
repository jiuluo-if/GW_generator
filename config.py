"""
config.py
=========
Central configuration for GW-YOLO synthetic data generation.

All physical, signal-processing, image and YOLO-format parameters live here
so that the rest of the code stays a thin pipeline.
"""
from dataclasses import dataclass, field
from typing import Tuple, List, Dict


# -----------------------------------------------------------------------------
# Class definitions for YOLO labels
# -----------------------------------------------------------------------------
CLASS_NAMES: List[str] = ["chirp", "noise"]
CLASS_ID: Dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}


@dataclass
class StrainCfg:
    """Time-domain strain / sampling configuration."""
    sample_rate: int = 4096            # Hz, LIGO standard
    duration: float = 3.0              # seconds, paper uses 3 s windows
    f_lower: float = 20.0              # Hz, low-frequency cutoff for waveforms
    detector_psd: str = "aLIGOaLIGODesignSensitivityT1800044"  # PyCBC PSD name
    use_real_psd_if_available: bool = False  # set True if you have a PSD txt
    use_real_chirp: bool = True                     # 启用真实引力波
    real_chirp_dir: str = "/home/jiuluo/python_demo/new/real_chirps"
    background_spike_rate: float = 0.01        # 尖峰密度
    background_spike_amplitude: float = 3.0    # 尖峰强度倍数

@dataclass
class BBHCfg:
    """Binary black hole prior — values cover real O1-O3 detections."""
    m1_range: Tuple[float, float] = (5.0,  80.0)   # M_sun
    m2_range: Tuple[float, float] = (5.0,  80.0)
    spin_range: Tuple[float, float] = (-0.9, 0.9)
    distance_range: Tuple[float, float] = (100.0, 4000.0)  # Mpc
    waveform_approximant: str = "IMRPhenomD"
    snr_bins: Tuple[float, ...] = tuple(range(6, 51, 3))    # 6,9,...,48


@dataclass
class BNSCfg:
    """Binary neutron star prior."""
    m1_range: Tuple[float, float] = (1.1, 2.4)
    m2_range: Tuple[float, float] = (1.1, 2.4)
    spin_range: Tuple[float, float] = (-0.05, 0.05)
    distance_range: Tuple[float, float] = (30.0, 400.0)
    waveform_approximant: str = "TaylorF2"
    snr_bins: Tuple[float, ...] = tuple(range(12, 51, 3))


@dataclass
class GlitchCfg:
    """Glitch handling. Choose ONE source.

    source = "gravityspy" -> load real timeseries from a directory
             "simulated"  -> sample from a parametric family below
    """
    source: str = "gravityspy"           # "gravityspy" | "simulated"
    gravityspy_dir: str = "/home/jiuluo/python_demo/new/npy"            # only used if source == "gravityspy"
    snr_range: Tuple[float, float] = (7.5, 60.0)
    # Mix of synthetic glitch families used when source == "simulated"
    families: Tuple[str, ...] = (
        "blip", "tomte", "koi_fish", "scattered_light",
        "low_freq_burst", "whistle", "extremely_loud",
    )
    num_glitches_range: Tuple[int, int] = (1, 3)


@dataclass
class QScanCfg:
    """Q-transform parameters for both training spectrograms and mask labels."""
    frange: Tuple[float, float] = (10.0, 1000.0)
    qrange: Tuple[float, float] = (4.0, 64.0)
    tres: float = 0.002                 # seconds
    fres: float = 0.2                   # Hz
    whiten: bool = True
    # Mask thresholds (normalised energy of the *isolated* component)
    chirp_mask_threshold: float = 0.1
    noise_mask_threshold: float = 6.0
    min_blob_pixels: int = 10          # drop tiny mask fragments


@dataclass
class ImageCfg:
    """Image rendering for the YOLO input."""
    width: int = 640                    # px
    height: int = 640
    log_freq: bool = True               # log-scaled y-axis
    cmap: str = "viridis"
    vmin: float = 0.0
    vmax: float = 25.0                  # paper uses ~25 normalised energy
    dpi: int = 100


@dataclass
class DatasetCfg:
    output_root: str = "./gw_yolo_dataset"
    split: Tuple[float, float, float] = (0.8, 0.1, 0.1)
    total_samples: int = 10000           # 总样本数（替代各种 x_only）
    seed: int = 42
    n_workers: int = 8
    save_intermediate: bool = False


@dataclass
class Config:
    strain: StrainCfg = field(default_factory=StrainCfg)
    bbh: BBHCfg = field(default_factory=BBHCfg)
    bns: BNSCfg = field(default_factory=BNSCfg)
    glitch: GlitchCfg = field(default_factory=GlitchCfg)
    qscan: QScanCfg = field(default_factory=QScanCfg)
    image: ImageCfg = field(default_factory=ImageCfg)
    dataset: DatasetCfg = field(default_factory=DatasetCfg)


DEFAULT = Config()
