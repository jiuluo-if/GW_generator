#!/usr/bin/env python3
"""
prepare_chirp_metadata.py (简化版)
仅通过 SNR 判断 .npy 文件是否包含引力波信号。
跳过 GPS 解析和 DQ 检查，避免模块依赖问题。
"""
import os
import json
import numpy as np
import shutil
from pycbc.filter import sigma
from pycbc.psd.analytical import from_string as psd_from_string
from pycbc.types import TimeSeries as PyCBC_TimeSeries

# 配置
REAL_CHIRP_DIR = "/home/jiuluo/python_demo/new/real_chirps"
METADATA_FILE = os.path.join(REAL_CHIRP_DIR, "chirp_metadata.json")
REJECT_DIR = os.path.join(REAL_CHIRP_DIR, "rejected")
SAMPLE_RATE = 4096
DURATION = 3.0
N_SAMPLES = int(SAMPLE_RATE * DURATION)
SNR_THRESHOLD = 8.0

def compute_snr(strain, sample_rate, f_lower=20.0):
    """计算相对于 aLIGO 设计 PSD 的信噪比"""
    # 确保长度一致
    if len(strain) != N_SAMPLES:
        if len(strain) > N_SAMPLES:
            strain = strain[:N_SAMPLES]
        else:
            strain = np.pad(strain, (0, N_SAMPLES - len(strain)))
    delta_t = 1.0 / sample_rate
    flen = N_SAMPLES // 2 + 1
    delta_f = 1.0 / DURATION
    psd = psd_from_string("aLIGOaLIGODesignSensitivityT1800044", flen, delta_f, f_lower)
    ts = PyCBC_TimeSeries(strain, delta_t=delta_t, epoch=0)
    snr = sigma(ts, psd=psd, low_frequency_cutoff=f_lower)
    return snr

def main():
    os.makedirs(REJECT_DIR, exist_ok=True)
    metadata = {}
    npy_files = [f for f in os.listdir(REAL_CHIRP_DIR) 
                 if f.endswith('.npy') and f != "chirp_metadata.json"]
    for filename in npy_files:
        filepath = os.path.join(REAL_CHIRP_DIR, filename)
        print(f"Processing {filename}...")
        strain = np.load(filepath).flatten()
        if len(strain) < 100:
            print(f"  -> Too short, moving to reject")
            shutil.move(filepath, os.path.join(REJECT_DIR, filename))
            continue
        
        # 截取中央 N_SAMPLES 长度的片段
        if len(strain) > N_SAMPLES:
            center = len(strain)//2
            strain_seg = strain[center - N_SAMPLES//2 : center + N_SAMPLES//2]
        else:
            strain_seg = strain[:N_SAMPLES]
        
        try:
            snr = compute_snr(strain_seg, SAMPLE_RATE)
        except Exception as e:
            print(f"  -> SNR computation failed: {e}, rejecting")
            shutil.move(filepath, os.path.join(REJECT_DIR, filename))
            continue
        
        if snr < SNR_THRESHOLD:
            print(f"  -> SNR too low ({snr:.2f} < {SNR_THRESHOLD}), rejecting")
            shutil.move(filepath, os.path.join(REJECT_DIR, filename))
            continue
        
        metadata[filename] = {"file": filename, "snr": float(snr), "length": len(strain)}
        print(f"  -> Accepted (SNR={snr:.2f})")
    
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved to {METADATA_FILE}")
    print(f"Accepted {len(metadata)} files out of {len(npy_files)}.")

if __name__ == "__main__":
    main()
