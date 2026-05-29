#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from config import DEFAULT
from waveforms import generate_bbh
from spectrograms import compute_qscan, energy_to_pixel_grid
from annotations import mask_from_energy, mask_to_polygons

# 设置随机种子
rng = np.random.default_rng(42)

# 生成一个 chirp 样本
strain_cfg = DEFAULT.strain
bbh_cfg = DEFAULT.bbh
target_snr = 50.0
chirp_strain, t_merger, params = generate_bbh(strain_cfg, bbh_cfg, target_snr, rng)

print(f"Chirp strain shape: {chirp_strain.shape}, max: {chirp_strain.max():.2e}")

# 计算孤立 chirp 的 Q-scan（不加背景）
e_chirp, t_q, f_q = compute_qscan(chirp_strain, strain_cfg, DEFAULT.qscan)
print(f"Energy shape: {e_chirp.shape}, max: {e_chirp.max():.2f}, mean: {e_chirp.mean():.2f}")

# 转换为像素网格
grid = energy_to_pixel_grid(e_chirp, f_q, DEFAULT.image)
print(f"Pixel grid shape: {grid.shape}, max: {grid.max():.2f}, mean: {grid.mean():.2f}")

# 使用当前配置的阈值生成 mask
threshold = DEFAULT.qscan.chirp_mask_threshold
min_pixels = DEFAULT.qscan.min_blob_pixels
mask = mask_from_energy(grid, threshold, min_pixels)
print(f"Mask has {np.sum(mask)} True pixels (threshold={threshold}, min_pixels={min_pixels})")

# 检查 mask 连通分量
from skimage import measure
labelled = measure.label(mask, connectivity=2)
regions = measure.regionprops(labelled)
print(f"Number of connected components: {len(regions)}")
for i, reg in enumerate(regions):
    print(f"  Region {i}: area={reg.area}, bbox={reg.bbox}")

# 生成多边形
polys = mask_to_polygons(mask, DEFAULT.image, min_pixels)
print(f"Number of polygons generated: {len(polys)}")
if polys:
    print(f"First polygon has {len(polys[0])} points")

# 保存图像以便查看
plt.figure(figsize=(8, 6))
plt.imshow(grid, origin='lower', aspect='auto', cmap='viridis', vmin=0, vmax=25)
plt.colorbar(label='Normalized Energy')
plt.title('Chirp Q-scan (pixel grid)')
plt.savefig('debug_chirp_grid.png', dpi=150, bbox_inches='tight')
print("Saved debug_chirp_grid.png")

# 也保存 mask 图像
plt.figure(figsize=(8, 6))
plt.imshow(mask, origin='lower', aspect='auto', cmap='gray')
plt.title('Binary Mask')
plt.savefig('debug_chirp_mask.png', dpi=150, bbox_inches='tight')
print("Saved debug_chirp_mask.png")

# 打印阈值信息
if threshold < 0:
    percentile = -threshold
    thresh_val = np.percentile(grid, percentile)
    print(f"Percentile threshold: keep top {100-percentile:.1f}% (value > {thresh_val:.3f})")
else:
    print(f"Fixed threshold: energy > {threshold}")
