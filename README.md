```markdown
# GW-YOLO Synthetic Data Generator

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

基于论文 [*GW-YOLO: Multi-transient segmentation in LIGO using computer vision*](https://arxiv.org/abs/2508.17399) 的 **增强版合成数据生成器**。

生成 **真实引力波信号 (Chirp) + 真实瞬态噪声 (Glitch) + 丰富背景噪声** 的 Q‑变换谱图，并自动输出 **YOLOv8‑seg 格式的像素级多边形标签**。

> ✨ **主要改进**：支持真实引力波事件（GWTC‑1～GWTC‑3）、每张图包含多个随机 Glitch、背景噪声模型更贴近 LIGO 真实环境、Chirp 可见性自动增强。

---

## 📦 安装

### 使用 Conda（推荐）
```bash
conda create -n gwyolo python=3.10 -y
conda activate gwyolo
conda install -c conda-forge lalsuite -y
pip install -r requirements.txt
```

### 使用 Pip + 系统 LALSuite
```bash
pip install -r requirements.txt
```

> **注意**：`pycbc` 和 `gwpy` 依赖 LALSuite，建议使用 Conda 环境。

---

## 🚀 快速开始

### 1. 准备真实 Chirp 数据（可选，但推荐）

下载引力波事件应变数据并筛选有效信号：
```bash
python download_all_gwtc.py                # 下载 GWTC-1~GWTC-3 事件（约 113 个文件）
python prepare_chirp_metadata.py           # 计算 SNR，自动过滤弱信号
```
生成的 `real_chirps/chirp_metadata.json` 将只保留 SNR ≥ 8 的文件。

> 如果跳过此步，管线将使用模拟 Chirp（BBH/BNS 随机参数）。

### 2. 准备真实 Glitch 数据（可选）

将 [Gravity Spy](https://www.gravityspy.org/) 的 glitch 时序数据（.npy 格式）按类别放入子文件夹，例如：
```
/path/to/glitch_data/
├── Blip/
├── Tomte/
├── Koi_Fish/
└── ...
```
管线会自动从子文件夹名读取类别标签。

如果未提供，则使用内置的 7 类参数化模拟 glitch。

### 3. 生成数据集

#### 冒烟测试（20 张，单进程，检查流程）
```bash
python generate_dataset.py --total 20 --workers 1 --out ./test_smoke
```

#### 使用真实 Glitch 生成 3 万张（16 进程）
```bash
python generate_dataset.py --total 30000 --workers 16 \
    --glitch-source gravityspy \
    --glitch-dir /path/to/glitch_data \
    --out ./gw_yolo_real
```

#### 使用模拟 Glitch（无需额外数据）
```bash
python generate_dataset.py --total 10000 --workers 8 \
    --glitch-source simulated \
    --out ./gw_yolo_simulated
```

#### 缩短时间窗口放大 Chirp（例如 1 秒）
```bash
python generate_dataset.py --total 5000 --duration 1.0 --out ./gw_yolo_1s
```

---

## 🧪 测试与验证

### 单元测试（关键函数）
```bash
python -m pytest tests/          # 若未编写，可手动运行示例脚本
```

### 可视化单样本（调试用）
```python
from config import DEFAULT
from pipeline import generate_sample
from glitches import build_glitch_source
import numpy as np

rng = np.random.default_rng(42)
glitch_source = build_glitch_source(DEFAULT.strain, DEFAULT.glitch)
sample = generate_sample("mixed", DEFAULT, glitch_source, rng)

# 保存图像和标签
from spectrograms import render_qscan_png
render_qscan_png(sample["energy_combined"], sample["times"], sample["freqs"],
                 "test.png", DEFAULT.image)
print("标签:", sample["isolated_grids"].keys())
```

### 检查生成数据集质量
```bash
# 统计标签文件数量
ls gw_yolo_real/labels/train/*.txt | wc -l
# 查看一个标签文件
head -3 gw_yolo_real/labels/train/0000000_mixed.txt
# 应同时包含 class 0 (chirp) 和 class 1 (noise) 的多边形
```

---

## 📁 输出文件结构

```
gw_yolo_dataset/
├── data.yaml                 # YOLO 数据集描述
├── metadata.json             # 每张图的详细参数（SNR、glitch类型等）
├── images/
│   ├── train/   *.png
│   ├── val/     *.png
│   └── test/    *.png
└── labels/
    ├── train/   *.txt
    ├── val/     *.txt
    └── test/    *.txt
```

### 标签格式（YOLOv8‑seg）
```
<class_id> x1 y1 x2 y2 ... xn yn
```
- `class_id = 0` → chirp（引力波信号）
- `class_id = 1` → noise（glitch 瞬态噪声）
- 所有坐标归一化到 `[0,1]`，原点在图像左上角。

---

## ⚙️ 配置与调优

所有参数集中在 `config.py` 的 dataclass 中，可灵活修改。

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `StrainCfg.duration` | 3.0 s | 时间窗口长度（秒），缩短可放大 chirp 在图像中的尺寸 |
| `StrainCfg.use_real_chirp` | `True` | 是否使用真实引力波事件（需先运行 `prepare_chirp_metadata.py`） |
| `GlitchCfg.num_glitches_range` | `(1,3)` | 每张图片包含的 glitch 数量范围 |
| `QScanCfg.chirp_mask_threshold` | `0.01` | Chirp mask 阈值（极低值确保标注完整） |
| `QScanCfg.noise_mask_threshold` | `6.0` | Glitch mask 阈值（可根据需要调高/调低） |
| `ImageCfg.width/height` | `640` | 输出图像尺寸（正方形） |
| `DatasetCfg.total_samples` | `10000` | 总样本数（可通过 `--total` 覆盖） |

### 背景噪声丰富度
在 `noise.py` 中，背景包含：
- 高斯色噪声（aLIGO 设计 PSD）
- 随机尖峰（泊松分布）
- 低频漂移（0.1‑5 Hz）
- 窄带谐振（20‑500 Hz）
- 短时突发脉冲

可通过调整各部分的幅度系数和概率来控制。

---

## 🧠 模型训练（YOLOv8）

```bash
pip install ultralytics
yolo segment train data=./gw_yolo_real/data.yaml \
    model=yolov8m-seg.pt \
    imgsz=640 \
    epochs=100 \
    batch=16
```

训练后评估：
```bash
yolo segment val data=./gw_yolo_real/data.yaml model=runs/segment/train/weights/best.pt
```

---

## 🤝 贡献

欢迎提交 Issue 或 Pull Request。请确保代码符合 [Black](https://github.com/psf/black) 格式化规范。

---

## 📄 许可证

[MIT License](LICENSE)

---

## 🙏 致谢

- [GWOSC](https://www.gwosc.org/) – 引力波开放数据
- [Gravity Spy](https://www.gravityspy.org/) – Glitch 分类数据集
- [PyCBC](https://pycbc.org/) & [GWpy](https://gwpy.github.io/) – 波形生成与 Q‑变换
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) – 分割模型训练框架
```
