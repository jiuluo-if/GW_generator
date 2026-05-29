# GW-YOLO Synthetic Data Generator

参考论文：*Soni, Mukund, Katsavounidis, "GW-YOLO: Multi-transient segmentation in LIGO using computer vision"* (arXiv:2508.17399).

本仓库根据论文第 IV 节描述的数据合成流程，提供一套**完全可复现、可大规模并行**的训练数据生成管线，输出格式直接兼容 [Ultralytics YOLOv8 segmentation](https://docs.ultralytics.com/datasets/segment/)。

---

## 论文中训练数据是如何生成的（简版回顾）

| 步骤 | 来源 / 工具 | 备注 |
|---|---|---|
| 1. 采集 glitch 时序 | LIGO O3 GravitySpy 数据集 | 真实瞬态噪声 |
| 2. 生成 BBH chirp | PyCBC，`m_p, m_s > 5 M⊙` | 参数取自 O1–O3 真实事件 |
| 3. 生成 BNS chirp | PyCBC，`m_p, m_s < 2.5 M⊙` | 同上 |
| 4. 把 chirp 时序 + glitch 时序按**随机时间偏移**相加 | numpy | 3 秒窗口 |
| 5. 对合成时序做 Q-transform（Q-scan）→ 谱图 | GWpy / Omega | 训练图 |
| 6. 在谱图上标注 `"chirp"` 和 `"noise"` 的像素 mask | 论文里逐张标注 | 这里我们用「分别 Q-transform 独立成分 + 阈值化」**自动生成 mask** |
| 7. 80/10/10 切分 | — | train/val/test |

---

## 本仓库的设计

```
gw_yolo_data_gen/
├── config.py            # 所有可调参数（dataclass）
├── waveforms.py         # PyCBC 生成 BBH/BNS chirp
├── glitches.py          # 两种 glitch 来源：真实文件 / 7 类参数化模拟
├── noise.py             # aLIGO PSD 着色高斯背景噪声
├── spectrograms.py      # Q-transform + PNG 渲染（与 GravitySpy 同一通路）
├── annotations.py       # 像素 mask → YOLO seg polygon 标签
├── pipeline.py          # 串/并行 6 类场景的样本组装
├── generate_dataset.py  # CLI 入口
└── requirements.txt
```

### 6 类合成场景

| scene | 内容 | 用途 |
|---|---|---|
| `bbh` | 纯 BBH chirp + 背景 | 教会模型识别孤立 chirp |
| `bns` | 纯 BNS chirp + 背景 | 同上 |
| `bbh_glitch` | BBH chirp + glitch + 背景 | 模型核心训练目标 |
| `bns_glitch` | BNS chirp + glitch + 背景 | 模型核心训练目标 |
| `glitch` | 纯 glitch + 背景 | 学到「只有 noise」也要正确识别 |
| `background` | 纯背景 | 学到「什么都没有」（true negatives） |

### 自动标注的关键技巧

我们**不**对合成谱图人肉划框。流程是：

1. **独立** 计算 chirp-only 的 Q-scan，阈值化得到 chirp 像素 mask。
2. **独立** 计算 glitch-only 的 Q-scan，阈值化得到 noise 像素 mask。
3. 合成时序的 Q-scan 作为训练图。

这样标签和图像在同一坐标系上严格对齐，且天然反映「该成分在该像素上**是否真的可见**」—— SNR 太低看不见，标签也就为空，模型不会被迫记忆人眼都看不见的东西。

---

## 安装

```bash
# 建议使用 conda（PyCBC 依赖 LAL）
conda create -n gwyolo python=3.10 -y
conda activate gwyolo
conda install -c conda-forge lalsuite -y
pip install -r requirements.txt
```

---

## 使用

### 冒烟测试（200 张，单进程）
```bash
python generate_dataset.py --total 200 --workers 1 --out ./tiny_dataset
```

python generate_dataset.py --total 200 --workers 1 --glitch-dir /home/jiuluo/python_demo/new/npy --out ./tiny_dataset

### 大规模生成（3 万张，16 进程，模拟 glitch）
```bash
python generate_dataset.py --total 30000 --workers 16 \
    --glitch-source simulated --out ./gw_yolo_30k
```

### 用真实 GravitySpy glitch
先把 GravitySpy 的 glitch 时序导成 `.npy` 文件放进一个目录，然后：
```bash
python generate_dataset.py --total 30000 --workers 16 \
    --glitch-source gravityspy \
    --glitch-dir /path/to/gravityspy_timeseries \
    --out ./gw_yolo_real
```

### 直接训练 YOLOv8
```bash
pip install ultralytics
yolo segment train data=./gw_yolo_30k/data.yaml model=yolov8m-seg.pt \
     imgsz=640 epochs=100
```

---

## 调参指南

- **想要更多的 SNR-glitch 重叠样本** → 在 `config.py` 里把 `n_bbh_plus_glitch` / `n_bns_plus_glitch` 调高。
- **想生成更大的图** → `--image-size 1024`。
- **想更长的时间窗口** → `--duration 5.0`，模型能看到更长的 BNS inspiral。
- **mask 看起来太胖/太瘦** → 调 `config.QScanCfg.chirp_mask_threshold` 和 `noise_mask_threshold`。
- **想加入 NSBH 类** → `waveforms.py` 中加一个 `sample_nsbh_params`，并在 `pipeline.py` 的 scene plan 里加一个分支。

---

## 输出

```
gw_yolo_30k/
├── data.yaml                       # YOLO 数据集描述
├── metadata.json                   # 每张图的 SNR、质量、glitch 类等
├── images/
│   ├── train/ 0000000_bbh.png ...
│   ├── val/
│   └── test/
└── labels/
    ├── train/ 0000000_bbh.txt ...
    ├── val/
    └── test/
```

每个 `.txt` 是 YOLOv8-seg 格式：
```
<class_id> x1 y1 x2 y2 ... xn yn       # 全部归一化到 [0,1]，原点左上
```
`class_id`：`0=chirp`，`1=noise`。

---

## 与论文的对应关系

| 论文要求 | 本仓库实现 |
|---|---|
| Q-transform 谱图，3 秒窗口 | `spectrograms.compute_qscan` + `StrainCfg.duration=3.0` |
| BBH `m > 5 M⊙` / BNS `m < 2.5 M⊙` | `BBHCfg.m1_range` / `BNSCfg.m1_range` |
| chirp 参数取自 O1–O3 真实事件 | `BBHCfg` 默认覆盖 GWTC-3 的事件范围 |
| 真实 GravitySpy glitch | `--glitch-source gravityspy --glitch-dir ...` |
| 随机 chirp-glitch 时间偏移 | `waveforms.generate_cbc_strain` 内有 `np.roll` 抖动，glitch 也独立采样中心时间 |
| 80/10/10 划分 | `DatasetCfg.split` |
| 两类标签 `chirp` / `noise` | `config.CLASS_NAMES` |

---

## 注意事项

- `noise.py`、`waveforms.py` 在 import 时**不会**立刻 import PyCBC，所以代码静态审查不需要 PyCBC 在场。运行时才需要。
- 默认 PSD 是 aLIGO 设计灵敏度（`aLIGOaLIGODesignSensitivityT1800044`）。若想匹配论文里 O3 的实际灵敏度，把 `StrainCfg.detector_psd` 改成 `"aLIGOZeroDetHighPower"` 或者 `use_real_psd_if_available=True` 并自行提供 PSD 文件。
- 模拟 glitch 的形态被刻意调成「在 Q-scan 上和真实 GravitySpy 类相似」，但它**不是**物理模型。若做最终发表级训练，强烈建议使用真实 glitch 时序。
