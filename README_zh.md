<div align="center">

# UniCaMo · Noise in Motion

### 让噪声沿三维轨迹移动，让视角共享同一片噪声。

独立的 UniCaMo 噪声构造实现 · 合成场景 · 可追溯统计

[![English](https://img.shields.io/badge/English-64748B?style=flat-square&logo=googletranslate&logoColor=white)](README.md) [![简体中文](https://img.shields.io/badge/%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-2563EB?style=flat-square&logo=googletranslate&logoColor=white)](README_zh.md) [![Paper](https://img.shields.io/badge/Paper-B31B1B?style=flat-square&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2607.02798v1) [![Paper Audit](https://img.shields.io/badge/Paper%20Audit-475569?style=flat-square&logo=readthedocs&logoColor=white)](docs/PAPER_AUDIT.md) [![Results](https://img.shields.io/badge/Results-0F766E?style=flat-square&logo=chartdotjs&logoColor=white)](results/REPORT.md)

</div>

---

用可检查的随机源，观察**三维运动如何成为噪声中的时间相关性**。本项目实现 UniCaMo 的独立噪声构造，并提供四组合成场景、完整浮点数据和统计验证。

> **项目范围：** 非官方的论文描述实现，聚焦噪声模块。论文未明确的细节逐项记录在[核对表](docs/PAPER_AUDIT.md)中；Wan 接入与 LoRA 训练不在本仓库范围内。

[**查看动画 ↓**](#运动可视化) · [**开始运行 ↓**](#快速开始) · [**接入自己的数据 ↓**](#接入自己的轨迹与相机)

## 运动可视化

每组动画从左到右依次呈现：

**轨迹噪声 → 球面采样 → Raw 合成 → 最终混合噪声 → 所有权**

青色与橙色表示两组前景，蓝灰色表示背景轨迹，紫色表示首帧未被轨迹覆盖的参考源。动画使用固定灰度范围；统计直接使用原始浮点值。

#### 物体平移

![物体平移](results/object_translation.gif)

#### 遮挡与前后关系交换

![遮挡与前后关系交换](results/occlusion.gif)

<details>
<summary><b>相机旋转</b></summary>

![相机旋转](results/camera_rotation.gif)

</details>

<details>
<summary><b>相机平移</b></summary>

![相机平移](results/camera_translation.gif)

</details>
## 快速开始

**Python 3.9+ · CPU 可运行 · 仅依赖 NumPy 与 Pillow**

**1. 配置环境**

```sh
git clone https://github.com/gulucaptain/UniCaMo-Implementation.git
cd UniCaMo-Implementation
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

**2. 运行测试与实验**

```sh
python -m unittest discover -s tests -v
python examples/run_synthetic.py
```

脚本会生成四组主场景、20 个独立随机种子的遮挡实验、首帧策略对照和纯球面诊断。输出覆盖 `results/` 中的同名文件，运行需预留数百 MB 内存。

<details>
<summary><b>输出文件与环境说明</b></summary>

| 文件 | 内容 |
| :--- | :--- |
| `*.gif` / `*.png` | 动画与静态预览 |
| `*.npz` | 完整噪声、几何、来源与配置，每场景约 28 MB |
| `statistics.json` | 分布矩、空间与时间相关性、敏感性对照 |
| `ensemble_samples.json` | 20 个种子的逐次测量 |
| `validation.json` | 来源一致性与复制误差检查 |
| `manifest.json` | 运行环境与源码哈希 |
| `REPORT.md` | 实验报告 |

随机种子固定；不同依赖版本可能影响浮点细节和图片编码，环境会写入 manifest。若需在其他项目导入，执行 `python -m pip install -e .`。

</details>

## 噪声如何构造

| 阶段 | 操作 | 表达的信息 |
| :--- | :--- | :--- |
| **Track-the-Noise** | 投影三维轨迹，复制首帧不可变的 3×3 噪声块 | 局部运动 |
| **深度选择** | 重叠位置保留正相机深度最近的贡献 | 前后遮挡关系 |
| **Move-the-World** | 射线与共享高斯噪声球面求交并采样 | 跨视角噪声对应 |
| **合成** | 轨迹优先，球面填充未覆盖区 | 完整噪声序列 |
| **Degradation** | 采用有明确标注的线性混合解释 | 额外独立噪声 |

物体轨迹与相机参数分别输入。相机固定时，物体的世界坐标仍可改变，投影位置及噪声块随之移动；球面分支则保持静止。

## 统计验证

| 检查 | 已记录结果 |
| :--- | :--- |
| 几何、复制、可见性与混合 | **10 项测试通过** |
| 相同来源的 Raw 噪声 | **逐元素完全一致** |
| 最大复制误差 | **0** |
| 20 个遮挡种子的最终方差 | **0.68034 ± 0.00191**（种子间标准差） |

默认采用 `0.8 × raw + 0.2 × iid` 的线性混合解释，理论边缘方差为 **0.68**，不进行方差归一化。相关性与噪声来源复用的预测对照，不把全部时空样本视为独立样本。

| 场景 | Raw 方差 | 最终方差 | Raw 时间相关 | 来源预测 |
| :--- | ---: | ---: | ---: | ---: |
| 物体平移 | 1.0075 | 0.6846 | 0.9304 | 0.9302 |
| 前后遮挡 | 1.0088 | 0.6854 | 0.8856 | 0.8842 |
| 相机旋转 | 1.0048 | 0.6830 | 0.0002 | 0.0000 |
| 相机平移 | 1.0037 | 0.6824 | 0.1520 | 0.1516 |

![方差、分布与时间相关性](results/statistics.png)

<details>
<summary><b>合成场景配置与数据读取</b></summary>

| 配置 | 数值 |
| :--- | :--- |
| 序列 | 25 个噪声时刻，16 通道 |
| 引导 | 每组主场景 1,024 条轨迹 |
| 图像 / 噪声网格 | 128×192 / 64×96 |
| 球面纹理 | 512×768，即图像尺寸 ×4 |
| 主实验种子 | 42 |
| 几何 | 合成前景与静态背景平面 |

这些尺寸是实验设置，不代表 Wan 的 VAE 配置。物体点数和轨迹是测试输入，未运行论文的 GUI 或真实视频几何估计流程。

```python
import numpy as np

with np.load('results/occlusion.npz', allow_pickle=False) as data:
    noise = data['noise']    # [T, H_lat, W_lat, C], float32
    tracks = data['tracks']  # [T, N, 3]，世界坐标
```

NPZ 同时保存 Raw 噪声、球面采样、首帧源、来源编号、所有权、覆盖、深度、投影、相机参数、可见性及配置。

</details>

[**查看完整实验报告 →**](results/REPORT.md)

## 接入自己的轨迹与相机

```python
import numpy as np
from unicamo_noise import construct_noise

with np.load('guidance.npz', allow_pickle=False) as g:
    result = construct_noise(
        g['tracks'], g['intrinsics'], g['w2c'],
        visibility=g['visibility'],
        height=64, width=96, channels=16,
        image_height=128, image_width=192,
        depth_bound=8.,  # 换成你的首帧最大深度，单位与轨迹一致
        sphere_downsample='nearest',
        first_frame_policy='preserve_reference',
        degradation=.2, blend='linear', seed=42,
    )
np.save('noise.npy', result['noise'])
```

| 输入 | 形状与含义 |
|---|---|
| tracks | `[T,N,3]`，世界坐标点轨迹 |
| intrinsics | `[T,3,3]`，图像像素单位的 K |
| w2c | `[T,4,4]`，世界到相机的刚体变换 |
| visibility | `[T,N]`，布尔可见性，可省略（默认全 true） |

相机轴为右、下、前。输出 raw/noise/sphere 均为 `[T,H_lat,W_lat,C]`，float32。模块不估计轨迹、深度或稠密可见性。图像和网格尺寸是调用方选择，不代表 Wan 的 VAE 配置。

其他输出包括 source（原始噪声来源编号）、owner（轨迹编号；-1 球面，-2 首帧未覆盖参考源）、track_coverage、depth、reference、投影和 config_json。NPZ 无需 pickle 即可加载。

## 论文实现核对

[逐项核对表](docs/PAPER_AUDIT.md)对应正文和附录，区分论文明确要求与实现时必须补充的约定。

<details>
<summary><b>展开未解决的细节与复现边界</b></summary>

论文明确部分严格按描述实现；具体对应见 [核对表](docs/PAPER_AUDIT.md)。未写清楚的部分均作标注：

1. alpha blending 没有精确公式。本实现采用普通线性解释，不偷偷做方差保持改进或归一化。
2. 图像到 latent 的降采样算子未定义，调用方需显式选择 nearest。
3. 首帧合成未充分定义。默认示例保留 A.2 的完整首帧参考源，并跑 compose 策略敏感性对照。
4. 首帧和球面随机源是否耦合、像素取整、边界及等深度处理为解释性选择。
5. 输入时刻与噪声帧一一对应；不处理 VAE 时间压缩。

测试通过表示代码符合这些明确约定，不能证明与作者代码逐值一致，也不能证明优于 Go-with-the-Flow 或得到论文级视频效果。此前探索版本保留于原 CameraNoise 工作区，此仓库仅发布论文核对版 v2。

</details>

## 项目结构

```text
unicamo_noise/core.py       噪声构造与几何采样
examples/run_synthetic.py  合成场景、可视化和统计
tests/                     几何与噪声统计测试
docs/PAPER_AUDIT.md         正文与附录逐项核对
results/                   动画、浮点数据与结果报告
```

## 引用

```bibtex
@misc{vu2026tracknoise,
  title         = {Track the Noise, Move the World: 3D-Grounded Motion-Consistent Noise for Controllable Video Generation},
  author        = {Long Vu and Tan Ngo and Animesh Karnewar and Amir Habibian and Binh-Son Hua and Hung Bui and Minh Hoai Nguyen and Phong Nguyen-Ha},
  year          = {2026},
  eprint        = {2607.02798},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2607.02798v1}
}
```
