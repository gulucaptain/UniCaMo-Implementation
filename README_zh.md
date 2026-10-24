# UniCaMo 独立噪声构造复现

本项目根据 [UniCaMo v1 论文](https://arxiv.org/html/2607.02798v1) 实现独立的噪声构造模块，包含物体平移、前后遮挡、相机旋转和平移四组实验及噪声统计。**这是非官方的论文描述实现**，不包含 Wan、LoRA 训练或生成视频效果复现。

[英文首页与动画预览](README.md) · [正文及附录逐项核对](docs/PAPER_AUDIT.md) · [结果报告](results/REPORT.md)

## 安装与运行

Python 3.9 及以上，仅需 NumPy 和 Pillow，可在 CPU 上运行。

```sh
git clone https://github.com/gulucaptain/UniCaMo-Implementation.git
cd UniCaMo-Implementation
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python examples/run_synthetic.py
```

运行后重新生成 results/ 下的全部动画、静态图、浮点 NPZ、统计 JSON 和报告，覆盖同名结果。固定随机种子；环境与源码哈希保存在 manifest.json。若需在其他项目中导入，执行 `python -m pip install -e .`。

## 原理

- Track-the-Noise：三维点经相机投影到噪声网格，沿轨迹复制首帧不可变的 3×3 噪声块；重叠时选相机深度最近者。
- Move-the-World：以首帧相机为球心，为整段视频共享一张球面高斯纹理，按每帧射线求交取样。
- 合成：轨迹噪声优先，未覆盖区域用球面噪声填充，再进行有明确标注的线性混合。

相机固定时物体仍可移动：输入的物体三维坐标随时间改变，投影位置也随之改变；球面分支保持静止。

## 输入与调用

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

## 结果与验证

四组主场景各有 1024 条轨迹、25 帧、16 通道；图像 128×192，噪声网格 64×96，球面纹理 512×768。背景为静态平面，物体轨迹为合成测试输入；没有复现论文的 GUI 和真实视频几何估计。

10 项测试通过；同源 raw 噪声逐元素一致，复制误差为 0。默认线性混合 `0.8 raw + 0.2 iid` 的理论方差为 0.68；20 个独立种子的遮挡场景平均方差为 0.68034，种子间标准差为 0.00191。

动画左到右为：轨迹、球面、raw 合成、混合后最终噪声、所有权。固定 [-3,3] 灰度显示，统计不截断。

- [物体平移](results/object_translation.gif)
- [前后遮挡](results/occlusion.gif)
- [相机旋转](results/camera_rotation.gif)
- [相机平移](results/camera_translation.gif)
- [完整统计](results/statistics.json)、[20 种子明细](results/ensemble_samples.json)、[验证记录](results/validation.json)

四个 NPZ 各约 28 MB，保留完整浮点结果和几何输入；可重跑生成。

## 对论文的忠实程度

论文明确部分严格按描述实现；具体对应见 [核对表](docs/PAPER_AUDIT.md)。未写清楚的部分均作标注：

1. alpha blending 没有精确公式。本实现采用普通线性解释，不偷偷做方差保持改进或归一化。
2. 图像到 latent 的降采样算子未定义，调用方需显式选择 nearest。
3. 首帧合成未充分定义。默认示例保留 A.2 的完整首帧参考源，并跑 compose 策略敏感性对照。
4. 首帧和球面随机源是否耦合、像素取整、边界及等深度处理为解释性选择。
5. 输入时刻与噪声帧一一对应；不处理 VAE 时间压缩。

测试通过表示代码符合这些明确约定，不能证明与作者代码逐值一致，也不能证明优于 Go-with-the-Flow 或得到论文级视频效果。此前探索版本保留于原 CameraNoise 工作区，此仓库仅发布论文核对版 v2。
