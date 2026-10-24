# UniCaMo Noise Construction — Independent Implementation

An independent NumPy implementation of the noise construction described in [Track the Noise, Move the World: 3D-Grounded Motion-Consistent Noise for Controllable Video Generation (arXiv:2607.02798v1)](https://arxiv.org/abs/2607.02798v1).

**This is not the authors' official implementation.** It reproduces the explicitly described noise operations and documents every unresolved implementation choice. It does not include Wan, LoRA training, learned depth/tracking models, or a claim to reproduce the paper's generated videos or benchmark scores.

[中文使用说明](README_zh.md) · [逐项论文核对表 / Paper audit](docs/PAPER_AUDIT.md) · [实验结果 / Results](results/REPORT.md)

## Quick start

Python 3.9+; CPU only. NumPy and Pillow are the only runtime dependencies.

```sh
git clone https://github.com/gulucaptain/UniCaMo-Implementation.git
cd UniCaMo-Implementation
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python examples/run_synthetic.py
```

The experiment exports four 25-frame GIFs, PNG previews, raw floating-point NPZ files, and statistical reports into `results/`. It also runs 20 independent seeds for the occlusion scene, first-frame policy sensitivity checks, and sphere-only diagnostics. Running it replaces the supplied results with the same seeded experiment. Numerical and image-encoding details can differ across dependency versions; `results/manifest.json` records the execution environment and source hashes. Allow a few hundred MB of available memory.

For importing outside the checkout, install with `python -m pip install -e .`.

## What is implemented

1. Project world-space 3D point tracks with camera intrinsics/extrinsics, then scale to the noise grid.
2. Copy immutable first-frame Gaussian noise patches along the projected tracks; default radius R=2 gives 3×3 patches.
3. Resolve overlapping patches using the smallest positive camera-space depth.
4. Sample a shared Gaussian sphere texture using the ray/sphere and UV equations from Appendix A.3, with nearest-neighbor texture lookup.
5. Fill uncovered locations with sphere noise and apply a separately identified linear alpha-blending interpretation.

Camera parameters alone do not move object noise: object tracks change in world space and consequently change their projections even when the camera is fixed. These inputs are explicit; the synthetic demos do not infer motion from images.

## Synthetic results

Each main scene has 1,024 tracks, 25 time steps, 16 noise channels, a 128×192 image grid, and a 64×96 noise grid. The sphere texture is 512×768 (image dimensions ×4). Foreground counts and trajectories are test fixtures; this is not the paper's mask GUI or training data pipeline.

Panels: track patches → sphere samples → composed raw noise → final blended noise → ownership. Cyan/orange identify foreground groups; blue-gray identifies static background tracks; purple identifies untracked first-frame reference noise. Display clipping affects previews only.

| Object translation | Occlusion / depth-order swap |
|---|---|
| ![Object translation](results/object_translation.gif) | ![Occlusion](results/occlusion.gif) |

| Camera rotation | Camera translation |
|---|---|
| ![Camera rotation](results/camera_rotation.gif) | ![Camera translation](results/camera_translation.gif) |

![Noise statistics](results/statistics.png)

The recorded run passed 10 tests. Identical source IDs have exactly identical raw noise vectors. For the default linear mixture `0.8 * raw + 0.2 * iid`, theoretical marginal variance is **0.68**, not 1. The 20-seed occlusion experiment measures a mean variance of **0.68034** (between-seed standard deviation **0.00191**). Correlation is measured against source-ID reuse, rather than assuming that every space/time sample is independent.

The four NPZ files (about 28 MB each) contain noise, raw noise, sphere samples, the initial reference, source IDs, ownership, coverage, depth, projections, input tracks, camera parameters, visibility, and configuration. Load them without pickle:

```python
import numpy as np
with np.load('results/occlusion.npz', allow_pickle=False) as data:
    noise = data['noise']  # [T, H_lat, W_lat, C], float32
    tracks = data['tracks']  # [T, N, 3], world coordinates
```

## Using your own tracks and cameras

```python
import numpy as np
from unicamo_noise import construct_noise

with np.load('guidance.npz', allow_pickle=False) as guidance:
    result = construct_noise(
        guidance['tracks'],       # [T,N,3], world coordinates
        guidance['intrinsics'],   # [T,3,3], IMAGE pixel units
        guidance['w2c'],          # [T,4,4], world-to-camera
        visibility=guidance['visibility'],  # [T,N], boolean
        height=64, width=96, channels=16,
        image_height=128, image_width=192,
        depth_bound=8.,           # replace with YOUR first-frame maximum depth
        sphere_downsample='nearest',
        first_frame_policy='preserve_reference',
        degradation=.2, blend='linear', seed=42,
    )
np.save('noise.npy', result['noise'])
```

Use right/down/forward camera axes and rigid extrinsics. All geometry must share a coordinate system and depth scale. Choose image/grid sizes to match your input; the values above are demo settings and do not define Wan's VAE compression. Visibility defaults to true when omitted; the module does not compute dense surface visibility.

## Paper fidelity and unresolved choices

Read [PAPER_AUDIT.md](docs/PAPER_AUDIT.md) for the section-by-section mapping to §3.1–3.3, §4.3, Appendix A, and Appendix B.3.

- Explicitly specified operations follow the paper: first-frame patch copying, depth selection, sphere center/radius, UV equations, nearest texture sampling, track-priority composition, R=2, k=4, and default degradation 0.2.
- **The exact degradation formula is unspecified.** This implementation uses ordinary linear alpha blending, without variance renormalization; this is an interpretation, not an author-confirmed formula. No square-root-weight variance-preserving improvement is silently applied.
- **Image-to-latent downsampling and first-frame composition are unspecified.** Callers must explicitly select the implemented nearest downsampling and either `preserve_reference` or `compose` first-frame policy. Both policies are examined in the recorded diagnostics.
- Reference and environment noise are independently sampled; source coupling is unspecified. Pixel rounding, patch boundary handling, visibility filtering, and equal-depth tie-breaking are documented implementation choices.
- Time steps map one-to-one to noise frames. VAE temporal compression is outside this module.

These tests establish consistency with the documented interpretation. They do not establish equivalence to the authors' code, improvement over Go-with-the-Flow, or downstream video quality.

## Repository layout

```text
unicamo_noise/core.py       Noise construction and geometric sampling
examples/run_synthetic.py  Deterministic demos, statistics, and reports
tests/                     Geometry, copying, visibility, and statistics tests
docs/PAPER_AUDIT.md         Explicit paper requirements versus interpretations
results/                   GIFs, PNGs, NPZ arrays, JSON metrics, and report
```

## Reference

Long Vu, Tan Ngo, Animesh Karnewar, Amir Habibian, Binh-Son Hua, Hung Bui, Minh Hoai Nguyen, and Phong Nguyen-Ha. *Track the Noise, Move the World: 3D-Grounded Motion-Consistent Noise for Controllable Video Generation*. arXiv:2607.02798v1, 2026. [Paper](https://arxiv.org/abs/2607.02798v1) · [Authors' project page](https://phongnhhn.info/Unicamo/).
