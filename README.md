# GDA: Grounded Depth Anything

GDA combines text-grounded instance segmentation, monocular depth estimation, and
3D position extraction. The default grounded-segmentation backend is native SAM3
text-to-mask inference. The previous GroundingDINO -> SAM2.1 box-to-mask chain is
kept as an explicit fallback.

The locked Pixi environments target Linux and Python 3.12. H200 uses CUDA 12.8
(`sm_90`); B300 uses CUDA 13.0 (`sm_103`) with the matching PyTorch `cu130`
wheels. The workspace is expected to contain these sibling repositories:

```text
eia/
├── Depth-Anything-3/
├── gda/
├── sam3/
└── sdf_compute/thirdparty/grounded_sam_2/
```

At runtime, `GDA_WORKSPACE_ROOT` may point the model loaders at an equivalent
sibling checkout, but Pixi's local path dependencies still require the documented
relative layout during `pixi install`.

## Installation

If Pixi was installed under the default per-user location, make it visible in
the current shell (new shells normally load this automatically):

```bash
export PATH="$HOME/.pixi/bin:$PATH"
pixi --version
```

For an already-open SSH session, run `source ~/.bashrc` after the first setup;
the B300 profile keeps this PATH entry for future sessions.

```bash
cd gda
# Optional: verify the sibling repository revisions used by pixi.toml.
python3 scripts/check-workspace.py
pixi install --platform h200 --locked
python3 scripts/ensure-sam2-checkpoint.py
pixi run build-sam2
```

On `b300-2`, select the B300 target instead:

```bash
CONDA_OVERRIDE_CUDA=13.0 CONDA_OVERRIDE_CUDA_ARCH=10.3 \
  pixi install --platform b300 --locked
```

Native SAM3 uses the gated `facebook/sam3` image checkpoint (`sam3.pt`). Request
access first, then authenticate when the checkpoint is not already cached:

```bash
pixi run hf auth login
```

SAM3 is distributed under Meta's SAM License rather than Apache/MIT; review its
use, redistribution, and trade-control restrictions before deployment. Select
the H200 or B300 Pixi target to match the host GPU; each target pins its CUDA
toolkit and PyTorch wheel.

For a resumable setup (large CUDA wheels are cached step by step), run:

```bash
ai run scripts/setup-gpu.sh
```

若未安装 `anchored-install`，也可以直接执行同一份脚本：

```bash
bash scripts/setup-gpu.sh
```

The wheel is a source-distribution artifact; model execution is intentionally
supported through this Pixi workspace because SAM3 and Depth-Anything-3 are
pinned sibling checkouts (and SAM2 is a locally patched CUDA extension).

## Quick Start

Run native SAM3 text-to-mask segmentation (the default):

```bash
pixi run segment \
  --image ../images/test.jpg \
  --prompts "bridge,car,person" \
  --output-dir outputs/seg_sam3
```

To use a local SAM3 image checkpoint, add
`--sam3-checkpoint /path/to/sam3.pt`. A local checkpoint bypasses the Hugging Face
download.

Run the GroundingDINO -> SAM2.1 fallback:

```bash
pixi run build-sam2
pixi run segment \
  --image ../images/test.jpg \
  --prompts "bridge,car,person" \
  --backend sam2 \
  --output-dir outputs/seg_sam2
```

Run depth estimation together with grounded segmentation:

```bash
pixi run pipeline \
  --image ../images/test.jpg \
  --prompts "bridge,car,person" \
  --output-dir outputs/gda
```

See [`docs/example_cmd.md`](docs/example_cmd.md) for standalone commands and
[`docs/grounded_segmentation_survey.md`](docs/grounded_segmentation_survey.md)
for the method survey, alternatives, limitations, and B300 benchmark.
