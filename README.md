# GDA: Grounded Depth Anything

GDA combines text-grounded instance segmentation, monocular depth estimation, and
3D position extraction. The default grounded-segmentation backend is native SAM3
text-to-mask inference. The previous GroundingDINO -> SAM2.1 box-to-mask chain is
kept as an explicit fallback.

The locked Pixi environments target Linux and Python 3.12. H200 uses CUDA 12.8
(`sm_90`); B300 uses CUDA 13.0 (`sm_103`) with the matching PyTorch `cu130`
wheels. GDA owns its required source dependencies as Git submodules:

```text
gda/
└── third_party/
    ├── depth-anything-3/
    ├── grounded-sam-2/
    └── sam3/
```

The parent repository records the default revisions as Git gitlinks. At this
revision they are:

- SAM3: `8f0b7f4d4e7eda2ed606ebde6702c93359ad01da`
- Depth-Anything-3: `2c21ea849ceec7b469a3e62ea0c0e270afc3281a`
- Grounded-SAM-2: `b7a9c29f196edff0eb54dbe14588d7ae5e3dde28`

Depth-Anything-3 itself contains a nested submodule, so always initialize
submodules recursively. Do not replace the recorded gitlinks with floating
branches when reproducing an environment.

## Installation

Clone the repository together with all recorded third-party sources:

```bash
git clone --recurse-submodules \
  https://github.com/OpenGHz/GroundedDepthAnything.git
cd GroundedDepthAnything
```

For an existing clone, or after changing revisions, synchronize and initialize
the same graph explicitly:

```bash
git submodule sync --recursive
git submodule update --init --recursive
python3 scripts/check-workspace.py
```

Install Pixi per user when it is not already available, then make its default
binary directory visible in the current shell:

```bash
command -v pixi >/dev/null || curl -fsSL https://pixi.sh/install.sh | bash
export PATH="$HOME/.pixi/bin:$PATH"
pixi --version
```

The installer normally updates shell startup files for new sessions. In an
already-open SSH session, exporting `PATH` as above takes effect immediately.

```bash
export GDA_PIXI_PLATFORM=h200
python3 scripts/check-setup-platform.py "$GDA_PIXI_PLATFORM"
pixi install --platform h200 --locked
pixi run --platform h200 --locked ensure-sam3
python3 scripts/ensure-sam2-checkpoint.py
pixi run --platform h200 --locked build-sam2
```

On `b300-2`, select the B300 target instead:

```bash
export GDA_PIXI_PLATFORM=b300
python3 scripts/check-setup-platform.py "$GDA_PIXI_PLATFORM"
CONDA_OVERRIDE_CUDA=13.0 CONDA_OVERRIDE_CUDA_ARCH=10.3 \
  pixi install --platform b300 --locked
pixi run --platform b300 --locked ensure-sam3
python3 scripts/ensure-sam2-checkpoint.py
pixi run --platform b300 --locked build-sam2
```

The SAM2.1 Hiera-L checkpoint is stored outside Git. Set `GDA_CACHE_DIR` to use
an explicit cache; the resulting path is
`$GDA_CACHE_DIR/checkpoints/sam2.1_hiera_large.pt`. Without it, the normal
default is `~/.cache/gda/checkpoints/sam2.1_hiera_large.pt` (or the equivalent
under `XDG_CACHE_HOME`). The downloader verifies the pinned SHA256 and refuses
to overwrite an unexpected file.

`pixi run build-sam2` copies the Grounded-SAM-2 submodule into
`.pixi/gda-build/`, applies GDA's CUDA-architecture patch there, and builds from
that isolated copy. The recorded submodule checkout remains clean.

Native SAM3 uses `facebook/sam3/sam3.pt`. The default provider is the public
ModelScope repository at the exact revision
`96f3e1b404ba14f2cfac60ee6ae87c269a7b7923`; it does not require a ModelScope
login. `pixi run ensure-sam3` downloads only `sam3.pt`, not the complete
repository containing both weight formats, and verifies its identity:

- size: `3450062241` bytes
- SHA256: `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`

With `GDA_CACHE_DIR` set, the content-addressed path is
`$GDA_CACHE_DIR/checkpoints/sam3/9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e/sam3.pt`.
Without it, GDA uses the corresponding path below `~/.cache/gda` (or
`XDG_CACHE_HOME`). Default inference also ensures the same pinned file lazily.

Hugging Face remains an explicit alternative. Request access, authenticate, and
pass `--sam3-load-from-hf` when selecting it:

```bash
pixi run --platform "$GDA_PIXI_PLATFORM" --locked hf auth login
```

Both providers deliver the same verified checkpoint; changing the download
provider does not change its Meta SAM License. Review its use, redistribution,
and trade-control restrictions before deployment. Select the H200 or B300 Pixi
target to match the host GPU; each target pins its CUDA toolkit and PyTorch
wheel.

Model weights are not committed to Git. Downloads use exact model revisions
rather than moving branch names:

- SAM3, default ModelScope provider:
  `facebook/sam3@96f3e1b404ba14f2cfac60ee6ae87c269a7b7923`
- SAM3, optional Hugging Face provider:
  `facebook/sam3@3c879f39826c281e95690f02c7821c4de09afae7`
- Depth-Anything-3: `depth-anything/DA3-LARGE@c54c26b16ec04d218e8d584ecf4bce082a9fcc20`
- GroundingDINO: `IDEA-Research/grounding-dino-base@12bdfa3120f3e7ec7b434d90674b3396eccf88eb`

The default `DA3-LARGE` weights are CC BY-NC 4.0 even though the DA3 source code
is Apache-2.0; commercial deployments need a suitably licensed model or separate
permission. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

The source gitlinks and model revisions are separate reproducibility pins: the
former select implementation code, while the latter select downloaded model
artifacts.

H200 and B300 resolve into the same mutable Pixi prefix, so one checkout is
bound to one GPU target. Use separate recursive checkouts for the two targets;
the setup scripts reject switching an existing `.pixi` prefix.

For a resumable setup (large CUDA wheels are cached step by step), run the
platform-specific manifest:

```bash
ai run scripts/setup-h200.sh  # or setup-b300.sh on B300
```

若未安装 `anchored-install`，可以直接执行自动选择 GPU 的 dispatcher：

```bash
bash scripts/setup-gpu.sh
```

The wheel and source distribution do not bundle submodule sources or model
weights. The source distribution is useful for packaging the Python wrapper, but
full model execution requires a recursive GDA Git checkout and the locked Pixi
workspace.

## Quick Start

Run native SAM3 text-to-mask segmentation (the default):

```bash
pixi run --platform "$GDA_PIXI_PLATFORM" --locked segment \
  --image third_party/sam3/assets/images/truck.jpg \
  --prompts "truck" \
  --output-dir outputs/seg_sam3
```

To require the pinned ModelScope file to be present without network access, add
`--sam3-local-files-only`. To use an explicit local SAM3 image checkpoint, add
`--sam3-checkpoint /path/to/sam3.pt`; a local path bypasses both providers and
the official-artifact hash check so intentionally customized checkpoints remain usable.

To select the optional Hugging Face provider instead, add
`--sam3-load-from-hf`. Its default revision remains pinned independently from
the ModelScope revision.

Run the GroundingDINO -> SAM2.1 fallback:

```bash
pixi run --platform "$GDA_PIXI_PLATFORM" --locked build-sam2
pixi run --platform "$GDA_PIXI_PLATFORM" --locked segment \
  --image third_party/sam3/assets/images/truck.jpg \
  --prompts "truck" \
  --backend sam2 \
  --output-dir outputs/seg_sam2
```

Run depth estimation together with grounded segmentation:

```bash
pixi run --platform "$GDA_PIXI_PLATFORM" --locked pipeline \
  --image third_party/sam3/assets/images/truck.jpg \
  --prompts "truck" \
  --output-dir outputs/gda
```

See [`docs/example_cmd.md`](docs/example_cmd.md) for standalone commands and
[`docs/grounded_segmentation_survey.md`](docs/grounded_segmentation_survey.md)
for the method survey, alternatives, limitations, and B300 benchmark. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for upstream repositories and
their checked-in license files.
