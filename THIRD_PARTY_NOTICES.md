# Third-Party Notices

GDA references the following upstream source repositories through Git submodules.
The parent repository records exact gitlink commits. GDA currently has no
project-wide license file; this notice documents third-party terms but does not
grant permission to use, modify, or distribute GDA itself.

| Submodule | Upstream | License reported by the submodule root `LICENSE` |
| --- | --- | --- |
| `third_party/sam3` | [facebookresearch/sam3](https://github.com/facebookresearch/sam3) | [SAM License](https://github.com/facebookresearch/sam3/blob/8f0b7f4d4e7eda2ed606ebde6702c93359ad01da/LICENSE) |
| `third_party/depth-anything-3` | [ByteDance-Seed/Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3) | [Apache License 2.0](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/2c21ea849ceec7b469a3e62ea0c0e270afc3281a/LICENSE) |
| `third_party/depth-anything-3/da3_streaming/loop_utils/salad` | [serizba/salad](https://github.com/serizba/salad) | [GNU GPL v3](https://github.com/serizba/salad/blob/6aede13a3f6c25750bf7fde10209c06cb73060bb/LICENSE) |
| `third_party/grounded-sam-2` | [IDEA-Research/Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) | [Apache License 2.0](https://github.com/IDEA-Research/Grounded-SAM-2/blob/b7a9c29f196edff0eb54dbe14588d7ae5e3dde28/LICENSE) |

The linked files above are authoritative for the checked-out revisions. Each
submodule may contain nested projects, datasets, assets, or model artifacts with
additional terms; review the relevant files in that checkout before use or
redistribution. In particular, the SAM3 root license is not an Apache/MIT license
and includes use, redistribution, publication, and trade-control conditions.

Model weights are not committed to GDA and may use terms different from their
source code:

- Access to the default `facebook/sam3` checkpoint is gated by Hugging Face and
  governed by the SAM License.
- The default `depth-anything/DA3-LARGE` model is listed by its upstream model
  table as **CC BY-NC 4.0**. It is not licensed for commercial use under the
  source repository's Apache-2.0 license.
- GroundingDINO and SAM2.1 artifacts remain governed by their respective
  upstream model terms.
