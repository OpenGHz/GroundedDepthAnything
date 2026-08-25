# GDA 命令与脚本输入/输出整理

本文整理当前 GDA checkout 提供的安装脚本、CLI 和稳定产物。所有命令均假设当前
目录是项目根目录。第三方源码位于 `third_party/` Git submodules；模型权重位于
可写 cache，均不复制到源码树或提交到 Git。

## 1. 复现所需输入

### 1.1 源码

GDA 的默认源码组合由父仓库的 gitlinks 固定：

- `third_party/sam3/`
- `third_party/depth-anything-3/`
- `third_party/grounded-sam-2/`

Depth-Anything-3 还包含嵌套 submodule，因此 fresh clone 和补初始化都必须递归：

```bash
git clone --recurse-submodules \
  https://github.com/OpenGHz/GroundedDepthAnything.git
cd GroundedDepthAnything

# 已有 clone 使用：
git submodule sync --recursive
git submodule update --init --recursive
```

`scripts/check-workspace.py` 从父仓库 index 读取 gitlink，而不是依赖浮动分支名；它会
检查直接和嵌套 submodule 是否初始化、HEAD 是否匹配以及 worktree 是否干净。

### 1.2 模型权重

权重不进入 Git。默认来源与定位规则如下：

- SAM3：默认从公开的 ModelScope `facebook/sam3` 单独下载 `sam3.pt`，固定 revision
  `96f3e1b404ba14f2cfac60ee6ae87c269a7b7923`。设置 `GDA_CACHE_DIR` 时写入
  `$GDA_CACHE_DIR/checkpoints/sam3/9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e/sam3.pt`；
  未设置时写入 GDA/XDG 默认 cache。文件固定为 `3450062241` bytes，SHA256 为
  `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`。
  `--sam3-load-from-hf` 显式选择 gated Hugging Face provider；
  `--sam3-local-files-only` 禁止所选 provider 访问网络；`--sam3-checkpoint` 则完全
  绕过 provider。
- SAM2.1 Hiera-L：由 `scripts/ensure-sam2-checkpoint.py` 从固定公开 URL 下载并校验
  SHA256。设置 `GDA_CACHE_DIR` 时写入
  `$GDA_CACHE_DIR/checkpoints/sam2.1_hiera_large.pt`；未设置时通常写入
  `~/.cache/gda/checkpoints/sam2.1_hiera_large.pt`。
- Depth-Anything-3 与 GroundingDINO：由 Hugging Face cache 管理。

默认 `depth-anything/DA3-LARGE` 权重由上游标注为 CC BY-NC 4.0，不能因为 DA3
源码使用 Apache-2.0 就推断权重可商用；部署前应核对
`THIRD_PARTY_NOTICES.md` 和对应模型卡。

默认模型 revision 都固定为 commit，而不是移动分支：

- SAM3 ModelScope（默认）：`96f3e1b404ba14f2cfac60ee6ae87c269a7b7923`
- SAM3 Hugging Face（可选）：`3c879f39826c281e95690f02c7821c4de09afae7`
- Depth-Anything-3：`c54c26b16ec04d218e8d584ecf4bce082a9fcc20`
- GroundingDINO：`12bdfa3120f3e7ec7b434d90674b3396eccf88eb`

SAM3 两个 provider 的 revision 空间彼此独立，但下载的 `sam3.pt` 还必须通过同一
size/SHA256 校验。项目已有固定 SAM3 源码 submodule，因此默认不下载包含两套权重和
仓库文件的完整 ModelScope snapshot。源码 gitlink 与模型 revision 是两组独立 pin，
缺少任意一组都不能完整复现。

## 2. 环境与维护脚本

### 2.1 `scripts/setup-gpu.sh`

输入：

- 已递归初始化且保持记录版本的 submodules
- 可用的 Pixi
- H200 或 B300 NVIDIA GPU
- 模型下载所需的网络与访问权限
- 可选 `GDA_PIXI_PLATFORM=h200|b300`
- 可选 `GDA_CACHE_DIR=/path/to/cache`

动作与输出：

- 校验 submodule gitlinks
- 从 ModelScope 下载并校验默认 SAM3 checkpoint
- 下载并校验 SAM2 checkpoint
- 创建锁定的 `.pixi/` 环境
- 在 `.pixi/gda-build/` 隔离副本内 patch、编译并安装 SAM2 CUDA 扩展
- 运行 GPU doctor、单元测试、format-check 和 lint

直接运行 `bash scripts/setup-gpu.sh` 会自动选择 GPU 平台；若使用
`anchored-install`，请对固定平台的 manifest 执行
`ai run scripts/setup-h200.sh` 或 `ai run scripts/setup-b300.sh`，这样成功步骤
才能在失败重跑时复用，且 H200/B300 不会共用缓存。

### 2.2 `scripts/ensure-sam2-checkpoint.py`

输入：可选 `GDA_CACHE_DIR`，否则遵循 GDA/XDG 默认 cache。

输出：

- `checkpoints/sam2.1_hiera_large.pt`
- 仅在文件不存在且下载内容 SHA256 正确时完成安装
- 若目标已有错误 hash，脚本会报错而不会覆盖

### 2.3 `scripts/ensure-sam3-checkpoint.py`

输入：锁定 Pixi 环境，以及可选 `GDA_CACHE_DIR`。对应 Pixi task 是：

```bash
pixi run --platform "$GDA_PIXI_PLATFORM" --locked ensure-sam3
```

输出：GDA content-addressed cache 中的 `sam3.pt`。脚本只请求 ModelScope 单文件，
使用固定 revision，并在完成前校验文件大小和 SHA256；不会下载完整双权重仓库。

### 2.4 `scripts/build-sam2.py`

输入：

- `third_party/grounded-sam-2/` 的固定、干净 gitlink checkout
- `scripts/patches/grounded-sam2-cuda-arch.patch`
- 当前 Pixi target 的 CUDA toolchain
- 自动检测的 GPU capability，或显式 `GDA_CUDA_ARCH`

输出：SAM2 Python 包与 CUDA extension 安装到当前 Pixi 环境。源码复制、patch 和
编译发生在 `.pixi/gda-build/` 下，不修改 submodule；因此构建后
`git submodule status --recursive` 仍可用于验证源码状态。

### 2.5 `scripts/doctor.py`

输入：已安装的锁定环境、可用 NVIDIA GPU、已编译的 SAM2 extension。

输出：stdout 打印 PyTorch/CUDA/GPU capability、xFormers 和 SAM2 extension 路径；
目标 capability 或关键导入不匹配时失败。

## 3. 深度估计：`gda-depth`

必要输入：

- `--image`：单张图像

常用可选输入：

- `--output-dir`：默认是输入图所在目录
- `--model-name`：默认 `depth-anything/DA3-LARGE`
- `--model-revision`：默认固定 revision
- `--device`、`--colormap`
- `--save-npy`、`--save-png`

输出：

- `depth.npy`：`float32 [H, W]`
- `depth.png`：默认彩色深度可视化

## 4. 目标检测：`gda-detect`

必要输入：

- `--image`
- `--prompts`：以逗号、分号或换行分隔，整段应由 shell 引号包裹

常用可选输入：

- `--output-dir`：默认是输入图所在目录
- `--model-id`、`--model-revision`
- `--box-th`、`--text-th`、`--device`

GroundingDINO 对每条 prompt 独立检测，因此实例通过精确 `prompt_ids` 回到原始
prompt，不依赖 phrase substring 猜测。

输出：

- `detections.json`
  - `image_size: [H, W]`
  - `prompts: list[str]`
  - `boxes_xyxy: [N, 4]`
  - `scores: [N]`
  - `prompt_ids: [N]`
  - `labels: [N]`
- `detections_vis.png`

## 5. 文本 Grounding 分割：`gda-segment`

必要输入：

- `--image`
- `--prompts`

### 5.1 默认 SAM3

`--backend sam3` 直接从文本预测 instances/masks，一张图的多个 prompt 共享一次图像
编码。可选 `--sam3-checkpoint` 使用本地权重；否则默认按固定 revision 从
ModelScope 单文件获取。`--sam3-load-from-hf` 选择固定 HF revision，
`--sam3-local-files-only` 要求所选 provider 只读缓存。跨 prompt 的重复实例按 mask
IoU 去重，并在 `prompt_matches` 中保留全部匹配关系。

高级覆盖项 `--sam3-modelscope-revision` 只作用于默认 ModelScope provider；
`--sam3-model-revision` 只作用于显式 Hugging Face provider。两个 revision 不能互换，
无论覆盖哪个来源，最终文件仍须通过固定 size/SHA256 校验。

### 5.2 GroundingDINO → SAM2.1 fallback

`--backend sam2` 先产生 boxes，再将 box 直接交给 SAM2.1，不再在框内采点。使用前
需要执行 `pixi run build-sam2`，并确保 cache 中存在默认 checkpoint；也可用
`--sam2-checkpoint` 覆盖。

两种 backend 写出相同的稳定产物：

- `detections.json`
- `detections_vis.png`
- `masks.npz`
  - `masks: bool [N, H, W]`
  - `boxes_xyxy: float32 [N, 4]`
  - `prompt_ids: int32 [N]`
  - `scores: float32 [N]`
  - `prompts`
  - `image_size: int32 [2]`
  - SAM3 去重启用时还包含 `prompt_matches`
- `masks_meta.json`：backend 与上述数组的 JSON 元数据
- `masks_vis.png`

## 6. 深度与分割主流程：`gda`

必要输入：`--image` 与 `--prompts`。`--seg-backend` 默认为 `sam3`；需要 fallback 时
设为 `sam2`。`--output-dir` 未提供时写到输入图所在目录。

输出：

- `depth.npy`
- `depth.png`
- `detections.json`
- `detections_vis.png`
- `masks.npz`
- `masks_meta.json`
- `masks_vis.png`
- `depth_with_masks.png`

## 7. 图像到代表位置：`gda-positions`

必要输入：`--image` 与 `--prompts`。流程依次运行深度估计、grounded segmentation，
再为每个 mask 选择有效深度中值附近的代表像素。

固定输出：

- `positions.npz`：`rep_uvs [N,2]`、`rep_depths [N]`、`valids [N]` 与元数据
- `positions.json`：便于人工查看的等价表示

默认还会保存第 6 节的中间产物；`--no-save-intermediate` 可关闭。启用
`--make-pointcloud` 时还会输出：

- `pointcloud.npz`
- `pointcloud.ply`（默认开启）

点云需要通过 `--k-file`，或同时提供 `--fx --fy --cx --cy`。GUI 可视化由
`--visualize-pointcloud` 开启，在无显示服务器上不应启用。

## 8. 输出兼容约定

- Python API 图像统一为 RGB `uint8 [H, W, 3]`。
- 深度统一转换为 `float32 [H, W]`。
- masks 统一为 `bool [N, H, W]`。
- boxes 使用像素坐标 `xyxy`。
- `prompt_ids[i]` 指向 `prompts`；跨 prompt 去重后以 `prompt_matches[i]` 表示多重匹配。
- 深度与 mask 分辨率不一致时，位置与点云后处理负责显式对齐。
- 图像入口 CLI 的 `output_dir` 未设置时默认使用输入图所在目录；独立后处理入口
  分别使用其 masks/depth 输入所在目录。
