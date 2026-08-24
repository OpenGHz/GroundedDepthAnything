# gda/modules 示例命令

以下命令假设工作目录是 `gda/`（即包含 `pixi.toml` 的目录），其父目录同时包含
`Depth-Anything-3/`、`sam3/`、`sdf_compute/` 和示例图 `images/test.jpg`。

首次使用先安装锁定环境：

```bash
export PATH="$HOME/.pixi/bin:$PATH"
pixi --version
# 已打开的 SSH shell 也可以执行：source ~/.bashrc
python3 scripts/check-workspace.py
pixi install --platform h200 --locked
pixi run build-sam2
```

`scripts/setup-gpu.sh` 会下载并校验公开的 SAM2.1 Hiera-L checkpoint；手动安装时
可先运行 `python3 scripts/ensure-sam2-checkpoint.py`。

本文统一使用 `pixi run`，无需手动激活环境。锁文件包含 Linux H200
（CUDA 12.8、`sm_90`）和 B300（CUDA 13.0、`sm_103`）两个目标。

## 1) 深度估计（Depth-Anything-3）

- 保存 `depth.npy` + 默认彩色化 `depth.png`（`output_dir` 不指定时默认输出到图片所在目录）：

```bash
pixi run gda-depth --image ../images/test.jpg
```

- 指定输出目录：

```bash
pixi run gda-depth \
  --image ../images/test.jpg \
  --output_dir outputs/depth
```

## 2) 目标检测（GroundingDINO, Transformers）

- `--prompts` 支持用 `,`、`;` 或换行分隔多个类别。提示词应整体加引号；示例使用
  `,`，避免 shell 将未转义的 `;` 当作命令分隔符。

```bash
pixi run gda-detect \
  --image ../images/test.jpg \
  --prompts "bridge,car,person" \
  --output_dir outputs/det
```

输出：

- `outputs/det/detections.json`
- `outputs/det/detections_vis.png`

## 3) 文本 Grounding 分割

### 3.1 默认：原生 SAM3 text-to-mask

默认后端是 `sam3`。它直接接受文本提示并输出实例 mask，不需要先运行
GroundingDINO，也不会把检测框转换成点提示。多个文本提示共享同一次图像编码。

SAM3 图像权重来自 gated 仓库 `facebook/sam3`，默认 checkpoint 是 `sam3.pt`。
当前入口处理单张图像，因此不使用面向视频 Object Multiplex 的 SAM3.1 checkpoint。
SAM3 使用 Meta SAM License，部署前请核对其用途和再分发限制。
首次下载前需要先申请访问权限并登录：

```bash
pixi run hf auth login
```

运行：

```bash
pixi run segment \
  --image ../images/test.jpg \
  --prompts "bridge,car,person" \
  --output-dir outputs/seg_sam3
```

若已有本地 checkpoint，可绕过 Hugging Face 下载：

```bash
pixi run segment \
  --image ../images/test.jpg \
  --prompts "bridge,car,person" \
  --sam3-checkpoint /path/to/sam3.pt \
  --output-dir outputs/seg_sam3
```

输出：

- `outputs/seg_sam3/detections.json`
- `outputs/seg_sam3/detections_vis.png`
- `outputs/seg_sam3/masks.npz`（跨 prompt 去重时包含 `prompt_matches`）
- `outputs/seg_sam3/masks_vis.png`
- `outputs/seg_sam3/masks_meta.json`

### 3.2 Fallback：GroundingDINO -> SAM2.1

需要复现旧链路或对照基线时，显式选择 `sam2`。该后端先用 GroundingDINO
生成检测框，再把 box 直接交给 SAM2.1 生成 mask。

首次使用先编译仓库内的 SAM2 CUDA 扩展；构建脚本会检测当前 GPU，在 H200
使用 `sm_90`、在 B300 使用 `sm_103`，构建失败会直接报错：

```bash
pixi run build-sam2
```

然后运行统一的 grounded segmentation 入口：

```bash
pixi run segment \
  --image ../images/test.jpg \
  --prompts "bridge,car,person" \
  --backend sam2 \
  --output-dir outputs/seg_sam2
```

默认使用
`../sdf_compute/thirdparty/grounded_sam_2/checkpoints/sam2.1_hiera_large.pt`。

### 3.3 单独调试 SAM2.1 box-to-mask

先按第 2 节生成 `detections.json`，再单独运行 SAM2.1 box 分割入口：

```bash
pixi run gda-segment-boxes \
  --image ../images/test.jpg \
  --detection-json outputs/det/detections.json \
  --output-dir outputs/seg
```

输出：

- `outputs/seg/masks.npz`（包含 `masks`/`boxes_xyxy`/`prompt_ids`/`scores`/`prompts`/`image_size`）
- `outputs/seg/masks_vis.png`
- `outputs/seg/masks_meta.json`

## 4) 主流程（深度 + Grounding 分割）

- 输入一张图和提示词，输出深度图与 grounded segmentation 结果。默认使用原生 SAM3：

```bash
pixi run pipeline \
  --image ../images/test.jpg \
  --prompts "cat,dog,car" \
  --output-dir outputs/gda
```

如需使用 GroundingDINO -> SAM2.1 fallback，先执行 `pixi run build-sam2`，并在
上述命令中加入 `--seg-backend sam2`。

输出（默认文件名）：

- `outputs/gda/depth.npy`
- `outputs/gda/depth.png`
- `outputs/gda/detections.json`
- `outputs/gda/detections_vis.png`
- `outputs/gda/masks.npz`
- `outputs/gda/masks_vis.png`
- `outputs/gda/masks_meta.json`
- `outputs/gda/depth_with_masks.png`

## 5) 后处理：位置表示（mask -> 代表点）

- 基于 `depth.npy` + `masks.npz`，为每个 mask 选择一个代表像素点（mask 内深度中值附近的像素）：

```bash
pixi run python -m gda.modules.position_representation \
  --depth_npy outputs/gda/depth.npy \
  --masks_npz outputs/gda/masks.npz \
  --output_dir outputs/post
```

输出：

- `outputs/post/positions.npz`
- `outputs/post/positions.json`

## 6) 后处理：点云生成（depth + K (+rgb/+masks) -> pointcloud）

- 你需要提供相机内参（像素单位）：`fx, fy, cx, cy`。

也可以把内参写入一个 `K.json`（两种格式都支持：`{K:[[...]]}` 或 `{fx,fy,cx,cy}`）。例如：

```json
{
  "fx": 500.0,
  "fy": 500.0,
  "cx": 320.0,
  "cy": 240.0
}
```

```bash
pixi run python -m gda.modules.pointcloud_generation \
  --depth_npy outputs/gda/depth.npy \
  --image ../images/test.jpg \
  --masks_npz outputs/gda/masks.npz \
  --positions_npz outputs/post/positions.npz \
  --fx 500 --fy 500 --cx 320 --cy 240 \
  --output_dir outputs/post
```

- 或使用 `--k_file`：

```bash
pixi run python -m gda.modules.pointcloud_generation \
  --depth_npy outputs/gda/depth.npy \
  --image ../images/test.jpg \
  --masks_npz outputs/gda/masks.npz \
  --positions_npz outputs/post/positions.npz \
  --k_file K.json \
  --output_dir outputs/post
```

输出：

- `outputs/post/pointcloud.npz`
- `outputs/post/pointcloud.ply`（默认开启，便于快速查看）

备注：

- 如果 `depth.npy` 与 `image/masks` 分辨率不同，会自动 resize 到目标尺寸（优先 masks，其次 image，否则 depth）。

## 7) 后处理：点云可视化（Open3D GUI）

```bash
pixi run python -m gda.modules.pointcloud_visualization \
  --pointcloud_npz outputs/post/pointcloud.npz
```

备注：

- 这是 GUI 交互窗口；在无显示环境下需要 X11 forwarding 或本地桌面。

## 8) 一体化：image -> positions（可选 pointcloud / 可选可视化）

- 输入一张图 + 提示词，直接输出每个 mask 的代表点。

```bash
pixi run positions \
  --image ../images/test.jpg \
  --prompts "cat,dog,car" \
  --output-dir outputs/image_to_positions
```

输出（固定会写）：

- `outputs/image_to_positions/positions.npz`
- `outputs/image_to_positions/positions.json`

可选：只保留 positions（不保存中间产物）：

```bash
pixi run positions \
  --image ../images/test.jpg \
  --prompts "cat,dog,car" \
  --output-dir outputs/image_to_positions \
  --no-save-intermediate
```

可选：生成 pointcloud 并打开 GUI 可视化（需要 Open3D + 有显示环境）：

```bash
pixi run positions \
  --image ../images/test.jpg \
  --prompts "cat,dog,car" \
  --output-dir outputs/image_to_positions \
  --make-pointcloud \
  --k-file K.json \
  --visualize-pointcloud \
  --visualize-seconds 10
```

注：不设置 `--visualize-seconds` 时，Open3D 窗口会一直阻塞直到手动关闭。

输出（若启用 pointcloud）：

- `outputs/image_to_positions/pointcloud.npz`
- `outputs/image_to_positions/pointcloud.ply`（默认开启）
