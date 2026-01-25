# gda/modules 示例命令

以下命令假设工作目录在仓库根目录（即包含 `gda/`、`Depth-Anything-3/`、`sam3/` 的目录），并使用仓库自带的示例图 `images/test.jpg`。

运行方式二选一：

- 方式 A：先激活环境再运行：`conda activate dinov3`
- 方式 B：不激活环境，直接前缀：`conda run -n dinov3 ...`

## 1) 深度估计（Depth-Anything-3）

- 保存 `depth.npy` + 默认彩色化 `depth.png`（`output_dir` 不指定时默认输出到图片所在目录）：

```bash
conda run -n dinov3 python -m gda.modules.depth_estimation --image images/test.jpg
```

- 指定输出目录：

```bash
conda run -n dinov3 python -m gda.modules.depth_estimation --image images/test.jpg --output_dir outputs/depth
```

## 2) 目标检测（GroundingDINO, Transformers）

- `--prompts` 支持用 `,` 或 `;` 分隔多个类别（示例：桥、车辆、人）。
  - 若你通过 `conda run -n dinov3 ...` 运行，建议使用 `,`，避免 `;` 在某些 shell/conda 版本下被当作命令分隔符。

```bash
conda run -n dinov3 python -m gda.modules.object_detection \
  --image images/test.jpg \
  --prompts "bridge,car,person" \
  --output_dir outputs/det
```

输出：
- `outputs/det/detections.json`
- `outputs/det/detections_vis.png`

## 3) 目标分割（SAM2 默认 / SAM3 可选）

### 3.1 默认：SAM2 后端（使用 sdf_compute 内置权重）

- 先用检测生成 `detections.json`，再把 box 喂给分割（默认使用 `sdf_compute/thirdparty/grounded_sam_2/checkpoints/sam2.1_hiera_large.pt`）：

```bash
conda run -n dinov3 python -m gda.modules.object_segmentation \
  --image images/test.jpg \
  --detection_json outputs/det/detections.json \
  --backend sam2 \
  --output_dir outputs/seg
```

### 3.2 真实分割：SAM3 后端（需要 checkpoint）

```bash
conda run -n dinov3 python -m gda.modules.object_segmentation \
  --image images/test.jpg \
  --detection_json outputs/det/detections.json \
  --backend sam3 \
  --sam3_checkpoint /path/to/sam3_checkpoint.pt \
  --sam3_load_from_hf false \
  --output_dir outputs/seg_sam3
```

输出：
- `outputs/seg/masks.npz`（包含 `masks`/`boxes_xyxy`/`prompt_ids`/`scores`/`prompts`/`image_size`）
- `outputs/seg/masks_vis.png`
- `outputs/seg/masks_meta.json`

备注：
- SAM3 的 HuggingFace 权重仓库是 gated，若你没有权限/未登录，会拉取失败；此时请提供本地 checkpoint 或完成 HF 登录。

## 4) 主流程（深度 + 检测 + 分割）

- 输入一张图 + 提示词，输出深度图与分割结果（默认 SAM2，本地权重）：

```bash
conda run -n dinov3 python -m gda.gda \
  --image images/test.jpg \
  --prompts "cat,dog,car" \
  --output_dir outputs/gda \
  --device cpu
```

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
conda run -n dinov3 python -m gda.modules.position_representation \
  --depth_npy outputs/gda/depth.npy \
  --masks_npz outputs/gda/masks.npz \
  --output_dir outputs/post
```

输出：
- `outputs/post/positions.npz`
- `outputs/post/positions.json`

## 6) 后处理：点云生成（depth + K (+rgb/+masks) -> pointcloud）

- 你需要提供相机内参（像素单位）：`fx, fy, cx, cy`。

```bash
conda run -n dinov3 python -m gda.modules.pointcloud_generation \
  --depth_npy outputs/gda/depth.npy \
  --image images/test.jpg \
  --masks_npz outputs/gda/masks.npz \
  --positions_npz outputs/post/positions.npz \
  --fx 500 --fy 500 --cx 320 --cy 240 \
  --output_dir outputs/post
```

输出：
- `outputs/post/pointcloud.npz`
- `outputs/post/pointcloud.ply`（默认开启，便于快速查看）

备注：
- 如果 `depth.npy` 与 `image/masks` 分辨率不同，会自动 resize 到目标尺寸（优先 masks，其次 image，否则 depth）。

## 7) 后处理：点云可视化（Open3D GUI）

```bash
conda run -n dinov3 python -m gda.modules.pointcloud_visualization \
  --pointcloud_npz outputs/post/pointcloud.npz
```

备注：
- 这是 GUI 交互窗口；在无显示环境下需要 X11 forwarding 或本地桌面。
