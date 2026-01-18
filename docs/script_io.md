# sdf_compute 脚本输入/输出整理

本文整理 [sdf_compute](../sdf_compute) 工程当前提供的脚本（含 .py/.sh）的**输入（参数、必需文件/目录）**与**输出（生成的文件/目录结构）**，便于工程接入与重构。

> 约定：下文用
> - `INPUT_DIR` 表示命令行的 `--input_dir`
> - `OUTPUT_DIR` 表示命令行的 `--output_dir`
> - `OUTPUT_BASE` 表示批处理脚本里的输出根目录

---

## 1. 总体流程（从 Bridge 数据集到 SDF）

1) [brdige_dataset_process_depth.py](../sdf_compute/brdige_dataset_process_depth.py)
- 输入：Bridge 数据集 episode 目录（含 `images0/images1/...` 或 `image0/image1/...`，内部为 `im_XXXX.jpg`）
- 输出：每个 stream 生成 `rgb.mp4`，并在同目录生成每帧的分割结果 `frame_XXXX.npz`（内部含 `annotated_frame_index` 等）以及可视化 GIF。

2) [process_sdf.py](../sdf_compute/process_sdf.py)
- 输入：上一步某个 episode/stream 的输出目录（至少包含 `rgb.mp4`，并包含每帧的 `frame_XXXX.npz` 或 `frame_XXXX/annotated_frame_index.npy`）
- 输出：在 `OUTPUT_DIR` 下生成两套结果：`raw/` 与 `filtered/`，包含深度图、SDF（npy+png）、gif/mp4、点云等。

3) 批处理/测试脚本
- [test_single_case.py](../sdf_compute/test_single_case.py) 把步骤 1+2 串起来，便于快速验证
- [batch_process_full.sh](../sdf_compute/batch_process_full.sh) / [batch_process_stepwise.sh](../sdf_compute/batch_process_stepwise.sh) 用于多 episode 的批量处理

---

## 2. brdige_dataset_process_depth.py

脚本：[sdf_compute/brdige_dataset_process_depth.py](../sdf_compute/brdige_dataset_process_depth.py)

### 2.1 入口参数

- `--input_dir`（必需）：Bridge 数据集输入目录，包含 episode 目录（目录名为纯数字字符串，例 `00000`）
- `--output_dir`（必需）：输出根目录
- `--max_videos`：最多处理多少个 episode（不传表示全部）
- `--extract_frame_idx`：用于生成 caption 的抽帧索引（默认 0）
- `--device`：推理设备（默认 `cuda:0`）
- `--sam2_checkpoint`：SAM2 权重路径（可选；未提供时使用脚本内默认路径）
- `--sam2_model_cfg`：SAM2 配置（可选；未提供时使用脚本内默认配置名）

### 2.2 主要输入（目录/文件约定）

`INPUT_DIR/{episode_id}/` 下：

- 至少存在一个 stream 目录：`images{sid}/` 或 `image{sid}/`（优先使用 `images{sid}`）
- stream 目录内要求图片命名为：`im_XXXX.jpg`（例如 `im_0000.jpg`）
- 其他非 image/images 开头的子目录/文件会在 Step 0 被原样复制到输出 episode 目录（便于保留元信息）

### 2.3 输出（生成的目录/文件）

**A) 按 episode/stream 的主要输出：**

- `OUTPUT_DIR/{episode_id}/images{sid}/rgb.mp4`
  - 由 `im_XXXX.jpg` 合成（默认 fps=30）

- `OUTPUT_DIR/{episode_id}/images{sid}/frame_XXXX.npz`
  - 每帧一个压缩 npz，来自 Step 3（GroundingDINO + SAM2）
  - 关键字段（按代码写入）：
    - `masks`：布尔 mask，形状约为 `[N, H, W]`
    - `track_labels`：本次跟踪到的类别文本（OBJECTS）
    - `object_ids`：局部对象 id（从 1 开始）
    - `label_ids`：全局 label id（uint8）
    - `sample_id`：形如 `{episode_id}_s{sid}`
  - Step 4 会在同一个 `frame_XXXX.npz` 里**回写**：
    - `annotated_frame_color`：彩色可视化图（H,W,3）
    - `annotated_frame_index`：每像素的 label id（uint8）

- `OUTPUT_DIR/{episode_id}/images{sid}/result.gif`
  - Step 3 生成，叠加 mask 与 label 的可视化

- `OUTPUT_DIR/{episode_id}/images{sid}/result2.gif`
  - Step 4 生成（概率性写出），主要用于查看 `annotated_frame_*` 后处理结果

**B) 全局 caption/label 相关输出：**

- `OUTPUT_DIR/captions/rank_0.jsonl`
  - Step 1 生成：对每个 sample（episode+stream）抽帧，调用 Qwen-VL 生成 `raw_labels`

- `OUTPUT_DIR/raw_labels.txt`
  - Step 2 生成：聚合后的原始 label 列表（每行一个）

- `OUTPUT_DIR/labels.txt`
  - Step 2 生成：聚类后的“top labels”（每行一个 label 字符串）

- `OUTPUT_DIR/label_clusters.jsonl`
  - Step 2 生成：聚类映射关系

- `OUTPUT_DIR/all_captions.jsonl`
  - Step 2 生成：每个 sample 的 `track_labels`、`label_ids` 等结构化结果

### 2.4 作为下游（process_sdf.py）的输入时要点

- 下游通常需要：
  - `rgb.mp4`
  - 每帧对应的 `frame_XXXX.npz` 中的 `annotated_frame_index`
- 注意：本脚本生成的 `OUTPUT_DIR/labels.txt` 是“每行一个 label 文本”，而 [process_sdf.py](../sdf_compute/process_sdf.py) 的 `load_object_labels()` 当前仅识别形如 `id: name` 的行（否则会忽略）。
  - 这不会阻止 SDF 计算，但会导致“对象名称”显示为空。

---

## 3. process_sdf.py

脚本：[sdf_compute/process_sdf.py](../sdf_compute/process_sdf.py)

### 3.1 入口参数

- `--input_dir`（必需）：输入目录（要求包含 `rgb.mp4`，以及分割/标注信息）
- `--output_dir`（必需）：输出目录
- `--model_name`：Depth-Anything-3 模型名（默认 `depth-anything/DA3-LARGE`）
- `--device`：设备（默认：有 CUDA 则 `cuda` 否则 `cpu`）
- `--max_frames`：最多处理帧数（不传表示全部）
- `--track_pixels`：要追踪的像素坐标字符串，支持 `x1,y1;x2,y2` 或空格分隔
- `--video_fps`：输出视频帧率（默认 10.0）

### 3.2 主要输入（目录/文件约定）

`INPUT_DIR/` 下：

- 必需：`rgb.mp4`
  - 用于抽帧（保存为临时 PNG），并用于与深度/SDF同步处理

- 必需（至少一种）：每帧的对象 id/label id 图（用于 SDF 计算）
  - 优先读取：`INPUT_DIR/frame_XXXX/annotated_frame_index.npy`
  - 若不存在则尝试：`INPUT_DIR/frame_XXXX.npz` 内的 `annotated_frame_index`

- 可选：`labels.txt`
  - 当前代码仅解析 `"<int> : <name>"` 形式的映射（例如 `3: mug`）

### 3.3 输出（生成的目录/文件）

`OUTPUT_DIR/` 下会生成两套模式：`raw/` 与 `filtered/`。

此外，脚本在开始阶段会先把视频抽帧到：
- `OUTPUT_DIR/frames/frame_XXXX.png`（临时/中间产物；不会自动删除）

**每个模式（raw 或 filtered）下：**

- `OUTPUT_DIR/{mode}/depths/depth_XXXX.png`：深度图可视化
- `OUTPUT_DIR/{mode}/frames/frame_XXXX.png`：RGB 帧拷贝

- `OUTPUT_DIR/{mode}/sdf_npy/sdf_XXXX.npy`：原始 SDF 数值
- `OUTPUT_DIR/{mode}/sdf_vis/sdf_XXXX.png`：原始 SDF 可视化

- `OUTPUT_DIR/{mode}/sdf_exp_npy/sdf_exp_XXXX.npy`：指数变换后的 SDF 数值
- `OUTPUT_DIR/{mode}/sdf_exp_vis/sdf_exp_XXXX.png`：指数变换后的 SDF 可视化

- `OUTPUT_DIR/{mode}/depth.gif`、`depth_video.mp4`
- `OUTPUT_DIR/{mode}/sdf_vis.gif`、`sdf_video.mp4`
- `OUTPUT_DIR/{mode}/sdf_exp_vis.gif`、`sdf_exp_video.mp4`
- `OUTPUT_DIR/{mode}/rgb.gif`、`rgb_video.mp4`

- 若启用 `--track_pixels`：
  - `OUTPUT_DIR/{mode}/pixel_details/pixel_details_XXXX.json`：每帧被追踪像素的详细计算信息
  - `OUTPUT_DIR/{mode}/pixel_details/pixel_details_summary.json`：所有帧汇总
  - `OUTPUT_DIR/{mode}/rgb_marked/rgb_marked_XXXX.png`：在 RGB 上标记追踪像素/最近点等
  - `OUTPUT_DIR/{mode}/rgb_marked.gif`、`rgb_marked_video.mp4`

- `OUTPUT_DIR/{mode}/pointcloud.ply`
  - 将所有帧点云合并后导出（当前不写颜色）

---

## 4. test_single_case.py

脚本：[sdf_compute/test_single_case.py](../sdf_compute/test_single_case.py)

### 4.1 入口参数（对外统一入口）

- `--input_dir`（必需）：Bridge 数据集目录（含 episode 子目录）
- `--episode_id`（必需）：要测试的 episode（例如 `00000`）
- `--stream_id`：默认 0
- `--output_base`：默认 `output/test`
- `--device`：默认 `cuda:0`
- `--max_frames`：默认 10
- `--skip_dataset_process`：跳过 bridge 数据集处理
- `--skip_sdf`：跳过 SDF 计算
- `--track_pixels`：透传给 process_sdf.py

### 4.2 输入与输出

- 输入：
  - `INPUT_DIR/{episode_id}/images{stream_id}/im_XXXX.jpg`（或 `image{stream_id}`）

- 输出：
  - 数据集处理输出：`OUTPUT_BASE/dataset_processed/{episode_id}/images{stream_id}/...`
    - 来自 brdige_dataset_process_depth.py 的输出
  - SDF 输出：`OUTPUT_BASE/sdf_processed/{episode_id}/...`
    - 来自 process_sdf.py 的输出

- 额外动作：
  - 若 `OUTPUT_BASE/dataset_processed/labels.txt` 存在，会复制到 `.../images{stream_id}/labels.txt`，以满足 process_sdf.py 的“同目录查找 labels.txt”的约定。

---

## 5. batch_process_full.sh

脚本：[sdf_compute/batch_process_full.sh](../sdf_compute/batch_process_full.sh)

### 5.1 作用

- 对多个 episode 循环调用 [test_single_case.py](../sdf_compute/test_single_case.py)
- 支持串行/并行（`PARALLEL_JOBS`）

### 5.2 输入与输出

- 输入：
  - 依赖脚本内配置：`INPUT_DIR`、`EPISODE_IDS`、`STREAM_ID`、`DEVICE`、`MAX_FRAMES`、`TRACK_PIXELS`

- 输出：
  - `OUTPUT_BASE/dataset_processed/`（由 test_single_case -> brdige_dataset_process_depth.py 产生）
  - `OUTPUT_BASE/sdf_processed/`（由 test_single_case -> process_sdf.py 产生）

---

## 6. batch_process_stepwise.sh

脚本：[sdf_compute/batch_process_stepwise.sh](../sdf_compute/batch_process_stepwise.sh)

### 6.1 作用

- Step1：一次性批量运行 brdige_dataset_process_depth.py（处理全部 episode）
- Step2：逐个 episode 运行 process_sdf.py
- 支持跳过步骤：`SKIP_STEP1` / `SKIP_STEP2`

### 6.2 输入与输出

- Step1 输入：`INPUT_DIR`（Bridge 数据集）
- Step1 输出：`OUTPUT_BASE/dataset_processed/`（同 brdige_dataset_process_depth.py 的输出结构）

- Step2 输入：`OUTPUT_BASE/dataset_processed/{episode_id}/images{stream_id}/rgb.mp4` + 分割 npz
- Step2 输出：`OUTPUT_BASE/sdf_processed/{episode_id}/raw` 与 `filtered`

---

## 7. install_dependencies.sh

脚本：[sdf_compute/install_dependencies.sh](../sdf_compute/install_dependencies.sh)

### 7.1 输入

- 当前目录需要是 `sdf_compute/`（脚本中使用相对路径）
- 需要网络访问：
  - `pip install -r requirements.txt`
  - `git clone https://github.com/IDEA-Research/Grounded-SAM-2.git`
  - `wget https://dl.fbaipublicfiles.com/.../sam2.1_hiera_large.pt`

### 7.2 输出

- 安装 Python 依赖（写入当前 Python 环境）
- 生成/更新：
  - `sdf_compute/thirdparty/grounded_sam_2/`（代码库）
  - `sdf_compute/thirdparty/grounded_sam_2/checkpoints/sam2.1_hiera_large.pt`（权重）
- 运行 [test_imports.py](../sdf_compute/test_imports.py) 做导入校验（只输出到 stdout）

---

## 8. test_imports.py

脚本：[sdf_compute/test_imports.py](../sdf_compute/test_imports.py)

- 输入：无（依赖当前 Python 环境与本地文件是否存在）
- 输出：stdout 打印各依赖导入是否成功，以及 `thirdparty/grounded_sam_2/checkpoints/sam2.1_hiera_large.pt` 是否存在

