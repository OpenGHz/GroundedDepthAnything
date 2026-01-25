# 模块化重构思路（基于 sdf_compute 现有实现）

本文面向“单张图片 + 提示词”的流程，将 sdf_compute 现有能力拆成可复用模块（每个模块：一个类 + 一个 main 脚本，放在同一 .py 文件中）。重点说明：复用哪些现有库/代码、核心参数、核心方法的输入输出数据类型。

---

## 0. 对模块划分的建议（在你给出的方案基础上小幅优化）

你的 4 模块划分（深度估计 / 目标检测 / 目标分割 / 主流程）是合理的，建议做两点“接口层面”的增强，原因是后续扩展/替换模型会更平滑：

1) **显式定义共享数据结构（推荐）**
- 检测结果与分割结果是“跨模块传递”的核心对象，建议用 `dataclass`（或 dict 约定）统一字段，避免后续脚本之间靠隐式约定拼接。
- 这不是额外模块，只是建议在每个文件里复用同一结构（或单独放一个 types.py 再 import）。

2) **把“提示词”标准化成“多目标描述列表”**
- 你希望“多目标，每个目标通过关联提示词区分”，建议 CLI 支持 `--prompts` 传入多条（例如用 `;` 分隔），内部统一转成 `List[str]`。
- 对 GroundingDINO 这类模型，通常需要把文本拼成一个 prompt（如 `"mug. bottle."`），但输出仍要能映射回每个子 prompt。

如果你认可以上两点，你原方案不需要改动模块数量，只需在文档/代码里把 I/O 结构定清楚。

### 0.1 统一约束（按你新增的要求固化）

1) **模块类初始化只允许一个参数 `config`**
- 形式：`__init__(self, config: XxxConfig)`
- `XxxConfig` 必须 `继承 pydantic.BaseModel`
- 其他初始化参数（model 名称、阈值、device、权重路径等）全部放进 config

2) **脚本命令行参数使用 `pydantic_settings` 实现**
- 形式：定义 `class CLIArgs(BaseSettings): ...`，从 CLI 读取参数并做校验/默认值
- 建议 `SettingsConfigDict(cli_parse_args=True)`（或项目中你偏好的等价配置）

3) **输出目录可不指定：默认与输入目录相同**
- 单图任务里“输入目录”指输入图片文件所在目录：`Path(image).parent`
- 约定：若 `--output_dir` 为空/未传，则 `output_dir = Path(image).parent`

4) **深度图片默认彩色化输出**
- `depth.png` 默认使用 colormap（例如 OpenCV `COLORMAP_TURBO`/`COLORMAP_INFERNO`）以便人眼查看
- 同时仍建议保存 `depth.npy`（float32 原始深度），便于下游计算

---

## 1. 建议的共享数据类型（跨模块传递）

### 1.1 DetectionResult（检测输出）

建议字段（Python `dataclass` 或 JSON dict）：

- `image_size: tuple[int, int]`：`(H, W)`
- `prompts: list[str]`：原始提示词列表（多目标）
- `boxes_xyxy: numpy.ndarray`：形状 `[N, 4]`，float32，像素坐标，格式 `x1,y1,x2,y2`
- `scores: numpy.ndarray`：形状 `[N]`，float32
- `prompt_ids: numpy.ndarray`：形状 `[N]`，int32，指向 `prompts` 的索引（用于“每个目标关联提示词”）
- `labels: list[str] | None`：可选，人类可读标签（有时与 prompts 相同）

序列化建议：
- `json`：便于脚本互相调用（boxes/scores/prompt_ids 用 list 存）
- 或 `npz`：便于保存 numpy（推荐 `np.savez_compressed`）

### 1.2 SegmentationResult（分割输出）

建议字段：

- `image_size: tuple[int, int]`：`(H, W)`
- `masks: numpy.ndarray`：形状 `[N, H, W]`，bool 或 uint8
- `boxes_xyxy: numpy.ndarray`：沿用检测 boxes（便于可视化/追溯）
- `scores: numpy.ndarray | None`：可选（SAM2 可能输出 mask 置信度/IoU 估计）
- `prompt_ids: numpy.ndarray`：沿用检测 prompt_ids

序列化建议：
- `npz`：`masks`+`boxes`+`prompt_ids` 最方便

---

## 2. 深度估计模块

### 2.1 复用现有实现

来源：sdf_compute 的深度推理在 process_sdf.py 里已经使用 Depth-Anything-3：
- `from depth_anything_3.api import DepthAnything3`
- `model = DepthAnything3.from_pretrained(model_name)`
- `prediction = model.inference([image_path], export_format=..., export_dir=None)`

### 2.2 类接口建议

类名示例：`DepthEstimatorDA3`

Config（必须继承 `pydantic.BaseModel`，并且类初始化仅接收该 config）：
- `model_name: str = "depth-anything/DA3-LARGE"`
- `device: str = "cuda"`（或自动探测后的默认值）
- `colorize_colormap: str = "turbo"`（用于默认彩色化导出）

核心方法：

- `predict(image: numpy.ndarray | str) -> numpy.ndarray`
  - 输入：
    - `image`：图片路径（str）或 BGR/RGB numpy 图像
  - 输出：
    - `depth: numpy.ndarray`，shape `[H, W]`，float32

可选扩展方法（非必须，但实用）：
- `colorize(depth) -> numpy.ndarray`：输出 `uint8` 彩色图（应用 colormap，默认用于保存 `depth.png`）

### 2.3 CLI 脚本约定

输入参数建议（用 `pydantic_settings` 实现）：
- `--image`：图片路径
- `--output_dir`：输出目录（可选；不传则与输入图片目录相同）
- `--model_name`、`--device`

输出建议：
- `depth.npy`：float32 深度
- `depth.png`：默认彩色化可视化（便于人看）

---

## 3. 目标检测模块（bbox）

### 3.1 复用现有实现

sdf_compute 已在 brdige_dataset_process_depth.py 里使用 Transformers 版本的 GroundingDINO：
- `AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")`
- `AutoModelForZeroShotObjectDetection.from_pretrained(...)`
- `processor.post_process_grounded_object_detection(...)`（注意不同 transformers 版本参数名差异）

这里可以直接复用同样的模型与 postprocess 兼容逻辑，但将输入从“视频首帧”改为“单张图片”。

### 3.2 类接口建议

类名示例：`GroundingDinoDetector`

Config（继承 `pydantic.BaseModel`）：
- `model_id: str = "IDEA-Research/grounding-dino-base"`
- `box_threshold: float = 0.25`
- `text_threshold: float = 0.3`
- `device: str = "cuda:0"`

核心方法：

- `detect(image: numpy.ndarray | str, prompts: list[str]) -> DetectionResult`
  - 输入：
    - `image`：路径或 numpy 图像
    - `prompts`：多目标提示词列表（例如 `["mug", "bottle"]`）
  - 输出：
    - `DetectionResult`（见 1.1），其中 `prompt_ids` 需要把每个 box 对应回哪条 prompt

提示词拼接建议：
- 统一把 prompts 拼成 `"p1. p2. ..."`（GroundingDINO 常见格式）
- postprocess 得到的 `labels` 通常是文本片段，建议用“字符串相等/归一化”或“包含关系”映射回 prompts

### 3.3 CLI 脚本约定

输入参数建议（用 `pydantic_settings` 实现）：
- `--image`
- `--prompts`：例如 `"mug; bottle; robot arm"`
- `--output_dir`：可选；不传则与输入图片目录相同
- `--box_th`、`--text_th`、`--device`

输出建议：
- `detections.json`（或 `detections.npz`）：包含 boxes/scores/prompt_ids/image_size/prompts
- `detections_vis.png`：把 bbox 画在图上（可用 opencv 或 supervision）

---

## 4. 目标分割模块（mask）

### 4.1 复用现有实现

sdf_compute 已使用 SAM2（来自 `thirdparty/grounded_sam_2`）：
- `build_sam2(...)`
- `SAM2ImagePredictor`
- 以及视频版 `build_sam2_video_predictor`（单图不需要 propagate）

单图分割建议使用 `SAM2ImagePredictor`：
- 输入：image + boxes
- 输出：每个 box 对应的 mask（多目标）

### 4.2 类接口建议

类名示例：`Sam2Segmentor`

Config（继承 `pydantic.BaseModel`）：
- `sam2_checkpoint: str`（默认指向 `thirdparty/grounded_sam_2/checkpoints/sam2.1_hiera_large.pt`）
- `sam2_model_cfg: str = "configs/sam2.1/sam2.1_hiera_l"`
- `device: str = "cuda:0"`

核心方法：

- `segment(image: numpy.ndarray | str, det: DetectionResult) -> SegmentationResult`
  - 输入：
    - `det.boxes_xyxy`（N 个 box）
  - 输出：
    - `SegmentationResult.masks`：`[N, H, W]`

可选扩展：
- `overlay(image, masks, ...) -> numpy.ndarray`：生成彩色叠加可视化（可用 supervision 的 `MaskAnnotator`）

### 4.3 CLI 脚本约定

输入参数建议（用 `pydantic_settings` 实现）：
- `--image`
- `--detection_json`（或 `--detection_npz`）：检测结果路径
- `--output_dir`：可选；不传则与输入图片目录相同

输出建议：
- `masks.npz`：保存 `masks` + `boxes_xyxy` + `prompt_ids` + `prompts` + `image_size`
- `masks_vis.png`：mask 叠加可视化

---

## 5. 主流程模块（单图：提示词 -> 深度 + 分割）

### 5.1 复用现有实现

- 深度：沿用 Depth-Anything-3 推理
- 检测：GroundingDINO
- 分割：SAM2

三者在 sdf_compute 里都已出现：
- Depth-Anything-3：process_sdf.py
- GroundingDINO+SAM2：brdige_dataset_process_depth.py

### 5.2 类接口建议

类名示例：`ImageDepthAndSegPipeline`

Config（继承 `pydantic.BaseModel`）：
- `depth: DepthEstimatorConfig`
- `det: DetectorConfig`
- `seg: SegmentorConfig`
- `output_depth_colormap: str = "turbo"`（主流程导出用）

核心参数（通过 config 构造内部三个模块）：
- `depth_estimator: DepthEstimatorDA3`
- `detector: GroundingDinoDetector`
- `segmentor: Sam2Segmentor`

核心方法：

- `run(image: numpy.ndarray | str, prompts: list[str]) -> tuple[numpy.ndarray, SegmentationResult]`
  - 输出：`(depth, seg)`

可选：输出“带分割结果的深度图”
- 方式 A：在深度可视化图上叠加 mask（更直观）
- 方式 B：导出“masked depth”（每个目标一个 depth ROI）

### 5.3 CLI 脚本约定

输入参数建议（用 `pydantic_settings` 实现）：
- `--image`
- `--prompts`（`;` 分隔）
- `--output_dir`：可选；不传则与输入图片目录相同
- `--device`（统一设备；或分别给 depth/det/seg 配置）

输出建议：
- `depth.npy`、`depth.png`（默认彩色化）
- `detections.json`（可选，便于调试）
- `masks.npz`、`masks_vis.png`
- `depth_with_masks.png`：深度彩色化可视化 + mask 叠加

---

## 6. 工程实现备注（避免踩坑）

- 设备字符串建议统一：
  - Depth-Anything-3 常用 `cuda`/`cpu`
  - Transformers/SAM2 常用 `cuda:0`/`cpu`
  - 主流程可以接受 `--device cuda:0`，内部派生出 `depth_device="cuda"`（或直接 `torch.device` 兼容）

- 输出目录默认策略（按约束）：
  - `output_dir = Path(image).parent`（当 `--output_dir` 未提供时）

- 颜色空间：
  - opencv 读图是 BGR；Transformers Processor 通常吃 RGB/PIL。
  - 建议检测/分割入口都接受路径，内部统一用 PIL 走一遍，减少颜色错误。

- 输出格式稳定性：
  - CLI 输出尽量固定文件名（例如 `depth.npy`、`masks.npz`），方便下游脚本串联。

---

## 7. 后处理模块（位置表示 / 点云生成 / 点云可视化）

你提出的 3 个后处理模块方向是合理的；我建议在接口上补充两点约束，这样后续扩展会更稳：

1) **深度/Mask/图像分辨率必须对齐**
- Depth-Anything-3 的深度输出可能不是原图分辨率（我们在主流程里也遇到过）。
- 因此后处理模块要明确：输入的 `depth` 与 `masks` 是否同一分辨率；若不同，需要在进入模块前对齐（推荐：把深度 resize 到 mask/原图大小，并记录缩放信息）。

2) **相机内参数据结构要固定**
- 推荐统一用 `K: np.ndarray`，shape `[3, 3]`，float64/float32。
- 同时明确坐标系约定（OpenCV 常用）：
  - 像素坐标 `(u, v)` 对应 `x=u, y=v`
  - 相机坐标：`X = (u - cx)/fx * Z`，`Y = (v - cy)/fy * Z`，`Z = depth`

下面给出各模块的实现思路（按 prompts/prepare.md 约束：类仅接收一个 `config: BaseModel`，脚本 CLI 用 `pydantic_settings`）。

### 7.1 位置表示模块（Mask -> 单点代表）

**目标**：在每个 mask 的深度区域中，选择“深度中值”代表该区域的唯一空间表示。

建议的“深度中值点”定义（更可复现且鲁棒）：
- 在 mask 内取所有有效深度值集合 $D$（过滤 `nan/inf/<=0`，必要时再加深度上下界）。
- 计算中值 $m = \mathrm{median}(D)$。
- 在 mask 内找到深度值最接近 $m$ 的像素点 $(u,v)$ 作为代表点。
- 若存在多个并列点，建议用“离 mask 质心最近”作为 tie-break（避免挑到边缘）。

Config（继承 `pydantic.BaseModel`）建议字段：
- `min_depth: float = 1e-6`
- `max_depth: float | None = None`
- `erode_kernel: int = 0`（可选，>0 时对 mask 做一次腐蚀后再取点，减少边缘噪声）

类接口建议：

- 类名：`MaskPositionRepresentor`
- 核心方法：
  - `represent(depth: np.ndarray, masks: np.ndarray, prompt_ids: np.ndarray | None = None) -> dict`
    - 输入：
      - `depth`: `np.ndarray`，shape `[H, W]`，float32/float64
      - `masks`: `np.ndarray`，shape `[N, H, W]`，bool/uint8
      - `prompt_ids`（可选）：shape `[N]`，int32
    - 输出（JSON 可序列化 dict，或 npz）：
      - `rep_uv: np.ndarray`，shape `[N, 2]`，int32，(u, v)
      - `rep_depth: np.ndarray`，shape `[N]`，float32
      - `valid: np.ndarray`，shape `[N]`，bool（mask 内无有效深度时为 False，`rep_uv=-1`）
      - `prompt_ids` 原样透传（可选）

CLI 脚本约定（建议新文件）：
- `--depth_npy`（或 `--depth`）
- `--masks_npz`
- `--output_dir`（可选；默认同输入）

输出建议：
- `positions.json` 或 `positions.npz`

### 7.2 点云生成模块（Depth/RGB/K -> PointCloud）

**目标**：从深度图生成点云；可选按 RGB 上色；可选按 masks 过滤/打标签；可选把“位置表示点”映射到点云索引。

实现思路（推荐库与方式）：
- 纯 numpy 也能做，但为了后续可视化/保存更方便，建议使用 `open3d`（用于点云数据结构/导出/显示）。
- 若环境缺少 `open3d`，按 prepare.md 规则：直接 `pip install open3d`。

Config（继承 `pydantic.BaseModel`）建议字段：
- `depth_scale: float = 1.0`（若深度是米就保持 1；若是 mm 则设为 1000）
- `max_points: int | None = None`（可选，下采样上限）
- `mask_mode: Literal["all", "union", "per_mask"] = "all"`

类接口建议：

- 类名：`PointCloudGenerator`
- 核心方法：
  - `generate(depth: np.ndarray, K: np.ndarray, rgb: np.ndarray | None = None, masks: np.ndarray | None = None, rep_uv: np.ndarray | None = None) -> dict`
    - 输入：
      - `depth`: `[H, W]`
      - `K`: `[3, 3]`
      - `rgb`（可选）：`[H, W, 3]`，uint8，RGB 或 BGR（需要在 config/文档里固定一种）
      - `masks`（可选）：`[N, H, W]` bool
      - `rep_uv`（可选）：`[N, 2]`，int32
    - 输出：
      - `points_xyz: np.ndarray`，shape `[M, 3]`，float32
      - `colors_rgb: np.ndarray | None`，shape `[M, 3]`，uint8（或 float32 0..1）
      - `pixel_uv: np.ndarray`，shape `[M, 2]`，int32（用于后续索引/反查）
      - `mask_ids: np.ndarray | None`，shape `[M]`，int32（若开启按 mask 输出/打标签）
      - `rep_point_indices: np.ndarray | None`，shape `[N]`，int32（`rep_uv[i]` 在点云里的索引；找不到则 -1）

CLI 脚本约定（建议新文件）：
- `--depth_npy`
- `--image`（可选，用于颜色）
- `--K_json` 或 `--K_npy`（内参输入）
- `--masks_npz`（可选）
- `--positions_json/npz`（可选）
- `--output_dir`

输出建议：
- `pointcloud.npz`（points/colors/pixel_uv/mask_ids/rep_point_indices）
- 可选：`pointcloud.ply`（便于外部工具查看）

### 7.3 点云可视化模块（交互式 GUI）

**目标**：可交互查看点云；若提供位置表示索引，需要“显著”标出这些点。

实现思路（推荐）：
- 用 `open3d.visualization`：
  - 主点云：正常点大小 + 原色
  - 位置表示点：单独作为一个点云或用 sphere mesh（更醒目），颜色用高对比（例如纯红/纯黄），并把点大小调大

Config（继承 `pydantic.BaseModel`）建议字段：
- `point_size: int = 2`
- `rep_point_size: int = 10`
- `rep_color: tuple[int, int, int] = (255, 0, 0)`

类接口建议：

- 类名：`PointCloudVisualizer`
- 核心方法：
  - `show(points_xyz: np.ndarray, colors_rgb: np.ndarray | None = None, rep_point_indices: np.ndarray | None = None) -> None`
    - 输出：弹出可交互 GUI（拖动/缩放）

CLI 脚本约定（建议新文件）：
- `--pointcloud_npz`（或 `--points_npy` + `--colors_npy`）
- `--rep_indices_npy`（可选）

输出：
- 按你的要求，主要输出是交互式 GUI（可选提供 `--screenshot` 保存截图，便于记录）。

