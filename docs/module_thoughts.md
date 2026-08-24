# GDA 当前模块架构与设计取舍

GDA 面向“单张图片 + 文本提示词”，把文本 grounded instance segmentation、单目
深度估计和三维位置后处理拆成可复用模块。当前默认分割链路是 native SAM3；
GroundingDINO → SAM2.1 作为显式 fallback 保留。

## 1. 源码边界与可复现性

项目直接使用的上游源码由 GDA 父仓库管理为 Git submodules：

- `third_party/sam3/`：native SAM3 image inference
- `third_party/depth-anything-3/`：Depth-Anything-3 API
- `third_party/grounded-sam-2/`：SAM2.1 fallback 的 Python/CUDA 实现

父仓库的 gitlinks 是源码版本的唯一默认 pin。Depth-Anything-3 自身还有嵌套
submodule，所以 checkout 必须执行 `git submodule update --init --recursive`。
`scripts/check-workspace.py` 会按 index 校验每层 gitlink、HEAD 与脏状态。

模型权重不进入 Git。SAM3、Depth-Anything-3、GroundingDINO 的 Hugging Face 加载
使用精确 revision；SAM2 checkpoint 使用固定 URL 与 SHA256。源码 pin 和权重 pin
共同构成可复现输入，不能用一个替代另一个。

SAM2 的 GPU 扩展需要针对 H200 `sm_90` 或 B300 `sm_103` 构建。GDA 将
Grounded-SAM-2 复制到 `.pixi/gda-build/`，只在该隔离副本中应用 CUDA patch 和
编译，避免让 submodule 因本机构建而变脏。

## 2. 稳定数据结构

跨模块数据结构集中在 `datatypes.py`，避免通过临时文件或隐式字段拼接。

### 2.1 `DetectionResult`

- `image_size: tuple[int, int]`：`(H, W)`
- `prompts: list[str]`
- `boxes_xyxy: np.ndarray`：`float32 [N, 4]`
- `scores: np.ndarray`：`float32 [N]`
- `prompt_ids: np.ndarray`：`int32 [N]`
- `labels: list[str]`
- `prompt_matches: list[list[int]] | None`：跨 prompt 去重后的全部归属

### 2.2 `SegmentationResult`

- `backend: str`
- `image_size: tuple[int, int]`
- `prompts: list[str]`
- `boxes_xyxy: np.ndarray`：`float32 [N, 4]`
- `prompt_ids: np.ndarray`：`int32 [N]`
- `scores: np.ndarray`：`float32 [N]`
- `masks: np.ndarray`：`bool [N, H, W]`
- `prompt_matches: list[list[int]] | None`

### 2.3 组合结果

- `GroundedSegmentationResult`：`DetectionResult + SegmentationResult`
- `DepthAndSegResult`：深度 + grounded segmentation
- `PositionsResult`：每个 mask 的代表像素、深度与有效性
- `ImageToPositionsResult`：完整 image → positions 结果
- `PointCloudResult`：点坐标、像素索引、相机内参以及可选颜色/实例 id

这些类型都使用内存对象连接模块；JSON/NPZ 仅是 CLI 边界的稳定序列化格式。

## 3. 模块划分

### 3.1 `modules/depth_estimation.py`

`DepthEstimatorDA3` 接收 `DepthEstimationConfig`，通过 Depth-Anything-3 预测单张图的
深度。模型从固定 Hugging Face revision 加载，并支持 local-files-only 模式。

核心接口：

```python
predict(image_rgb: np.ndarray | PIL.Image.Image) -> np.ndarray
```

输出统一为 `float32 [H, W]`；CLI 同时保存原始 `depth.npy` 和彩色
`depth.png`。

### 3.2 `modules/object_detection.py`

`GroundingDinoDetector` 使用 Transformers 版本的 GroundingDINO。多个 prompt 逐条
检测，之后直接写入确定的 `prompt_ids`，避免根据返回 phrase 做 substring 猜测。

核心接口：

```python
detect(image_rgb, prompts: list[str]) -> DetectionResult
```

该模块可单独用于 bbox 调试，也是 SAM2 fallback 的 detector。

### 3.3 `modules/object_segmentation.py`

`Sam2BoxSegmentor` 接受一张 RGB 图和 `DetectionResult`，把 boxes 直接作为 SAM2.1
prompt。它不再从 box 内寻找点，因而检测与分割之间的数据契约更简单、可复现。

核心接口：

```python
segment(image_rgb, det: DetectionResult) -> SegmentationResult
```

默认 checkpoint 位于
`$GDA_CACHE_DIR/checkpoints/sam2.1_hiera_large.pt`；未设置 cache 时通常位于
`~/.cache/gda/checkpoints/sam2.1_hiera_large.pt`。单实例和 multi-mask 输出会在模块
边界规范化成稳定的 `[N, H, W]` shape；CUDA 默认使用 BF16 autocast。

### 3.4 `modules/grounded_segmentation.py`

该模块定义统一的 `GroundedSegmentor` 接口和两个 backend：

1. `Sam3ConceptSegmentor`：默认路径，文本直接产生 boxes/masks/scores；多个 prompt
   共享一次图像编码。
2. `GroundingDinoSam2Segmentor`：显式 fallback，先检测 boxes，再进行 box-to-mask。

SAM3 会按 prompt 收集 instances，再依据 mask IoU 做跨 prompt 去重。保留实例的
`prompt_ids` 给出主归属，`prompt_matches` 保留所有匹配 prompt，防止去重丢语义。

SAM3 checkpoint `facebook/sam3/sam3.pt` 是 gated artifact。默认加载需要先获得
Hugging Face 权限；离线运行必须显式提供 `--sam3-checkpoint`。当前单图入口使用
SAM3 image model，不用面向视频 Object Multiplex 的 SAM3.1 链路。

### 3.5 `pipeline.py`

`ImageDepthAndSegPipeline` 组合深度估计与一个 grounded segmentation backend：

```text
RGB image ─┬─> Depth-Anything-3 ─> depth
           └─> SAM3 (default) ───> detections + masks
               or
               GroundingDINO ─> boxes ─> SAM2.1 ─> masks
```

核心接口只在内存中传递 RGB、depth 和结果 dataclasses：

```python
process(image_rgb: np.ndarray, prompts: list[str]) -> DepthAndSegResult
```

`run(image_path, prompts)` 仅作为兼容封装负责读图。CLI 输出深度、检测、mask 及
`depth_with_masks.png`。

### 3.6 后处理模块

- `position_representation.py`：在每个 mask 的有效深度中计算中值，并选择深度最接近
  中值的代表像素；无有效深度时设置 `valid=False`。
- `pointcloud_generation.py`：使用像素单位的 `K [3,3]` 将 depth 反投影，可携带 RGB、
  mask id 和代表点索引。
- `pointcloud_visualization.py`：可选 Open3D GUI；服务器无 display 时不应默认启动。
- `image_to_positions.py`：组合完整 image → depth/masks → positions 流程，并可选输出
  point cloud。

## 4. 配置和 CLI 约束

- 每个模块类的 `__init__` 仅接收一个 Pydantic config；主推理链配置使用冻结模型。
- 顶层 CLI 由 `pydantic-settings`/`CliApp` 解析，参数在进入推理前完成类型校验。
- Python API 的图像统一为 RGB `uint8 [H, W, 3]`；OpenCV BGR 只出现在 I/O 或
  可视化边界。
- prompt 支持逗号、分号和换行；shell 中应整体加引号。
- 输出目录未指定时默认为输入图目录。
- Hugging Face 离线模式必须使用已缓存的固定 revision；SAM3 离线模式还要求本地
  checkpoint 路径。
- 懒导入隔离可选重依赖：SAM3-only 不需要先构建 SAM2，单独检测也不加载 DA3。

## 5. 模型与源码版本

默认源码 gitlinks：

- SAM3：`8f0b7f4d4e7eda2ed606ebde6702c93359ad01da`
- Depth-Anything-3：`2c21ea849ceec7b469a3e62ea0c0e270afc3281a`
- Grounded-SAM-2：`b7a9c29f196edff0eb54dbe14588d7ae5e3dde28`

默认模型 revisions：

- `facebook/sam3@3c879f39826c281e95690f02c7821c4de09afae7`
- `depth-anything/DA3-LARGE@c54c26b16ec04d218e8d584ecf4bce082a9fcc20`
- `IDEA-Research/grounding-dino-base@12bdfa3120f3e7ec7b434d90674b3396eccf88eb`

升级任一源码或模型 pin 时，应在 H200/B300 至少重新验证 workspace check、lint、
单元测试、SAM2 CUDA doctor 和一张真实图片推理，并在同一次变更中更新相关文档。
