# Grounded segmentation 技术调研（2026-08-25 更新）

## 结论

- **单图默认：native SAM 3 text-to-mask。** 使用官方
  `build_sam3_image_model + Sam3Processor`，图像只编码一次，对每个短概念提示调用
  `reset_all_prompts + set_text_prompt`，直接得到实例 `masks / boxes / scores`。
- **单图 fallback：GroundingDINO -> SAM 2.1 box prompt。** 检测框应直接交给
  SAM 2.1，不再在框中采点；点采样会额外引入误差，且 SAM 2.1 原生支持 box prompt。
- **视频多目标：SAM 3.1 Object Multiplex。** SAM 3.1 是共享 memory 的视频多目标
  联合跟踪更新，不是新的单图模型；单图仍使用 `facebook/sam3/sam3.pt`。

## 为什么替换旧链路

旧链路是 `text -> GroundingDINO box -> 框内选点 -> SAM 2 mask`。native SAM 3
直接针对“给定开放词汇概念，穷举匹配实例并分割”训练，省去了检测器、点采样和两套阈值
之间的误差级联。官方报告中，SAM 3 在 LVIS instance segmentation 上为 AP 48.5，
在 SA-Co/Gold 上为 cgF1 54.1；这些结果说明它适合作为该任务的原生默认模型，但跨论文
指标仍不应脱离数据集设置直接横比。

SAM 3 image API 会缓存一次 image backbone output。本项目按 prompt 顺序逐个设置文本，
保留精确 `prompt_ids`；可选的跨 prompt mask-IoU 去重会记录一个实例匹配到的全部
`prompt_matches`，避免 `cup`/`mug` 等近义提示重复输出同一实例。

## 备选方法

| 方法 | 能力与适用场景 | 不作为默认的原因 |
| --- | --- | --- |
| Florence-2-large-ft | MIT；phrase grounding、open-vocabulary detection、referring-expression polygon；适合长描述与自动标注 | 生成式 box/polygon 再接 SAM 的链路更长，不如 SAM 3 原生 exhaustive masks |
| OmDet-Turbo | Apache-2.0；高速 open-vocabulary detector | 没有 mask head，仍需接 SAM |
| APE | Apache-2.0；单模型 detection/grounding/instance/semantic segmentation | Detectron2/EVA 依赖重，部署和维护成本较高 |
| OpenWorldSAM | Apache-2.0；冻结 SAM 2 上统一 semantic/instance/panoptic/referring segmentation | 研究结果强，但依赖定制 Detectron2，总体工程复杂 |
| YOLOE / YOLOE-26 | text/visual/prompt-free detection + segmentation；适合低延迟部署 | AGPL-3.0/商业许可约束；质量优先时不替代 SAM 3 |

## 权重与许可

- 单图 SAM3 默认从公开的 ModelScope `facebook/sam3` 获取 `sam3.pt`，固定 revision
  `96f3e1b404ba14f2cfac60ee6ae87c269a7b7923`；Hugging Face gated 仓库作为显式
  可选来源，固定 revision `3c879f39826c281e95690f02c7821c4de09afae7`。
- 两个来源的 `sam3.pt` 均应为 `3450062241` bytes，SHA256
  `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`。项目只下载
  此单文件，不默认拉取还包含另一套权重的完整仓库，也不会在 provider 失败时静默
  切换来源或 checkpoint。
- ModelScope 只改变分发来源，不改变 SAM 3 的 Meta SAM License。该许可证不是
  Apache/MIT；再分发、贸易管制及军事/武器用途有额外条款，部署前必须审阅。

## SAM2 fallback 的 B300 工程验证

验证节点：8x NVIDIA B300 SXM6 AC，driver 580.82.07，单卡约 275 GiB。

- Pixi B300 target：Python 3.12、PyTorch 2.10.0+cu130、CUDA toolkit 13.0、
  xFormers 0.0.35、Flash Attention 3.0.0。
- 实测 capability 为 `(10, 3)`；SAM 2 CUDA extension 已按 `sm_103` 编译并成功
  导入 `sam2._C.so`。
- `cars.jpg`、prompt `car`：GroundingDINO 检出 2 个实例，SAM 2.1 输出
  2 个 `1500 x 2250` mask；检测分数约 0.806 / 0.668。
- 同一进程三次推理（检测 + 分割）：FP32 为 1.622 / 0.200 / 0.185 秒；SAM 2.1
  BF16 autocast 为 0.794 / 0.165 / 0.154 秒。峰值 allocated 约 3.0 GiB。
- 单 box + `multimask_output=True` 的真实 predictor 回归通过，输出形状为
  `[1, 1500, 2250]`，修复了旧代码对 singleton batch squeeze 的崩溃。

## 一手来源

- [SAM 3 官方仓库](https://github.com/facebookresearch/sam3)
- [SAM 3 论文](https://arxiv.org/abs/2511.16719)
- [SAM 3 模型卡](https://huggingface.co/facebook/sam3)
- [SAM 3 ModelScope 仓库](https://modelscope.cn/models/facebook/sam3)
- [ModelScope 模型下载文档](https://modelscope.cn/docs/models/download)
- [SAM 3.1 Object Multiplex 发布说明](https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md)
- [SAM 3.1 模型卡](https://huggingface.co/facebook/sam3.1)
- [Grounded-SAM-2 官方仓库](https://github.com/IDEA-Research/Grounded-SAM-2)
- [GroundingDINO 官方仓库](https://github.com/IDEA-Research/GroundingDINO)
- [SAM 2 官方仓库](https://github.com/facebookresearch/sam2)
- [Florence-2-large-ft 模型卡](https://huggingface.co/microsoft/Florence-2-large-ft)
- [OmDet 官方仓库](https://github.com/om-ai-lab/OmDet)
- [APE 官方仓库](https://github.com/shenyunhang/APE)
- [OpenWorldSAM 官方仓库](https://github.com/GinnyXiao/OpenWorldSAM)
- [YOLOE 官方仓库](https://github.com/THU-MIG/yoloe)
