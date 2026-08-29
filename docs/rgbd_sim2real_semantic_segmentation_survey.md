# 直接基于单帧 RGB-D 图像的仿真到真实 2D 多类语义分割调研

**更新时间：** 2026-08-29
**检索问题：** 是否存在只用仿真数据训练、直接接收单帧 RGB-D、在真实环境上零样本泛化良好的 2D 多类语义分割工作，并且结果足以支持“模型学到了几何理解”？

**持久化研究 wiki：** [research-wiki/index.md](../research-wiki/index.md)（论文节点、G1–G6 gap map 与关系图）

## 结论先行

截至 2026-08-29，本轮**未确认到同时满足全部严格条件的无歧义论文**。这里的严格条件是：

1. 输入是一对同一时刻的 RGB 与 depth 图像，模型直接使用 depth（不是只把 depth 用作训练时的 privileged target）。
2. 输出是 2D、多类别、逐像素 semantic segmentation。
3. 监督、初始化、预训练和 checkpoint 选择都不接触真实图像。
4. 训练时不使用目标域适配、伪标签、校准、调参或目标域 checkpoint 选择。
5. 在真实环境上报告定量 zero-shot 结果。

采用论文中更常见的“**分割任务训练仅使用合成源域，允许通用预训练**”口径时，有两个最相关结果：

- **DSSS（2025）**是目前最接近的主方法：GTA5 训练，直接 RGB-D 输入，目标域不可见，在 Cityscapes 和 InfraParis 上分别达到 **42.07** 和 **42.05 mIoU**。但它明确使用 ImageNet 预训练 backbone，且 GTA5 depth 由 Monodepth2 生成。
- **MISFIT（WACVW 2024）**的论文内 source-only 消融行是一个很干净的任务协议基线：SYNTHIA→Cityscapes，RGB-D，16 类，**39.79 mIoU16**。不过 MiT-B5 的初始化未披露，因此无法判定是否满足“任何阶段不得接触真实图像”。MISFIT 主方法本身使用无标签真实目标域适配，不能算 zero-shot。

因此，最稳妥的判断是：**已有工作说明深度能提供可迁移的几何先验，但尚没有被严格隔离的证据证明单帧 RGB-D 多类分割模型在 literal synthetic-only 条件下学到了完整几何理解。**

## 判据与结果分层

| 层级 | 允许的设置 | 本轮结论 |
|---|---|---|
| 严格 literal synthetic-only | 任何初始化都不能接触真实图像；depth 最好是仿真器原生/物理传感器模型输出；无目标域训练或选择 | 未确认无歧义完整命中 |
| 任务级 synthetic-only | 分割任务的监督训练只用合成数据；允许 ImageNet 等通用预训练 | DSSS 主方法、MISFIT source-only 基线命中 |
| 几何迁移近邻 | depth-only，或实例/二类分割，或真实结果只有定性，或使用了校准/目标场景信息 | 有多项强证据，但不能替代目标问题 |

## 最相关论文

### 1. DSSS：最接近的主方法

**Binbin Wei et al., “Depth-Sensitive Soft Suppression with RGB-D Inter-Modal Stylization Flow for Domain Generalization Semantic Segmentation,” arXiv:2505.07050, 2025.**

- [论文（arXiv）](https://arxiv.org/abs/2505.07050)
- 任务：RGB-D、19 类城市道路 2D semantic segmentation；Domain Generalization，训练阶段只使用源域。
- 数据：GTA5、SYNTHIA、SELMA 为合成源/对照域；Cityscapes、InfraParis 为真实域。
- 真实 zero-shot 结果（ResNet-50，GTA5 训练）：

| 输入/方法 | Cityscapes | InfraParis |
|---|---:|---:|
| RGB baseline | 34.39 | 35.20 |
| 普通 RGB-D fusion | 37.99 | 37.39 |
| DSSS（完整方法） | **42.07** | **42.05** |

- 论文 Table 1/Table 3 的完整行还报告 GTA5→SYNTHIA `31.80`、GTA5→SELMA `38.28`，四个目标域平均 `38.55`。
- 深度处理：SYNTHIA/SELMA 的 depth 来自渲染器；GTA5 depth 按论文说明使用 Monodepth2 生成，不是 GTA5 渲染器原生深度。
- 初始化：所有 backbone 明确使用 ImageNet 预训练权重。
- 判定：**任务级 synthetic-only：是；严格 literal synthetic-only：否。**
- 几何含义：普通 RGB-D 相对 RGB 的提升，以及完整方法进一步提升，支持“深度中存在跨域有用的几何线索”；但不能单独排除 ImageNet 先验、深度估计器偏差和道路场景捷径。

### 2. MISFIT：source-only 行满足任务协议，主方法不满足

**Giulia Rizzoli, Donald Shenaj, Pietro Zanuttigh, “Source-Free Domain Adaptation for RGB-D Semantic Segmentation with Vision Transformers,” WACVW 2024 (arXiv:2305.14269).**

- [CVF 原文](https://openaccess.thecvf.com/content/WACV2024W/Pretrain/html/Rizzoli_Source-Free_Domain_Adaptation_for_RGB-D_Semantic_Segmentation_With_Vision_Transformers_WACVW_2024_paper.html)
- [arXiv](https://arxiv.org/abs/2305.14269)
- 任务：直接 RGB+depth，SYNTHIA→Cityscapes，16 个共同类别。
- 论文 Table 3/Table 4 的消融表注明：关闭所有模块时对应 source-only：

| 设置 | Cityscapes mIoU16 |
|---|---:|
| RGB source-only | 36.93 |
| RGB-D source-only（Key Swap） | **39.79** |

- SYNTHIA 有 9,400 张合成源图；Cityscapes 是真实目标域，提供 stereo depth。
- 主 MISFIT 会使用无标签真实目标图像进行 source-free adaptation，最终 `54.5 mIoU` 的主结果**不能**作为 zero-shot 结果。
- 初始化审计：正文、补充材料和 TeX 源只说明 SegFormer/MiT-B5 在 source data 上训练 40 epochs/160k iterations，没有说明 ImageNet、其他真实图像预训练或随机初始化。不能按 SegFormer 惯例推断。
- 判定：**source-only 基线在任务级口径下命中；严格口径为 indeterminate（初始化未披露）。**

### 3. Watanabe et al.：重要的负例

**Kohei Watanabe, Kuniaki Saito, Yoshitaka Ushiku, Tatsuya Harada, “Multichannel Semantic Segmentation with Unsupervised Domain Adaptation,” AUTONUE Workshop of ECCV 2018 (arXiv:1812.04351).**

- [论文（arXiv）](https://arxiv.org/abs/1812.04351)
- 任务：SUNCG→NYUv2，34 个共同室内类别；直接 RGB+HHA（depth 的三通道编码）。
- source-only EarlyFusion 结果（Table 1）：

| 设置 | pixAcc | mAcc | fwIoU | mIoU |
|---|---:|---:|---:|---:|
| RGB source-only | 13.0 | 6.7 | 9.9 | 3.2 |
| HHA source-only | 15.6 | 9.7 | 9.0 | 3.8 |
| RGB+HHA EarlyFusion | 17.9 | 9.9 | 10.0 | **4.2** |

- 训练使用 568,793 张 SUNCG 合成图；真实 NYUv2 测试集有 654 张图。
- 论文明确写明 DRN-D-38 使用 ImageNet 预训练。
- 其 Adapt 行使用 795 张无标签 NYUv2 训练图，属于 UDA，不应与 source-only 混列。
- 判定：是直接 RGB-D 多类 source-only 证据，但**不满足严格的无真实预训练条件**；同时说明“有 depth”并不自动带来强 sim-to-real 泛化。

## 强几何迁移近邻（但不满足完整目标）

| 工作 | 输入/任务 | 仿真→真实证据 | 不能作为严格匹配的原因 |
|---|---|---|---|
| **Lim et al., EMBC 2019**, [DOI](https://doi.org/10.1109/EMBC.2019.8857700) | depth-only；手、臂、物体、桌面等多标签 2D 分割 | 10,000 张合成 depth 训练、1,000 张 Kinect-v2 真实测试；IEEE 摘要报告 **70.4% mIoU**，约 6 ms | 没有 RGB，类别/场景较窄；是很强的几何迁移证据，但不是 RGB-D 结果 |
| **Handa et al., SceneNet, 2015**, [arXiv](https://arxiv.org/abs/1511.07041) | depth-only；13 类室内语义分割 | SceneNet→NYUv2：global/class accuracy `54.4/37.1`；SceneNet→SUN RGB-D：`56.9/30.2` | depth-only；使用 VGG/ImageNet 相关初始化；部分设置含真实微调 |
| **Planche & Singh, DDS, 2021**, [arXiv](https://arxiv.org/abs/2103.16563) | depth-only；2D-3D-S，8 个 depth 可区分类别 | 真实测试 pixel accuracy：clean `35.3%`、DepthSynth `65.3%`、off-the-shelf DDS `62.9%`；经真实扫描校准的 DDS `69.8%` | 校准版本使用真实扫描；重建的目标场景用于渲染，存在场景重叠风险；不是 RGB-D |
| **Sharma et al., 2016**, [arXiv](https://arxiv.org/abs/1605.08464) | depth-only；人体部位与工业物体，约 10 类 | 20,000 合成 depth 训练、65 张真实 Kinect 测试；平均 F1 `0.84` | 合成动作/人体模型由真实 Kinect skeleton 记录辅助生成；不是 RGB-D 深度网络 |
| **Li et al., Sensors 2020**, [DOI](https://doi.org/10.3390/s20185318) | depth-only；PCB 多组件像素分类 | 合成测试 `98.96%`、真实测试 `83.64%` pixel accuracy | 真实测试仅 10 张；为应对真实噪声，真实测试使用了不同的 yaw/pitch 采样范围和特征尺度参数，不能视为完全无调参 zero-shot |
| **Joukovsky et al., Electronics Letters 2020**, [DOI](https://doi.org/10.1049/el.2019.4150) | 直接 RGB-D；服装 9 类 | 53,354 张合成 RGB-D；合成测试 **92.05 mIoU**；Kinect-v2 真实结果展示为定性图 | 真实域无定量指标；Xception 使用 Pascal VOC 预训练权重 |
| **Digumarti et al., ICRA 2019**, [DOI](https://doi.org/10.1109/ICRA.2019.8793576) | 单帧 RGB-D；树干、枝、细枝、叶 | 合成数据跨树种最高约 **92.5% accuracy**；真实数据做了定性展示 | 真实域无定量结果 |

## 实例分割近邻（不应与 semantic segmentation 混淆）

**Back et al., “Segmenting Unseen Industrial Components in a Heavy Clutter Using RGB-D Fusion and Synthetic Data,” ICIP 2020, [arXiv:2002.03501](https://arxiv.org/abs/2002.03501), [DOI](https://doi.org/10.1109/ICIP40778.2020.9190804).**

- 35,000 张合成 RGB-D 训练/验证数据，100 对真实 RealSense 测试图；训练只使用合成数据。
- RGB-D fusion 在真实数据上达到 AP50 `69.0`、AP `57.7`、AR `66.1`。
- 这是类别无关 industrial **instance segmentation**，不是多类 semantic segmentation；RGB branch 使用 ImageNet 预训练 ResNet-50。

类似的 WISDOM、UCN、SupeRGB-D 和 2025 年 CGA-ASNet 也提供了有价值的 sim-to-real RGB-D 实例分割证据，但都不满足本文的多类 semantic segmentation 定义；CGA-ASNet 还是单一番茄类别的 amodal instance segmentation。

## 明确排除的工作类型

| 工作/类型 | 排除理由 |
|---|---|
| DADA（DADA: Depth-Aware Domain Adaptation） | depth 主要作为 UDA 的 privileged information；训练使用真实目标图像 |
| RaSim | 有合成 RGB-D 和像素标签，但实验没有真实语义分割定量结果，重点是 depth completion/6D pose |
| Ansari et al. | RGB-D 直接输入，但 Cityscapes 与 CARLA 分别训练和评测，没有 synthetic→real 定量协议 |
| Syn-Mediverse | 合成 RGB-D、多类标签，但真实结果只有定性展示 |
| Guardrail | 分割网络实际只输入 RGB；depth 用于后续尺寸/类型推理 |
| Crack segmentation (CVIU 2025) | 二类分割，且训练混合真实与合成数据 |
| Mahé et al. | 使用真实 RGB-D 训练；CAD 只用于产生 noisy labels |

## 对“学到几何理解”的可辩护表述

### 可以说

- 模型利用了跨域相对稳定的几何线索，例如距离、高度、轮廓、遮挡、平面和粗粒度空间关系。
- 在 DSSS 和 MISFIT 的 source-only 对照中，加入 depth 相对 RGB 有可测提升，说明 depth 不是完全冗余的输入。
- depth-only 工作（尤其 Lim/SceneNet/DDS）说明，即使外观信息极弱，合成几何仍可能迁移到真实传感器。

### 不能直接说

- “已经证明模型具有完整 3D/几何理解”。现有结果可能依赖 ImageNet/ Pascal VOC 先验、深度估计器、固定相机和场景布局、传感器特定噪声，或目标 CAD/扫描重叠。
- “DSSS 是纯仿真从零训练”。它使用 ImageNet backbone 初始化，且 GTA5 depth 不是渲染器原生深度。
- “MISFIT 的 39.79 是主方法 zero-shot 结果”。这是关闭适配模块后的 source-only 消融行；主方法使用真实目标域无标签数据。

## 建议的严格验证协议

若目标是把“几何理解”作为论文 claim，建议至少做以下对照：

1. 随机初始化、相同容量的 RGB-only / depth-only / RGB-D 三组模型；另报通用预训练版本，隔离预训练贡献。
2. 同一几何随机纹理、同一纹理随机几何，以及 held-out CAD、布局、相机内外参和传感器组合。
3. 真实测试集在模型和 checkpoint 选择中完全不可见；至少使用两个互不相关的真实传感器/场景集。
4. 做 depth shuffle、尺度扰动、边缘破坏、孔洞/噪声、RGB-depth 错位和遮挡干预，区分“真正空间关系”与“深度边缘捷径”。
5. 同时报告 mIoU、按距离/遮挡/平面类别分层的结果，以及 RGB 与 depth 的条件互补性，而不只报告一个总平均值。

## 参考文献与一手来源

- Wei et al. (2025), DSSS — [arXiv:2505.07050](https://arxiv.org/abs/2505.07050)
- Rizzoli, Shenaj, Zanuttigh (2024), MISFIT — [CVF 原文](https://openaccess.thecvf.com/content/WACV2024W/Pretrain/html/Rizzoli_Source-Free_Domain_Adaptation_for_RGB-D_Semantic_Segmentation_With_Vision_Transformers_WACVW_2024_paper.html), [arXiv:2305.14269](https://arxiv.org/abs/2305.14269)
- Watanabe et al. (2018), Multichannel Semantic Segmentation with UDA — [arXiv:1812.04351](https://arxiv.org/abs/1812.04351)
- Handa et al. (2015), SceneNet — [arXiv:1511.07041](https://arxiv.org/abs/1511.07041)
- Planche & Singh (2021), Physics-based Differentiable Depth Sensor Simulation — [arXiv:2103.16563](https://arxiv.org/abs/2103.16563)
- Lim et al. (2019), Hand and Object Segmentation from Depth Image using FCN — [IEEE DOI](https://doi.org/10.1109/EMBC.2019.8857700)
- Sharma et al. (2016), Low-Cost Scene Modeling using a Density Function — [arXiv:1605.08464](https://arxiv.org/abs/1605.08464)
- Li et al. (2020), Semantic Segmentation of a Printed Circuit Board — [Sensors DOI](https://doi.org/10.3390/s20185318)
- Joukovsky, Hu, Munteanu (2020), Multi-modal deep network for RGB-D segmentation of clothes — [DOI](https://doi.org/10.1049/el.2019.4150)
- Digumarti et al. (2019), An Approach for Semantic Segmentation of Tree-like Vegetation — [IEEE DOI](https://doi.org/10.1109/ICRA.2019.8793576)
- Back et al. (2020), Segmenting Unseen Industrial Components — [arXiv:2002.03501](https://arxiv.org/abs/2002.03501), [IEEE DOI](https://doi.org/10.1109/ICIP40778.2020.9190804)

## 检索记录

- 检索日期：2026-08-29。
- 使用了 arXiv、IEEE/CVF、IEEE Xplore、Wiley、MDPI、OpenAlex/Crossref 等来源，并对关键候选阅读了论文正文或官方页面中的实验与实现细节。
- 本报告不把搜索摘要、博客或二手榜单当作定量证据；表中数字优先来自论文表格、论文正文或出版方页面。
- 本项目未配置可供本轮扫描的本地论文库；临时核验材料保存在 `/tmp/rgbd_research/`，未复制进仓库。

WARN: local contributed nothing — no PDFs found in papers/, literature/, or a configured paper library. To include yours, add a "## Paper Library" heading to AGENTS.md followed by the directory path.
