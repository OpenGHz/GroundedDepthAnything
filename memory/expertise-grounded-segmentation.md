# Grounded image segmentation expertise

Updated 2026-08-25.

- For single-image open-vocabulary concept segmentation, the official SAM 3
  image API is `build_sam3_image_model` plus `Sam3Processor`; encode an image
  once and call `set_text_prompt` for each short concept. It returns masks,
  boxes, and scores directly, avoiding detector/point-error cascades.
- SAM 3.1 is an Object Multiplex video multi-object tracking update. It is not
  a replacement image checkpoint; single-image inference uses `facebook/sam3`
  (`sam3.pt`). The 3.1 checkpoint is `sam3.1_multiplex.pt` and is intended for
  the video path.
- GDA downloads the single-image `sam3.pt` from public ModelScope by default at
  revision `96f3e1b404ba14f2cfac60ee6ae87c269a7b7923`; gated Hugging Face is an
  explicit alternative pinned at `3c879f39826c281e95690f02c7821c4de09afae7`.
  Both must resolve to the same `3450062241`-byte file with SHA256
  `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`.
- ModelScope changes only distribution, not the Meta SAM License or its
  redistribution and trade-control/use restrictions. Do not silently switch
  provider or substitute another checkpoint when download or licensing fails.
- GroundingDINO -> SAM 2.1 remains a useful explicit baseline: pass detector
  boxes directly to SAM 2.1's box prompt. Normalize singleton batch dimensions
  because SAM 2 can squeeze `[N,C,H,W]` to `[C,H,W]` when `N=1`.

Authoritative sources:

- https://github.com/facebookresearch/sam3
- https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md
- https://huggingface.co/facebook/sam3
- https://modelscope.cn/models/facebook/sam3
- https://modelscope.cn/docs/models/download
- https://huggingface.co/facebook/sam3.1
- https://github.com/IDEA-Research/Grounded-SAM-2
