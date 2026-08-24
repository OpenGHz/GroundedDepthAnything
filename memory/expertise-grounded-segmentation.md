# Grounded image segmentation expertise

Updated 2026-08-22.

- For single-image open-vocabulary concept segmentation, the official SAM 3
  image API is `build_sam3_image_model` plus `Sam3Processor`; encode an image
  once and call `set_text_prompt` for each short concept. It returns masks,
  boxes, and scores directly, avoiding detector/point-error cascades.
- SAM 3.1 is an Object Multiplex video multi-object tracking update. It is not
  a replacement image checkpoint; single-image inference uses `facebook/sam3`
  (`sam3.pt`). The 3.1 checkpoint is `sam3.1_multiplex.pt` and is intended for
  the video path.
- Both SAM 3 checkpoints are gated on Hugging Face and use Meta's SAM License,
  which has redistribution and trade-control/use restrictions. Treat access or
  licensing failure as a hard requirement, not a reason to silently substitute
  another checkpoint.
- GroundingDINO -> SAM 2.1 remains a useful explicit baseline: pass detector
  boxes directly to SAM 2.1's box prompt. Normalize singleton batch dimensions
  because SAM 2 can squeeze `[N,C,H,W]` to `[C,H,W]` when `N=1`.

Authoritative sources:

- https://github.com/facebookresearch/sam3
- https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md
- https://huggingface.co/facebook/sam3
- https://huggingface.co/facebook/sam3.1
- https://github.com/IDEA-Research/Grounded-SAM-2
