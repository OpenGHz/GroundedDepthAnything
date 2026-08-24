from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from gda.datatypes import DetectionResult
from gda.modules import grounded_segmentation as grounded
from gda.modules.object_detection import GroundingDinoDetector
from PIL import Image
from pydantic import ValidationError


class FakeSam3Processor:
    last_instance: FakeSam3Processor | None = None

    def __init__(self, model, resolution, device, confidence_threshold):
        self.model = model
        self.resolution = resolution
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.set_image_calls = 0
        self.reset_calls = 0
        self.prompts: list[str] = []
        type(self).last_instance = self

    def set_image(self, image):
        assert isinstance(image, Image.Image)
        self.set_image_calls += 1
        return {"backbone_out": object()}

    def reset_all_prompts(self, state):
        assert "backbone_out" in state
        self.reset_calls += 1

    def set_text_prompt(self, state, prompt):
        self.prompts.append(prompt)
        height, width = 6, 8
        mask = torch.zeros((1, 1, height, width), dtype=torch.bool)
        if prompt == "cup":
            mask[:, :, 1:4, 2:5] = True
            score = 0.8
        else:
            mask[:, :, 2:5, 4:7] = True
            score = 0.7
        state.update(
            masks=mask,
            boxes=torch.tensor([[2.0, 1.0, 5.0, 4.0]], dtype=torch.float32),
            scores=torch.tensor([score], dtype=torch.float32),
        )
        return state


def _build_fake_segmentor(monkeypatch, checkpoint: Path, *, deduplicate_mask_iou=None):
    builder_calls: list[dict] = []

    def fake_builder(**kwargs):
        builder_calls.append(kwargs)
        return object()

    monkeypatch.setattr(
        grounded,
        "_import_sam3_components",
        lambda: (fake_builder, lambda version: checkpoint, FakeSam3Processor),
    )
    config = grounded.Sam3ConceptSegmentationConfig(
        device="cpu",
        checkpoint=checkpoint,
        load_from_hf=False,
        autocast_dtype="none",
        deduplicate_mask_iou=deduplicate_mask_iou,
    )
    return grounded.Sam3ConceptSegmentor(config), builder_calls


def test_sam3_concept_reuses_image_embedding_and_maps_prompts(monkeypatch, tmp_path):
    checkpoint = tmp_path / "sam3.pt"
    checkpoint.touch()
    segmentor, builder_calls = _build_fake_segmentor(monkeypatch, checkpoint)

    image = np.zeros((6, 8, 3), dtype=np.uint8)
    result = segmentor.segment(image, [" cup ", "bottle"])

    processor = FakeSam3Processor.last_instance
    assert processor is not None
    assert processor.set_image_calls == 1
    assert processor.reset_calls == 2
    assert processor.prompts == ["cup", "bottle"]
    assert result.seg.masks.shape == (2, 6, 8)
    assert result.seg.masks.dtype == np.bool_
    np.testing.assert_array_equal(result.seg.prompt_ids, [0, 1])
    assert result.det.labels == ["cup", "bottle"]
    assert builder_calls == [
        {
            "device": "cpu",
            "checkpoint_path": str(checkpoint),
            "load_from_HF": False,
            "enable_segmentation": True,
            "enable_inst_interactivity": False,
            "compile": False,
        }
    ]


def test_sam3_output_normalizes_empty_and_channel_shapes():
    boxes, scores, masks = grounded.Sam3ConceptSegmentor._normalize_output(
        {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "scores": torch.zeros((0,), dtype=torch.float32),
            "masks": torch.zeros((0, 1, 5, 7), dtype=torch.bool),
        },
        image_size=(5, 7),
    )
    assert boxes.shape == (0, 4)
    assert scores.shape == (0,)
    assert masks.shape == (0, 5, 7)
    assert masks.dtype == np.bool_


def test_sam3_bfloat16_output_converts_to_numpy_float32():
    output = grounded._to_numpy(torch.tensor([0.5], dtype=torch.bfloat16))
    assert output.dtype == np.float32
    np.testing.assert_allclose(output, [0.5], rtol=1e-2)


def test_cross_prompt_dedup_keeps_highest_score_and_aliases():
    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:3] = True
    boxes = np.asarray([[1, 1, 3, 3], [1, 1, 3, 3]], dtype=np.float32)
    scores = np.asarray([0.4, 0.9], dtype=np.float32)
    masks = np.stack([mask, mask])
    prompt_ids = np.asarray([0, 1], dtype=np.int32)

    output = grounded.Sam3ConceptSegmentor._deduplicate(
        boxes=boxes,
        scores=scores,
        masks=masks,
        prompt_ids=prompt_ids,
        labels=["cup", "mug"],
        threshold=0.9,
    )
    out_boxes, out_scores, out_masks, out_prompt_ids, out_labels, prompt_matches = output
    assert out_boxes.shape == (1, 4)
    assert out_masks.shape == (1, 4, 4)
    np.testing.assert_allclose(out_scores, [0.9])
    np.testing.assert_array_equal(out_prompt_ids, [1])
    assert out_labels == ["mug"]
    assert prompt_matches == [[0, 1]]


def test_same_prompt_overlaps_are_not_deduplicated():
    mask = np.ones((3, 3), dtype=bool)
    output = grounded.Sam3ConceptSegmentor._deduplicate(
        boxes=np.zeros((2, 4), dtype=np.float32),
        scores=np.asarray([0.8, 0.7], dtype=np.float32),
        masks=np.stack([mask, mask]),
        prompt_ids=np.asarray([0, 0], dtype=np.int32),
        labels=["person", "person"],
        threshold=0.5,
    )
    assert len(output[1]) == 2
    assert output[-1] == [[0], [0]]


def test_detection_json_round_trip_preserves_prompt_matches():
    result = DetectionResult(
        image_size=(10, 20),
        prompts=["cup", "mug"],
        boxes_xyxy=np.asarray([[1, 2, 3, 4]], dtype=np.float32),
        scores=np.asarray([0.9], dtype=np.float32),
        prompt_ids=np.asarray([1], dtype=np.int32),
        labels=["mug"],
        prompt_matches=[[0, 1]],
    )
    restored = DetectionResult.from_json_dict(result.to_json_dict())
    assert restored.prompt_matches == [[0, 1]]
    np.testing.assert_array_equal(restored.boxes_xyxy, result.boxes_xyxy)


def test_sam3_config_requires_a_checkpoint_source():
    with pytest.raises(ValidationError):
        grounded.Sam3ConceptSegmentationConfig(
            device="cpu",
            checkpoint=None,
            load_from_hf=False,
        )


def test_sam3_offline_mode_requires_explicit_checkpoint(monkeypatch):
    monkeypatch.setattr(
        grounded,
        "_import_sam3_components",
        lambda: (object(), lambda version: pytest.fail("must not download"), object()),
    )
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    config = grounded.Sam3ConceptSegmentationConfig(
        device="cpu",
        checkpoint=None,
        load_from_hf=True,
        autocast_dtype="none",
    )
    with pytest.raises(FileNotFoundError, match="offline mode requires"):
        grounded.Sam3ConceptSegmentor(config)


def test_grounding_dino_assigns_prompt_ids_without_phrase_guessing():
    class Inputs(dict):
        input_ids = torch.tensor([[1]])

        def to(self, device):
            return self

    class Processor:
        def __init__(self):
            self.texts: list[str] = []

        def __call__(self, *, images, text, return_tensors):
            self.texts.append(text)
            return Inputs()

        def post_process_grounded_object_detection(self, outputs, input_ids, **kwargs):
            return [
                {
                    "boxes": torch.tensor([[1.0, 1.0, 4.0, 4.0]]),
                    "scores": torch.tensor([0.8]),
                    "labels": ["ambiguous phrase"],
                }
            ]

    class Model:
        def __call__(self, **inputs):
            return {}

    detector = GroundingDinoDetector.__new__(GroundingDinoDetector)
    detector.device = torch.device("cpu")
    detector.config = type("Config", (), {"box_threshold": 0.2, "text_threshold": 0.3})()
    detector.processor = Processor()
    detector.model = Model()
    result = detector.detect(np.zeros((6, 8, 3), dtype=np.uint8), ["car", "car wheel"])
    np.testing.assert_array_equal(result.prompt_ids, [0, 1])
    assert detector.processor.texts == ["car.", "car wheel."]
