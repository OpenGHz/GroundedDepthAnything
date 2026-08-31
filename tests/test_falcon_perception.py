from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils

from gda.modules import falcon_perception as falcon
from gda.modules import grounded_segmentation as grounded
from gda.modules import object_detection


class _FakeRunner:
    def __init__(self, outputs: dict[str, list[dict]]):
        self.outputs = outputs
        self.calls: list[tuple[str, str]] = []

    def generate(self, image, query, *, task):
        self.calls.append((query, task))
        return self.outputs.get(query, [])

    @staticmethod
    def box(prediction, image_size):
        return falcon._prediction_box(prediction, image_size)

    @staticmethod
    def mask(prediction, image_size):
        return np.asarray(prediction["mask"], dtype=bool)


def _prediction(*, x=0.5, y=0.5, h=0.2, w=0.4, mask=None):
    result = {"xy": {"x": x, "y": y}, "hw": {"h": h, "w": w}}
    if mask is not None:
        result["mask_rle"] = {"counts": "unused", "size": list(mask.shape)}
        result["mask"] = mask
    return result


def test_falcon_runner_passes_pinned_revision_and_generation_options(monkeypatch):
    calls: list[tuple[str, dict]] = []
    export_dir = Path(__file__).parent / "_fake_falcon_export"

    class FakeModel:
        def to(self, device):
            self.device = device
            return self

        def eval(self):
            return self

        def generate(self, image, query, task=None, **kwargs):
            calls.append(("generate", {"image": image, "query": query, "task": task, **kwargs}))
            return [[_prediction()]]

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls.append((model_id, kwargs))
            return FakeModel()

    monkeypatch.setattr(falcon, "_import_falcon_model", lambda: FakeAutoModel)
    monkeypatch.setattr(falcon, "_resolve_falcon_export", lambda config: export_dir)
    monkeypatch.setattr(falcon, "_FalconTokenizer", lambda export_dir: object())
    config = falcon.FalconPerceptionConfig(
        model_id="tiiuae/test-falcon",
        model_revision="pinned-revision",
        device="cpu",
        compile=False,
        max_new_tokens=64,
    )
    runner = falcon.FalconPerceptionRunner(config)
    output = runner.generate(Image.new("RGB", (8, 6)), "cup", task="detection")

    assert len(output) == 1
    assert calls[0] == (
        str(export_dir),
        {
            "trust_remote_code": True,
            "local_files_only": False,
            "dtype": torch.float32,
        },
    )
    _, generate_call = calls[1]
    assert generate_call["query"] == "cup"
    assert generate_call["task"] == "detection"
    assert generate_call["max_new_tokens"] == 64
    assert generate_call["compile"] is False


def test_falcon_runner_omits_task_for_full_model_signature(monkeypatch):
    class FakeModel:
        def to(self, device):
            return self

        def eval(self):
            return self

        def generate(self, image, query, **kwargs):
            assert "task" not in kwargs
            return [[]]

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            return FakeModel()

    monkeypatch.setattr(falcon, "_import_falcon_model", lambda: FakeAutoModel)
    export_dir = Path(__file__).parent / "_fake_falcon_export"
    monkeypatch.setattr(falcon, "_resolve_falcon_export", lambda config: export_dir)
    monkeypatch.setattr(falcon, "_FalconTokenizer", lambda export_dir: object())
    runner = falcon.FalconPerceptionRunner(
        falcon.FalconPerceptionConfig(model_id="local", model_revision=None, device="cpu")
    )
    assert runner.generate(Image.new("RGB", (8, 6)), "cup", task="segmentation") == []


def test_falcon_box_and_rle_are_normalized_to_gda_contract():
    box = falcon._prediction_box(_prediction(), image_size=(100, 200))
    np.testing.assert_allclose(box, [60.0, 40.0, 140.0, 60.0])

    mask = np.zeros((5, 7), dtype=np.uint8)
    mask[1:4, 2:6] = 1
    rle = mask_utils.encode(np.asfortranarray(mask))
    rle["counts"] = rle["counts"].decode("utf-8")
    decoded = falcon._decode_rle_mask(rle, image_size=(5, 7))
    np.testing.assert_array_equal(decoded, mask.astype(bool))


def test_falcon_detector_maps_multiple_prompts_without_confidence():
    config = object_detection.ObjectDetectionConfig(
        backend="falcon",
        device="cpu",
        falcon=falcon.FalconPerceptionConfig(device="cpu", score=0.75),
    )
    detector = object_detection.FalconPerceptionDetector.__new__(
        object_detection.FalconPerceptionDetector
    )
    detector.config = config
    detector.runner = _FakeRunner({"cup": [_prediction()], "bottle": [_prediction(x=0.25, w=0.1)]})

    result = detector.detect(np.zeros((100, 200, 3), dtype=np.uint8), [" cup ", "bottle"])

    assert result.image_size == (100, 200)
    assert result.prompts == ["cup", "bottle"]
    assert result.labels == ["cup", "bottle"]
    np.testing.assert_array_equal(result.prompt_ids, [0, 1])
    np.testing.assert_allclose(result.scores, [0.75, 0.75])
    assert detector.runner.calls == [("cup", "detection"), ("bottle", "detection")]


def test_falcon_segmentor_maps_masks_and_prompts():
    cup_mask = np.zeros((6, 8), dtype=bool)
    cup_mask[1:4, 2:5] = True
    bottle_mask = np.zeros((6, 8), dtype=bool)
    bottle_mask[2:5, 5:7] = True
    segmentor = grounded.FalconConceptSegmentor.__new__(grounded.FalconConceptSegmentor)
    segmentor.config = falcon.FalconPerceptionConfig(device="cpu", score=0.6)
    segmentor.runner = _FakeRunner(
        {
            "cup": [_prediction(mask=cup_mask)],
            "bottle": [_prediction(mask=bottle_mask)],
        }
    )

    result = segmentor.segment(np.zeros((6, 8, 3), dtype=np.uint8), ["cup", "bottle"])

    assert result.seg.backend == "falcon"
    assert result.seg.masks.shape == (2, 6, 8)
    np.testing.assert_array_equal(result.seg.masks[0], cup_mask)
    np.testing.assert_array_equal(result.seg.prompt_ids, [0, 1])
    np.testing.assert_allclose(result.seg.scores, [0.6, 0.6])
    assert result.det.labels == ["cup", "bottle"]
    assert segmentor.runner.calls == [
        ("cup", "segmentation"),
        ("bottle", "segmentation"),
    ]


def test_backend_factories_select_falcon(monkeypatch):
    detector_sentinel = object()
    segmentor_sentinel = object()
    monkeypatch.setattr(
        object_detection,
        "FalconPerceptionDetector",
        lambda config: detector_sentinel,
    )
    monkeypatch.setattr(grounded, "FalconConceptSegmentor", lambda config: segmentor_sentinel)

    detector_config = object_detection.ObjectDetectionConfig(backend="falcon", device="cpu")
    segmentor_config = grounded.GroundedSegmentationConfig(backend="falcon")
    assert object_detection.build_object_detector(detector_config) is detector_sentinel
    assert grounded.build_grounded_segmentor(segmentor_config) is segmentor_sentinel
