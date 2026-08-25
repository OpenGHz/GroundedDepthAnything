from __future__ import annotations

from gda.modules import depth_estimation, grounded_segmentation, object_detection


class _FakeModel:
    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self


def test_depth_estimator_passes_pinned_revision(monkeypatch):
    calls: list[tuple[str, dict]] = []

    class FakeDepthAnything:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            calls.append((model_name, kwargs))
            return _FakeModel()

    monkeypatch.setattr(depth_estimation, "_import_depth_anything3", lambda: FakeDepthAnything)
    config = depth_estimation.DepthEstimationConfig(device="cpu")
    depth_estimation.DepthEstimatorDA3(config)

    assert calls == [
        (
            depth_estimation.DEFAULT_DA3_MODEL_ID,
            {
                "revision": depth_estimation.DEFAULT_DA3_MODEL_REVISION,
                "local_files_only": False,
            },
        )
    ]


def test_grounding_dino_passes_pinned_revision(monkeypatch):
    calls: list[tuple[str, str, dict]] = []

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls.append(("processor", model_id, kwargs))
            return object()

    class FakeDetectorModel:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls.append(("model", model_id, kwargs))
            return _FakeModel()

    monkeypatch.setattr(
        object_detection,
        "_import_grounding_dino_components",
        lambda: (FakeDetectorModel, FakeProcessor),
    )
    config = object_detection.ObjectDetectionConfig(device="cpu")
    object_detection.GroundingDinoDetector(config)

    expected_kwargs = {
        "revision": object_detection.DEFAULT_GROUNDING_DINO_MODEL_REVISION,
        "local_files_only": False,
    }
    assert calls == [
        ("processor", object_detection.DEFAULT_GROUNDING_DINO_MODEL_ID, expected_kwargs),
        ("model", object_detection.DEFAULT_GROUNDING_DINO_MODEL_ID, expected_kwargs),
    ]


def test_sam3_huggingface_revision_alias_remains_pinned():
    assert (
        grounded_segmentation.DEFAULT_SAM3_MODEL_REVISION
        == grounded_segmentation.DEFAULT_SAM3_HUGGINGFACE_REVISION
    )
