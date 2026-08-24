from __future__ import annotations

import numpy as np
import pytest
import torch
from gda.datatypes import DetectionResult
from gda.modules.object_segmentation import Sam2BoxSegmentor


def test_normalize_single_box_multimask_output() -> None:
    masks = np.zeros((3, 5, 7), dtype=np.float32)
    masks[2, 1:4, 2:6] = 1
    scores = np.asarray([0.2, 0.4, 0.9], dtype=np.float32)

    normalized_masks, normalized_scores = Sam2BoxSegmentor._normalize_predictor_output(
        masks=masks,
        scores=scores,
        num_boxes=1,
        image_size=(5, 7),
    )

    assert normalized_masks.shape == (1, 3, 5, 7)
    assert normalized_masks.dtype == np.bool_
    assert normalized_scores.shape == (1, 3)
    best = np.argmax(normalized_scores, axis=1)
    np.testing.assert_array_equal(best, [2])


def test_normalize_bfloat16_scores() -> None:
    masks, scores = Sam2BoxSegmentor._normalize_predictor_output(
        masks=np.zeros((1, 1, 5, 7), dtype=bool),
        scores=torch.tensor([[0.5]], dtype=torch.bfloat16),
        num_boxes=1,
        image_size=(5, 7),
    )
    assert scores.dtype == np.float32
    np.testing.assert_allclose(scores, [[0.5]], rtol=1e-2)


def test_normalize_multiple_boxes_single_mask_output() -> None:
    masks = np.zeros((2, 5, 7), dtype=bool)
    scores = np.asarray([0.7, 0.8], dtype=np.float32)

    normalized_masks, normalized_scores = Sam2BoxSegmentor._normalize_predictor_output(
        masks=masks,
        scores=scores,
        num_boxes=2,
        image_size=(5, 7),
    )

    assert normalized_masks.shape == (2, 1, 5, 7)
    assert normalized_scores.shape == (2, 1)


def test_normalize_rejects_inconsistent_candidate_counts() -> None:
    with pytest.raises(RuntimeError, match="inconsistent masks and scores"):
        Sam2BoxSegmentor._normalize_predictor_output(
            masks=np.zeros((1, 3, 5, 7), dtype=bool),
            scores=np.asarray([[0.5]], dtype=np.float32),
            num_boxes=1,
            image_size=(5, 7),
        )


def test_segment_rejects_detection_from_another_image() -> None:
    segmentor = Sam2BoxSegmentor.__new__(Sam2BoxSegmentor)
    segmentor.config = type("Config", (), {"sam2_multimask_output": False})()
    det = DetectionResult(
        image_size=(4, 4),
        prompts=["car"],
        boxes_xyxy=np.asarray([[0, 0, 2, 2]], dtype=np.float32),
        scores=np.asarray([0.9], dtype=np.float32),
        prompt_ids=np.asarray([0], dtype=np.int32),
        labels=["car"],
    )
    with pytest.raises(ValueError, match="image_size does not match"):
        segmentor.segment(np.zeros((5, 5, 3), dtype=np.uint8), det)
