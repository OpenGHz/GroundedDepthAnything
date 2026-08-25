from __future__ import annotations

import hashlib
import importlib
from pathlib import Path

import pytest

from gda.modules import sam3_checkpoint

modelscope_file_download = importlib.import_module("modelscope.hub.file_download")


@pytest.fixture
def pinned_test_checkpoint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> bytes:
    content = b"pinned sam3 test checkpoint"
    monkeypatch.setattr(sam3_checkpoint, "SAM3_CHECKPOINT_SIZE", len(content))
    monkeypatch.setattr(
        sam3_checkpoint,
        "SAM3_CHECKPOINT_SHA256",
        hashlib.sha256(content).hexdigest(),
    )
    monkeypatch.setenv("GDA_CACHE_DIR", str(tmp_path / "gda-cache"))
    return content


def test_modelscope_is_default_and_uses_exact_revision(
    monkeypatch: pytest.MonkeyPatch,
    pinned_test_checkpoint: bytes,
) -> None:
    calls: list[dict] = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        checkpoint = Path(kwargs["local_dir"]) / kwargs["file_path"]
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(pinned_test_checkpoint)
        return str(checkpoint)

    monkeypatch.setattr(modelscope_file_download, "model_file_download", fake_download)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    checkpoint = sam3_checkpoint.download_sam3_checkpoint()

    assert checkpoint == sam3_checkpoint.default_modelscope_checkpoint_path()
    assert calls == [
        {
            "model_id": sam3_checkpoint.SAM3_MODEL_ID,
            "file_path": sam3_checkpoint.SAM3_CHECKPOINT_FILENAME,
            "revision": sam3_checkpoint.DEFAULT_SAM3_MODELSCOPE_REVISION,
            "local_dir": str(checkpoint.parent),
        }
    ]


def test_modelscope_cache_is_reused_without_network(
    monkeypatch: pytest.MonkeyPatch,
    pinned_test_checkpoint: bytes,
) -> None:
    checkpoint = sam3_checkpoint.default_modelscope_checkpoint_path()
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(pinned_test_checkpoint)
    monkeypatch.setattr(
        modelscope_file_download,
        "model_file_download",
        lambda **kwargs: pytest.fail(f"unexpected download: {kwargs}"),
    )

    assert sam3_checkpoint.download_sam3_checkpoint(local_files_only=True) == checkpoint


def test_modelscope_offline_cache_miss_fails_without_network(
    monkeypatch: pytest.MonkeyPatch,
    pinned_test_checkpoint: bytes,
) -> None:
    monkeypatch.setattr(
        modelscope_file_download,
        "model_file_download",
        lambda **kwargs: pytest.fail(f"unexpected download: {kwargs}"),
    )

    with pytest.raises(FileNotFoundError, match="network access is disabled"):
        sam3_checkpoint.download_sam3_checkpoint(local_files_only=True)


def test_modelscope_rejects_unexpected_download_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    pinned_test_checkpoint: bytes,
) -> None:
    def fake_download(**kwargs):
        unexpected = tmp_path / "unexpected" / kwargs["file_path"]
        unexpected.parent.mkdir(parents=True)
        unexpected.write_bytes(pinned_test_checkpoint)
        return str(unexpected)

    monkeypatch.setattr(modelscope_file_download, "model_file_download", fake_download)

    with pytest.raises(RuntimeError, match="unexpected SAM3 checkpoint path"):
        sam3_checkpoint.download_sam3_checkpoint()


def test_invalid_cached_modelscope_checkpoint_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    pinned_test_checkpoint: bytes,
) -> None:
    checkpoint = sam3_checkpoint.default_modelscope_checkpoint_path()
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"wrong")
    monkeypatch.setattr(
        modelscope_file_download,
        "model_file_download",
        lambda **kwargs: pytest.fail(f"must not overwrite invalid cache: {kwargs}"),
    )

    with pytest.raises(RuntimeError, match="size mismatch"):
        sam3_checkpoint.download_sam3_checkpoint()


def test_same_size_wrong_hash_is_rejected(
    pinned_test_checkpoint: bytes,
) -> None:
    checkpoint = sam3_checkpoint.default_modelscope_checkpoint_path()
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"x" * len(pinned_test_checkpoint))

    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        sam3_checkpoint.download_sam3_checkpoint()


def test_huggingface_is_explicit_and_uses_its_own_revision(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    pinned_test_checkpoint: bytes,
) -> None:
    checkpoint = tmp_path / "hf" / "sam3.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(pinned_test_checkpoint)
    calls: list[dict] = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return str(checkpoint)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    monkeypatch.setenv("HF_HUB_OFFLINE", "TRUE")
    result = sam3_checkpoint.download_sam3_checkpoint(load_from_hf=True)

    assert result == checkpoint
    assert calls == [
        {
            "repo_id": sam3_checkpoint.SAM3_MODEL_ID,
            "filename": sam3_checkpoint.SAM3_CHECKPOINT_FILENAME,
            "revision": sam3_checkpoint.DEFAULT_SAM3_HUGGINGFACE_REVISION,
            "local_files_only": True,
        }
    ]


def test_modelscope_failure_does_not_fall_back_to_huggingface(
    monkeypatch: pytest.MonkeyPatch,
    pinned_test_checkpoint: bytes,
) -> None:
    def fail_modelscope(**kwargs):
        raise ConnectionError(f"modelscope unavailable: {kwargs}")

    monkeypatch.setattr(modelscope_file_download, "model_file_download", fail_modelscope)
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        lambda **kwargs: pytest.fail(f"unexpected fallback: {kwargs}"),
    )

    with pytest.raises(ConnectionError, match="modelscope unavailable"):
        sam3_checkpoint.download_sam3_checkpoint()


def test_all_sam3_clis_default_to_modelscope() -> None:
    from gda import image_to_positions, pipeline
    from gda.modules import grounded_segmentation

    for args_type in (
        grounded_segmentation.GroundedSegmentationArgs,
        pipeline.GDAArgs,
        image_to_positions.ImageToPositionsArgs,
    ):
        assert args_type.model_fields["sam3_load_from_hf"].default is False
        assert (
            args_type.model_fields["sam3_modelscope_revision"].default
            == sam3_checkpoint.DEFAULT_SAM3_MODELSCOPE_REVISION
        )
        assert args_type.model_fields["sam3_local_files_only"].default is False
