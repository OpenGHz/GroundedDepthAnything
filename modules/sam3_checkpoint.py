"""Resolve and verify the pinned SAM3 image checkpoint."""

from __future__ import annotations

import fcntl
import hashlib
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from gda.modules.workspace import cache_root

SAM3_MODEL_ID = "facebook/sam3"
SAM3_CHECKPOINT_FILENAME = "sam3.pt"
SAM3_CHECKPOINT_SIZE = 3_450_062_241
SAM3_CHECKPOINT_SHA256 = "9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e"
DEFAULT_SAM3_MODELSCOPE_REVISION = "96f3e1b404ba14f2cfac60ee6ae87c269a7b7923"
DEFAULT_SAM3_HUGGINGFACE_REVISION = "3c879f39826c281e95690f02c7821c4de09afae7"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sam3_checkpoint(path: Path) -> Path:
    """Verify that ``path`` is the exact pinned SAM3 image checkpoint."""

    if not path.is_file():
        raise FileNotFoundError(f"SAM3 checkpoint not found after download: {path}")
    actual_size = path.stat().st_size
    if actual_size != SAM3_CHECKPOINT_SIZE:
        raise RuntimeError(
            f"SAM3 checkpoint size mismatch at {path}: "
            f"expected {SAM3_CHECKPOINT_SIZE}, found {actual_size}"
        )
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != SAM3_CHECKPOINT_SHA256:
        raise RuntimeError(
            f"SAM3 checkpoint SHA256 mismatch at {path}: "
            f"expected {SAM3_CHECKPOINT_SHA256}, found {actual_sha256}"
        )
    return path


def default_modelscope_checkpoint_path() -> Path:
    """Return the content-addressed cache path used by the default provider."""

    return (
        cache_root() / "checkpoints" / "sam3" / SAM3_CHECKPOINT_SHA256 / SAM3_CHECKPOINT_FILENAME
    )


@contextmanager
def _checkpoint_lock(checkpoint: Path) -> Iterator[None]:
    """Serialize writers that share the same GDA checkpoint cache."""

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    lock_path = checkpoint.with_name(f".{checkpoint.name}.lock")
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _download_from_modelscope(revision: str, *, local_files_only: bool) -> Path:
    checkpoint = default_modelscope_checkpoint_path()
    with _checkpoint_lock(checkpoint):
        if checkpoint.exists():
            return verify_sam3_checkpoint(checkpoint)
        if local_files_only:
            raise FileNotFoundError(
                "SAM3 is not present in the GDA ModelScope cache and network access is "
                f"disabled: {checkpoint}"
            )

        from modelscope.hub.file_download import model_file_download

        downloaded = (
            Path(
                model_file_download(
                    model_id=SAM3_MODEL_ID,
                    file_path=SAM3_CHECKPOINT_FILENAME,
                    revision=revision,
                    local_dir=str(checkpoint.parent),
                )
            )
            .expanduser()
            .resolve()
        )
        if downloaded != checkpoint:
            raise RuntimeError(
                "ModelScope returned an unexpected SAM3 checkpoint path: "
                f"expected {checkpoint}, found {downloaded}"
            )
        return verify_sam3_checkpoint(checkpoint)


def _download_from_huggingface(revision: str, *, local_files_only: bool) -> Path:
    from huggingface_hub import hf_hub_download

    checkpoint = Path(
        hf_hub_download(
            repo_id=SAM3_MODEL_ID,
            filename=SAM3_CHECKPOINT_FILENAME,
            revision=revision,
            local_files_only=local_files_only,
        )
    )
    return verify_sam3_checkpoint(checkpoint)


def download_sam3_checkpoint(
    *,
    load_from_hf: bool = False,
    modelscope_revision: str = DEFAULT_SAM3_MODELSCOPE_REVISION,
    huggingface_revision: str = DEFAULT_SAM3_HUGGINGFACE_REVISION,
    local_files_only: bool = False,
) -> Path:
    """Download and verify SAM3 from ModelScope by default or Hugging Face explicitly."""

    if load_from_hf:
        offline_values = {"1", "on", "true", "yes"}
        hf_offline = any(
            os.environ.get(name, "").strip().lower() in offline_values
            for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
        )
        return _download_from_huggingface(
            huggingface_revision,
            local_files_only=local_files_only or hf_offline,
        )
    return _download_from_modelscope(
        modelscope_revision,
        local_files_only=local_files_only,
    )
