"""Compatibility entry point for the former :mod:`gda.gda` module.

When Python is launched from the repository directory, this file can otherwise
shadow the package directory. Exposing ``__path__`` keeps ``python -m
gda.modules.<name>`` working in that common Pixi workflow.
"""

from pathlib import Path

if __name__ == "gda":
    __path__ = [str(Path(__file__).resolve().parent)]  # type: ignore[assignment]

from gda.pipeline import GDAArgs, ImageDepthAndSegPipeline, PipelineConfig, main

__all__ = ["GDAArgs", "ImageDepthAndSegPipeline", "PipelineConfig", "main"]


if __name__ == "__main__":
    main()
