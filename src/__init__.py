"""Boat race prediction package."""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = [
    "BoatRaceModelBundle",
    "backfill_rowdata_files",
    "build_dataset_from_rowdata",
    "build_dataset_from_rowdata_streaming",
    "load_bundle",
    "load_prediction_input",
    "predict_today",
    "predict_ranking",
    "predict_trifecta",
    "train_from_config",
]

if TYPE_CHECKING:
    from src.api import (
        BoatRaceModelBundle,
        backfill_rowdata_files,
        build_dataset_from_rowdata,
        build_dataset_from_rowdata_streaming,
        load_bundle,
        load_prediction_input,
        predict_ranking,
        predict_today,
        predict_trifecta,
        train_from_config,
    )


def __getattr__(name: str):
    if name in __all__:
        from src import api as _api

        return getattr(_api, name)
    raise AttributeError(f"module 'src' has no attribute {name!r}")
