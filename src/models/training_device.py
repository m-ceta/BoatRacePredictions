from __future__ import annotations

from typing import Any

import lightgbm as lgb


def get_gpu_settings(config: dict[str, Any] | None) -> dict[str, Any]:
    model_config = (config or {}).get("model", {})
    gpu_settings = model_config.get("gpu", {})
    training_device = str(model_config.get("training_device", "cpu")).lower()
    settings = {
        "training_device": training_device,
        "catboost": bool(gpu_settings.get("catboost", training_device == "gpu")),
        "lightgbm": bool(gpu_settings.get("lightgbm", training_device == "gpu")),
        "device_id": gpu_settings.get("device_id"),
        "platform_id": gpu_settings.get("platform_id"),
    }
    return settings


def catboost_training_kwargs(config: dict[str, Any] | None) -> dict[str, Any]:
    settings = get_gpu_settings(config)
    if not settings["catboost"]:
        return {}

    kwargs: dict[str, Any] = {"task_type": "GPU"}
    device_id = settings.get("device_id")
    if device_id is not None:
        kwargs["devices"] = str(device_id)
    return kwargs


def apply_lightgbm_training_device(
    params: dict[str, Any],
    config: dict[str, Any] | None,
) -> dict[str, Any]:
    settings = get_gpu_settings(config)
    updated = dict(params)
    if not settings["lightgbm"]:
        return updated

    updated["device_type"] = "gpu"
    updated.setdefault("gpu_use_dp", False)

    device_id = settings.get("device_id")
    platform_id = settings.get("platform_id")
    if device_id is not None:
        updated["gpu_device_id"] = int(device_id)
    if platform_id is not None:
        updated["gpu_platform_id"] = int(platform_id)
    return updated


def strip_lightgbm_gpu_params(params: dict[str, Any]) -> dict[str, Any]:
    updated = dict(params)
    for key in ("device_type", "device", "gpu_use_dp", "gpu_device_id", "gpu_platform_id"):
        updated.pop(key, None)
    return updated


def train_lightgbm_with_optional_gpu(
    params: dict[str, Any],
    train_set: lgb.Dataset,
    config: dict[str, Any] | None,
    **train_kwargs: Any,
) -> lgb.Booster:
    effective_params = apply_lightgbm_training_device(params, config)
    try:
        return lgb.train(effective_params, train_set, **train_kwargs)
    except Exception as exc:
        settings = get_gpu_settings(config)
        if not settings["lightgbm"]:
            raise
        fallback_params = strip_lightgbm_gpu_params(params)
        print(f"LightGBM GPU training failed; falling back to CPU. Reason: {exc}")
        return lgb.train(fallback_params, train_set, **train_kwargs)
