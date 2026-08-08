from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def _serialize(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _serialize(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(item) for item in value]
    if hasattr(value, "detach"):
        tensor = value.detach().cpu()
        if tensor.numel() == 1:
            return float(tensor.item())
        return tensor.tolist()
    return value


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(_serialize(value), indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(destination)


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("CSV output requires at least one row")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows([{key: _serialize(value) for key, value in row.items()} for row in rows])
    temporary.replace(destination)


def format_mean_standard_deviation(mean: float, standard_deviation: float, digits: int = 3) -> str:
    return f"{mean:.{digits}f} ± {standard_deviation:.{digits}f}"


def format_p_value(value: float) -> str:
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def format_effect_size(value: float) -> str:
    return f"{value:.2f}"


def segmentation_table_row(
    method: str,
    dice_mean: float,
    dice_standard_deviation: float,
    auc_mean: float,
    auc_standard_deviation: float,
    hd95_mean: float,
    hd95_standard_deviation: float,
    gap: float,
) -> dict[str, str | float]:
    return {
        "method": method,
        "dice": format_mean_standard_deviation(dice_mean, dice_standard_deviation),
        "auc": format_mean_standard_deviation(auc_mean, auc_standard_deviation),
        "hd95": format_mean_standard_deviation(hd95_mean, hd95_standard_deviation, 2),
        "gap_pp": gap * 100.0,
    }


def ablation_table_row(
    configuration: str,
    cspm: bool,
    smt: bool,
    dice_mean: float,
    dice_standard_deviation: float,
    auc_mean: float,
    auc_standard_deviation: float,
    delta: float,
) -> dict[str, Any]:
    return {
        "configuration": configuration,
        "cspm": cspm,
        "smt": smt,
        "dice": format_mean_standard_deviation(dice_mean, dice_standard_deviation),
        "auc": format_mean_standard_deviation(auc_mean, auc_standard_deviation),
        "delta_pp": delta * 100.0,
    }


def material_table_row(
    feature: str,
    target: str,
    r2_mean: float,
    r2_standard_deviation: float,
    relative_error_mean: float,
    relative_error_standard_deviation: float,
) -> dict[str, Any]:
    return {
        "features": feature,
        "target": target,
        "r2": format_mean_standard_deviation(r2_mean, r2_standard_deviation),
        "relative_error_percent": format_mean_standard_deviation(
            relative_error_mean,
            relative_error_standard_deviation,
            2,
        ),
    }


def compute_table_row(
    method: str,
    volumes: int,
    epochs: int,
    gpus: str,
    gpu_hours: float,
    parameters_millions: float,
) -> dict[str, Any]:
    return {
        "method": method,
        "volumes": volumes,
        "epochs": epochs,
        "gpus": gpus,
        "gpu_hours": gpu_hours,
        "parameters_millions": parameters_millions,
    }


def subgroup_table_row(
    group: str,
    count: int,
    dice_mean: float,
    dice_standard_deviation: float,
    gap_from_overall: float,
) -> dict[str, Any]:
    return {
        "group": group,
        "count": count,
        "dice": format_mean_standard_deviation(dice_mean, dice_standard_deviation),
        "gap_pp": gap_from_overall * 100.0,
    }

