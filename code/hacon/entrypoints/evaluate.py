from __future__ import annotations

import argparse
from pathlib import Path

import torch

from hacon.evaluation.statistics import paired_t_test, summarize
from hacon.reporting.tables import write_json


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(prog="hacon-evaluate")
    value.add_argument("--predictions", required=True)
    value.add_argument("--reference", default=None)
    value.add_argument("--output", default="runs/evaluation/summary.json")
    return value


def main() -> None:
    arguments = parser().parse_args()
    predictions = torch.load(arguments.predictions, map_location="cpu")
    if isinstance(predictions, dict):
        values = torch.as_tensor(predictions["values"])
    else:
        values = torch.as_tensor(predictions)
    result: dict[str, object] = {"summary": summarize(values)}
    if arguments.reference is not None:
        reference_data = torch.load(arguments.reference, map_location="cpu")
        if isinstance(reference_data, dict):
            reference = torch.as_tensor(reference_data["values"])
        else:
            reference = torch.as_tensor(reference_data)
        result["comparison"] = paired_t_test(values, reference)
    write_json(Path(arguments.output), result)


if __name__ == "__main__":
    main()

