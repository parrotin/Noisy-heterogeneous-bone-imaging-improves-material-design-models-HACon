from __future__ import annotations

import argparse
from pathlib import Path

from hacon.data.records import VolumeRecord, assign_site_indices, filter_metadata, write_manifest
from hacon.data.volume_io import dicom_metadata, load_volume


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(prog="hacon-prepare")
    value.add_argument("--root", required=True)
    value.add_argument("--output", required=True)
    value.add_argument("--dataset", required=True)
    value.add_argument("--site", required=True)
    value.add_argument("--split", default="train")
    value.add_argument("--maximum-missing", type=float, default=0.30)
    return value


def discover_volumes(root: Path) -> list[Path]:
    nifti = sorted(root.rglob("*.nii")) + sorted(root.rglob("*.nii.gz"))
    dicom_directories = sorted(
        {
            path.parent
            for path in root.rglob("*.dcm")
        }
    )
    return nifti + dicom_directories


def record_for_path(
    path: Path,
    dataset: str,
    site: str,
    split: str,
) -> VolumeRecord:
    loaded = load_volume(path)
    metadata: dict[str, str | float] = {}
    if path.is_dir():
        metadata = dicom_metadata(path)
    return VolumeRecord(
        image=str(path),
        label=None,
        site=site,
        site_index=0,
        region=0,
        manufacturer=str(metadata.get("manufacturer", "")),
        kernel=str(metadata.get("kernel", "")),
        slice_thickness=float(metadata.get("slice_thickness", loaded.spacing[2])),
        spacing=loaded.spacing,
        split=split,
        dataset=dataset,
    )


def main() -> None:
    arguments = parser().parse_args()
    root = Path(arguments.root)
    records = [
        record_for_path(path, arguments.dataset, arguments.site, arguments.split)
        for path in discover_volumes(root)
    ]
    records = assign_site_indices(records)
    records = filter_metadata(records, arguments.maximum_missing)
    write_manifest(arguments.output, records)


if __name__ == "__main__":
    main()

