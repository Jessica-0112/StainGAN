from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List


def select_patient_keys_by_prefix(keys: List[str], patient_id: str) -> List[str]:
    """
    Fast path:
    LMDB keys are expected to start with patient_id, for example:
        S2224769_CK7_L1_001792_012544_xxxxx
    """
    prefixes = [
        patient_id + "_",
        patient_id + "-",
        patient_id + ".",
    ]

    selected = []
    for key in keys:
        if any(key.startswith(prefix) for prefix in prefixes):
            selected.append(key)

    return selected


def make_patient_split(
    src_split_json: Path,
    out_json: Path,
    fold: str,
    split: str,
    patient_id: str,
) -> None:
    with src_split_json.open("r", encoding="utf-8") as f:
        split_info = json.load(f)

    if "folds" not in split_info:
        raise KeyError("Cannot find 'folds' in split JSON.")

    if fold not in split_info["folds"]:
        raise KeyError(
            f"Cannot find fold '{fold}' in split JSON. "
            f"Available folds: {list(split_info['folds'].keys())}"
        )

    fold_data = split_info["folds"][fold]
    key_field = f"{split}_keys"

    if key_field not in fold_data:
        raise KeyError(
            f"Cannot find '{key_field}' in {fold}. "
            f"Available fields: {list(fold_data.keys())}"
        )

    keys = list(fold_data[key_field])
    patient_keys = select_patient_keys_by_prefix(keys, patient_id)

    if len(patient_keys) == 0:
        raise RuntimeError(
            f"No keys found for patient_id='{patient_id}' in {fold}/{split}.\n"
            "This fast script assumes LMDB keys start with the patient ID, e.g. S2224769_...\n"
            "If your keys do not start with patient_id, use the slower LMDB-meta based script instead."
        )

    new_split = {
        "source_split_json": str(src_split_json),
        "patient_id": patient_id,
        "fold": fold,
        "split": split,
        "selection": {
            "mode": "patient_all_by_key_prefix",
            "num_selected_keys": len(patient_keys),
        },
        "folds": {
            fold: {
                "train": [],
                "val": [patient_id] if split == "val" else [],
                "test": [patient_id] if split == "test" else [],
                "train_keys": [],
                "val_keys": patient_keys if split == "val" else [],
                "test_keys": patient_keys if split == "test" else [],
                "counts": {
                    f"{split}_patches": len(patient_keys),
                },
            }
        },
    }

    if split == "train":
        new_split["folds"][fold]["train"] = [patient_id]
        new_split["folds"][fold]["train_keys"] = patient_keys
        new_split["folds"][fold]["val"] = []
        new_split["folds"][fold]["val_keys"] = []

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(new_split, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 80)
    print("[DONE] Fast patient-level split saved")
    print(f"source split_json : {src_split_json}")
    print(f"out_json          : {out_json}")
    print(f"fold              : {fold}")
    print(f"split             : {split}")
    print(f"patient_id        : {patient_id}")
    print(f"selected keys     : {len(patient_keys)}")
    print("=" * 80)
    print("[First 5 keys]")
    for key in patient_keys[:5]:
        print(key)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fast patient-level split generator using key prefix. "
            "This does not read LMDB records, so it is much faster."
        )
    )
    parser.add_argument("--split-json", required=True, help="Original split JSON.")
    parser.add_argument("--fold", default="fold0")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--patient-id", required=True)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    make_patient_split(
        src_split_json=Path(args.split_json),
        out_json=Path(args.out_json),
        fold=args.fold,
        split=args.split,
        patient_id=args.patient_id,
    )


if __name__ == "__main__":
    main()
