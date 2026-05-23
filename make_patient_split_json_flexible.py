from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import lmdb
import msgpack


META_KEYS = {b"__len__", b"__keys__"}


def decode_meta(meta: Any) -> Dict[str, Any]:
    if isinstance(meta, dict):
        return meta

    if isinstance(meta, bytes):
        meta = meta.decode("utf-8", errors="ignore")

    if isinstance(meta, str):
        try:
            obj = json.loads(meta)
            if isinstance(obj, dict):
                return obj
        except Exception:
            return {"meta_raw": meta}

    return {"meta_raw": str(meta)}


def get_patient_id(record: dict) -> str:
    meta = decode_meta(record.get("meta", {}))

    for key in ["patient_id", "patient", "case_id", "case", "slide_id"]:
        value = meta.get(key, None)
        if value not in [None, ""]:
            return str(value)

    raise KeyError(f"Cannot find patient_id from meta keys: {list(meta.keys())}")


def load_keys(env) -> List[bytes]:
    with env.begin(write=False) as txn:
        packed_keys = txn.get(b"__keys__")
        if packed_keys is not None:
            keys = msgpack.unpackb(packed_keys, raw=False)
            return [
                k if isinstance(k, bytes) else str(k).encode("utf-8")
                for k in keys
            ]

        keys: List[bytes] = []
        with txn.cursor() as cursor:
            for key, _ in cursor:
                if key in META_KEYS:
                    continue
                if key.startswith(b"__"):
                    continue
                keys.append(key)

    return keys


def collect_patient_summary(src_lmdb: Path) -> Dict[str, dict]:
    env = lmdb.open(
        str(src_lmdb),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=2048,
    )

    all_keys = load_keys(env)

    patient_info: Dict[str, dict] = defaultdict(
        lambda: {
            "keys": [],
            "num_patches": 0,
            "slides": set(),
            "he_paths": set(),
            "target_paths": set(),
        }
    )

    print(f"[INFO] Total keys: {len(all_keys)}")
    print("[INFO] Reading LMDB meta and grouping keys by patient...")

    with env.begin(write=False) as txn:
        for idx, key in enumerate(all_keys, start=1):
            value = txn.get(key)
            if value is None:
                continue

            record = msgpack.unpackb(value, raw=False)
            meta = decode_meta(record.get("meta", {}))
            patient_id = get_patient_id(record)
            key_str = key.decode("utf-8", errors="ignore")

            patient_info[patient_id]["keys"].append(key_str)
            patient_info[patient_id]["num_patches"] += 1

            he_path = str(meta.get("he_path", ""))
            target_path = str(meta.get("target_path", ""))

            if he_path:
                patient_info[patient_id]["he_paths"].add(he_path)

            if target_path:
                patient_info[patient_id]["target_paths"].add(target_path)

            if target_path:
                patient_info[patient_id]["slides"].add(target_path)
            elif he_path:
                patient_info[patient_id]["slides"].add(he_path)

            if idx % 100000 == 0:
                print(f"  processed {idx}/{len(all_keys)} keys")

    env.close()

    output: Dict[str, dict] = {}
    for patient_id, info in patient_info.items():
        output[patient_id] = {
            "keys": sorted(info["keys"]),
            "num_patches": int(info["num_patches"]),
            "num_slides": len(info["slides"]),
            "slides": sorted(list(info["slides"])),
            "he_paths": sorted(list(info["he_paths"])),
            "target_paths": sorted(list(info["target_paths"])),
        }

    return output


def flatten_keys(patient_ids: List[str], patient_summary: Dict[str, dict]) -> List[str]:
    keys: List[str] = []
    for patient_id in patient_ids:
        keys.extend(patient_summary[patient_id]["keys"])
    return keys


def make_patient_to_keys(
    patient_ids: List[str],
    patient_summary: Dict[str, dict],
) -> Dict[str, List[str]]:
    return {
        patient_id: list(patient_summary[patient_id]["keys"])
        for patient_id in patient_ids
    }


def make_split(
    patient_summary: Dict[str, dict],
    seed: int,
    test_n: int,
    n_folds: int,
) -> dict:
    patients = sorted(patient_summary.keys())

    if test_n <= 0:
        raise ValueError("test_n must be > 0")
    if n_folds <= 1:
        raise ValueError("n_folds must be > 1")
    if len(patients) <= test_n:
        raise ValueError("Number of patients must be larger than test_n")

    rng = random.Random(seed)
    shuffled_patients = list(patients)
    rng.shuffle(shuffled_patients)

    test_patients = sorted(shuffled_patients[:test_n])
    trainval_patients = shuffled_patients[test_n:]

    folds_val: List[List[str]] = [[] for _ in range(n_folds)]
    for i, patient_id in enumerate(trainval_patients):
        folds_val[i % n_folds].append(patient_id)

    trainval_set = set(trainval_patients)
    test_keys = flatten_keys(test_patients, patient_summary)

    folds = {}

    for fold_idx in range(n_folds):
        val_patients = sorted(folds_val[fold_idx])
        val_set = set(val_patients)
        train_patients = sorted(list(trainval_set - val_set))

        train_keys = flatten_keys(train_patients, patient_summary)
        val_keys = flatten_keys(val_patients, patient_summary)

        fold_dict = {
            "train": train_patients,
            "val": val_patients,
            "test": test_patients,
            "train_keys": train_keys,
            "val_keys": val_keys,
            "test_keys": test_keys,
            "train_patient_to_keys": make_patient_to_keys(
                train_patients,
                patient_summary,
            ),
            "counts": {
                "train_patients": len(train_patients),
                "val_patients": len(val_patients),
                "test_patients": len(test_patients),
                "train_patches": len(train_keys),
                "val_patches": len(val_keys),
                "test_patches": len(test_keys),
                "train_slides": sum(
                    patient_summary[p]["num_slides"] for p in train_patients
                ),
                "val_slides": sum(
                    patient_summary[p]["num_slides"] for p in val_patients
                ),
                "test_slides": sum(
                    patient_summary[p]["num_slides"] for p in test_patients
                ),
            },
        }

        folds[f"fold{fold_idx}"] = fold_dict

    patient_summary_without_keys = {}
    for patient_id, info in patient_summary.items():
        patient_summary_without_keys[patient_id] = {
            "num_patches": info["num_patches"],
            "num_slides": info["num_slides"],
            "slides": info["slides"],
            "he_paths": info["he_paths"],
            "target_paths": info["target_paths"],
        }

    return {
        "seed": seed,
        "n_total_patients": len(patients),
        "n_test_patients": len(test_patients),
        "n_trainval_patients": len(trainval_patients),
        "n_folds": n_folds,
        "test_patients": test_patients,
        "patient_summary": patient_summary_without_keys,
        "folds": folds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create flexible patient-level split JSON with train_patient_to_keys."
    )
    parser.add_argument("--src-lmdb", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-n", type=int, default=10)
    parser.add_argument("--n-folds", type=int, default=5)
    args = parser.parse_args()

    src_lmdb = Path(args.src_lmdb)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    patient_summary = collect_patient_summary(src_lmdb)
    split_info = make_split(
        patient_summary=patient_summary,
        seed=args.seed,
        test_n=args.test_n,
        n_folds=args.n_folds,
    )
    split_info["src_lmdb"] = str(src_lmdb)

    out_json.write_text(
        json.dumps(split_info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 80)
    print("[DONE] Flexible split JSON saved")
    print(f"out_json: {out_json}")
    print(f"src_lmdb: {src_lmdb}")
    print(f"patients: {split_info['n_total_patients']}")
    print(f"test_n: {args.test_n}")
    print(f"n_folds: {args.n_folds}")
    print("=" * 80)

    for fold_name, fold_data in split_info["folds"].items():
        print(f"[{fold_name}]")
        print(f"  train patients: {fold_data['counts']['train_patients']}")
        print(f"  val patients  : {fold_data['counts']['val_patients']}")
        print(f"  test patients : {fold_data['counts']['test_patients']}")
        print(f"  train patches : {fold_data['counts']['train_patches']}")
        print(f"  val patches   : {fold_data['counts']['val_patches']}")
        print(f"  test patches  : {fold_data['counts']['test_patches']}")


if __name__ == "__main__":
    main()