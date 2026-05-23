from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set

import lmdb
import msgpack


META_KEYS = {
    b"__len__",
    b"__keys__",
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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

    candidate_keys = [
        "patient_id",
        "patient",
        "case_id",
        "case",
        "slide_id",
    ]

    for key in candidate_keys:
        if key in meta and meta[key] not in [None, ""]:
            return str(meta[key])

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


def collect_patient_to_keys(src_lmdb: Path) -> Dict[str, List[bytes]]:
    env = lmdb.open(
        str(src_lmdb),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=2048,
    )

    keys = load_keys(env)
    patient_to_keys: Dict[str, List[bytes]] = defaultdict(list)

    print(f"[INFO] Total keys from LMDB: {len(keys)}")
    print("[INFO] Collecting patient_id for each key...")

    with env.begin(write=False) as txn:
        for idx, key in enumerate(keys, start=1):
            value = txn.get(key)
            if value is None:
                continue

            record = msgpack.unpackb(value, raw=False)
            patient_id = get_patient_id(record)
            patient_to_keys[patient_id].append(key)

            if idx % 100000 == 0:
                print(f"  processed {idx}/{len(keys)} keys")

    env.close()

    return dict(patient_to_keys)


def make_fixed_test_and_5fold(
    patients: List[str],
    test_n: int,
    n_folds: int,
    seed: int,
) -> dict:
    if test_n <= 0:
        raise ValueError("test_n must be > 0")

    if n_folds <= 1:
        raise ValueError("n_folds must be > 1")

    if len(patients) <= test_n:
        raise ValueError("Number of patients must be larger than test_n")

    rng = random.Random(seed)
    patients = list(patients)
    rng.shuffle(patients)

    test_patients = sorted(patients[:test_n])
    trainval_patients = patients[test_n:]

    folds_val: List[List[str]] = [[] for _ in range(n_folds)]

    for i, patient_id in enumerate(trainval_patients):
        folds_val[i % n_folds].append(patient_id)

    folds = {}

    trainval_set = set(trainval_patients)

    for fold_idx in range(n_folds):
        val_patients = sorted(folds_val[fold_idx])
        val_set = set(val_patients)
        train_patients = sorted(list(trainval_set - val_set))

        folds[f"fold{fold_idx}"] = {
            "train": train_patients,
            "val": val_patients,
            "test": test_patients,
            "counts": {
                "train": len(train_patients),
                "val": len(val_patients),
                "test": len(test_patients),
            },
        }

    split_info = {
        "seed": seed,
        "n_total_patients": len(patients),
        "n_test_patients": len(test_patients),
        "n_trainval_patients": len(trainval_patients),
        "n_folds": n_folds,
        "test_patients": test_patients,
        "folds": folds,
    }

    return split_info


def write_lmdb_subset(
    src_lmdb: Path,
    out_lmdb: Path,
    selected_keys: Set[bytes],
    commit_every: int = 2000,
    map_size: int = 1024 ** 4,
) -> int:
    ensure_dir(out_lmdb)

    src_env = lmdb.open(
        str(src_lmdb),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=2048,
    )

    dst_env = lmdb.open(
        str(out_lmdb),
        map_size=map_size,
        subdir=True,
        meminit=False,
        map_non_blocking=True,
        writemap=False,
        metasync=False,
        sync=False,
    )

    selected_keys_list = sorted(selected_keys)
    written = 0
    written_keys: List[str] = []

    txn = dst_env.begin(write=True)

    try:
        with src_env.begin(write=False) as src_txn:
            for idx, key in enumerate(selected_keys_list, start=1):
                value = src_txn.get(key)
                if value is None:
                    continue

                ok = txn.put(key, value, overwrite=False)
                if ok:
                    written += 1
                    written_keys.append(key.decode("utf-8", errors="ignore"))

                if idx % commit_every == 0:
                    txn.commit()
                    txn = dst_env.begin(write=True)

        txn.put(b"__len__", str(written).encode("utf-8"))
        txn.put(b"__keys__", msgpack.packb(written_keys, use_bin_type=True))
        txn.commit()

    except Exception:
        txn.abort()
        raise

    finally:
        dst_env.sync()
        dst_env.close()
        src_env.close()

    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create fixed-test + 5-fold patient-level LMDB splits."
    )

    parser.add_argument("--src-lmdb", required=True, help="Path to merged LMDB.")
    parser.add_argument("--out-dir", required=True, help="Output root directory.")
    parser.add_argument("--prefix", default="ck7", help="Output prefix, e.g., ck7, ttf1, evg.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-n", type=int, default=8)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--commit-every", type=int, default=2000)
    parser.add_argument("--map-size", type=int, default=1024 ** 4)

    args = parser.parse_args()

    src_lmdb = Path(args.src_lmdb)
    out_root = Path(args.out_dir) / f"seed{args.seed}"
    ensure_dir(out_root)

    patient_to_keys = collect_patient_to_keys(src_lmdb)
    patients = sorted(patient_to_keys.keys())

    print("=" * 80)
    print("[PATIENT SUMMARY]")
    print(f"Number of patients: {len(patients)}")
    print("Patients:")
    for p in patients:
        print(f"  {p}: {len(patient_to_keys[p])} patches")
    print("=" * 80)

    split_info = make_fixed_test_and_5fold(
        patients=patients,
        test_n=args.test_n,
        n_folds=args.n_folds,
        seed=args.seed,
    )

    split_json_path = out_root / f"{args.prefix}_patient_split_5fold_56patients.json"
    split_json_path.write_text(
        json.dumps(split_info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("[SPLIT INFO]")
    print(json.dumps(
        {
            "seed": split_info["seed"],
            "n_total_patients": split_info["n_total_patients"],
            "n_test_patients": split_info["n_test_patients"],
            "n_trainval_patients": split_info["n_trainval_patients"],
            "n_folds": split_info["n_folds"],
        },
        ensure_ascii=False,
        indent=2,
    ))

    for fold_name, fold_data in split_info["folds"].items():
        fold_dir = out_root / fold_name
        ensure_dir(fold_dir)

        print("=" * 80)
        print(f"[WRITE {fold_name}]")
        print(f"train patients: {fold_data['counts']['train']}")
        print(f"val patients  : {fold_data['counts']['val']}")
        print(f"test patients : {fold_data['counts']['test']}")

        for split_name in ["train", "val", "test"]:
            split_patients = fold_data[split_name]

            selected_keys: Set[bytes] = set()
            for patient_id in split_patients:
                selected_keys.update(patient_to_keys[patient_id])

            out_lmdb = fold_dir / f"{args.prefix}_{split_name}.lmdb"

            written = write_lmdb_subset(
                src_lmdb=src_lmdb,
                out_lmdb=out_lmdb,
                selected_keys=selected_keys,
                commit_every=args.commit_every,
                map_size=args.map_size,
            )

            print(f"  [{split_name}] patients={len(split_patients)}, patches={written}, out={out_lmdb}")

    print("=" * 80)
    print(f"[DONE] split json saved to: {split_json_path}")


if __name__ == "__main__":
    main()