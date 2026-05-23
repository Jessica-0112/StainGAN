from __future__ import annotations

import argparse
import io
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import lmdb
import msgpack
from PIL import Image


META_KEYS = {
    b"__len__",
    b"__keys__",
}


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


def get_patient_id(meta: Dict[str, Any]) -> str:
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

    return "UNKNOWN_PATIENT"


def png_bytes_to_image(png_bytes: bytes) -> Image.Image:
    if not isinstance(png_bytes, (bytes, bytearray)):
        raise TypeError(f"image field is not bytes, got {type(png_bytes)}")

    if len(png_bytes) == 0:
        raise ValueError("image bytes is empty")

    return Image.open(io.BytesIO(png_bytes)).convert("RGB")


def load_all_keys(env) -> List[bytes]:
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check merged LMDB format for StainGAN."
    )
    parser.add_argument("--lmdb-path", required=True)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--save-preview-dir", type=str, default="")
    args = parser.parse_args()

    lmdb_path = Path(args.lmdb_path)
    if not lmdb_path.exists():
        raise FileNotFoundError(f"LMDB path not found: {lmdb_path}")

    env = lmdb.open(
        str(lmdb_path),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=2048,
    )

    with env.begin(write=False) as txn:
        raw_len = txn.get(b"__len__")
        raw_keys = txn.get(b"__keys__")
        stored_len = raw_len.decode("utf-8") if raw_len is not None else None
        has_keys = raw_keys is not None

    keys = load_all_keys(env)
    num_records = len(keys)

    print("=" * 80)
    print("[LMDB BASIC INFO]")
    print(f"LMDB path      : {lmdb_path}")
    print(f"__len__        : {stored_len}")
    print(f"has __keys__   : {has_keys}")
    print(f"number of keys : {num_records}")
    print("=" * 80)

    if num_records == 0:
        raise RuntimeError("No valid records found in LMDB.")

    preview_dir = None
    if args.save_preview_dir:
        preview_dir = Path(args.save_preview_dir)
        preview_dir.mkdir(parents=True, exist_ok=True)

    field_counter = Counter()
    patient_counter = Counter()
    input_size_counter = Counter()
    target_size_counter = Counter()
    meta_key_counter = Counter()

    checked = 0
    failed = 0
    max_check = min(args.num_samples, num_records)

    print("\n[CHECK SAMPLES]")

    with env.begin(write=False) as txn:
        for i, key in enumerate(keys[:max_check]):
            value = txn.get(key)
            if value is None:
                print(f"[ERROR] key not found: {key!r}")
                failed += 1
                continue

            try:
                record = msgpack.unpackb(value, raw=False)

                if not isinstance(record, dict):
                    raise TypeError(f"record is not dict, got {type(record)}")

                field_counter.update(record.keys())

                for field in ["input", "target", "meta"]:
                    if field not in record:
                        raise KeyError(f"missing required field: {field}")

                input_img = png_bytes_to_image(record["input"])
                target_img = png_bytes_to_image(record["target"])

                meta = decode_meta(record["meta"])
                patient_id = get_patient_id(meta)

                patient_counter[patient_id] += 1
                input_size_counter[input_img.size] += 1
                target_size_counter[target_img.size] += 1
                meta_key_counter.update(meta.keys())

                key_str = key.decode("utf-8", errors="ignore")

                print("-" * 80)
                print(f"[{i}] key        : {key_str}")
                print(f"    patient_id : {patient_id}")
                print(f"    input size : {input_img.size}")
                print(f"    target size: {target_img.size}")
                print(f"    record keys: {list(record.keys())}")
                print(f"    meta keys  : {list(meta.keys())}")

                if preview_dir is not None:
                    safe_key = key_str.replace("/", "_").replace("\\", "_").replace(":", "_")
                    input_img.save(preview_dir / f"{i:04d}_{safe_key}_input_HE.png")
                    target_img.save(preview_dir / f"{i:04d}_{safe_key}_target.png")

                checked += 1

            except Exception as e:
                print("-" * 80)
                print(f"[ERROR] failed to read key={key!r}")
                print(f"        error={repr(e)}")
                failed += 1

    print("\n" + "=" * 80)
    print("[SUMMARY]")
    print(f"checked samples : {checked}")
    print(f"failed samples  : {failed}")
    print(f"unique patients in checked samples: {len(patient_counter)}")

    print("\nrecord field counts:")
    for k, v in field_counter.most_common():
        print(f"  {k}: {v}")

    print("\ninput image sizes:")
    for k, v in input_size_counter.most_common():
        print(f"  {k}: {v}")

    print("\ntarget image sizes:")
    for k, v in target_size_counter.most_common():
        print(f"  {k}: {v}")

    print("\nmeta key counts:")
    for k, v in meta_key_counter.most_common():
        print(f"  {k}: {v}")

    if preview_dir is not None:
        print(f"\npreview images saved to: {preview_dir}")

    print("=" * 80)

    env.close()


if __name__ == "__main__":
    main()