from __future__ import annotations

import io
import json
import os
from typing import Any, Dict, List

import lmdb
import msgpack
from PIL import Image

from data.base_dataset import BaseDataset, get_transform


_META_KEYS = {
    b"__len__",
    b"__keys__",
}


def _png_bytes_to_pil_rgb(png_bytes: bytes) -> Image.Image:
    if png_bytes is None or len(png_bytes) == 0:
        raise ValueError("Empty image bytes in LMDB record.")

    return Image.open(io.BytesIO(png_bytes)).convert("RGB")


def _decode_meta(meta: Any) -> Dict[str, Any]:
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


def _get_patient_id_from_meta(meta: Dict[str, Any]) -> str:
    for key in ["patient_id", "patient", "case_id", "case", "slide_id"]:
        if key in meta and meta[key] not in [None, ""]:
            return str(meta[key])

    raise KeyError(f"Cannot find patient_id from meta keys: {list(meta.keys())}")


class LMDBSingleDataset(BaseDataset):
    """
    Inference / test dataset.

    Default:
        A = record["input"] = H&E

    Recommended:
        --split_json ...
        --fold fold0
        --split val / test

    If split JSON contains val_keys / test_keys,
    this dataset directly uses those keys without scanning all LMDB records.
    """

    def initialize(self, opt):
        self.opt = opt
        self.transform = get_transform(opt)

        self.lmdb_path = getattr(opt, "lmdb_path", "")
        self.lmdb_field = getattr(opt, "lmdb_field", "input")
        self.split_json = getattr(opt, "split_json", "")
        self.fold = getattr(opt, "fold", "")
        self.split = getattr(opt, "split", "")

        if not self.lmdb_path:
            raise ValueError("Please specify --lmdb_path")

        if not os.path.exists(self.lmdb_path):
            raise FileNotFoundError(f"LMDB path not found: {self.lmdb_path}")

        self.env = lmdb.open(
            self.lmdb_path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=2048,
        )

        self.keys = self._load_keys_for_current_split()
        self.size = len(self.keys)

        if self.size == 0:
            raise RuntimeError(
                f"No valid samples found. "
                f"lmdb_path={self.lmdb_path}, split_json={self.split_json}, "
                f"fold={self.fold}, split={self.split}"
            )

        print(f"[LMDBSingleDataset] lmdb_path = {self.lmdb_path}")
        print(f"[LMDBSingleDataset] lmdb_field = {self.lmdb_field}")
        print(f"[LMDBSingleDataset] split_json = {self.split_json}")
        print(f"[LMDBSingleDataset] fold = {self.fold}")
        print(f"[LMDBSingleDataset] split = {self.split}")
        print(f"[LMDBSingleDataset] num samples = {self.size}")

    def _load_all_keys_from_lmdb(self) -> List[bytes]:
        with self.env.begin(write=False) as txn:
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
                    if key in _META_KEYS:
                        continue
                    if key.startswith(b"__"):
                        continue
                    keys.append(key)

        return keys

    def _load_keys_for_current_split(self) -> List[bytes]:
        if not self.split_json:
            return self._load_all_keys_from_lmdb()

        if not self.fold or not self.split:
            raise ValueError(
                "When --split_json is specified, please also specify --fold and --split."
            )

        with open(self.split_json, "r", encoding="utf-8") as f:
            split_info = json.load(f)

        if self.fold not in split_info["folds"]:
            raise KeyError(f"Fold not found in split JSON: {self.fold}")

        fold_data = split_info["folds"][self.fold]

        key_field = f"{self.split}_keys"

        if key_field in fold_data:
            key_list = fold_data[key_field]
            print(f"[LMDBSingleDataset] Using key list from JSON: {key_field}")
            return [
                k if isinstance(k, bytes) else str(k).encode("utf-8")
                for k in key_list
            ]

        if self.split not in fold_data:
            raise KeyError(f"Split not found in split JSON: {self.split}")

        allowed_patients = set(fold_data[self.split])
        print("[LMDBSingleDataset] No key list found. Fallback to patient filtering.")

        all_keys = self._load_all_keys_from_lmdb()
        filtered_keys: List[bytes] = []

        with self.env.begin(write=False) as txn:
            for idx, key in enumerate(all_keys, start=1):
                value = txn.get(key)
                if value is None:
                    continue

                record = msgpack.unpackb(value, raw=False)
                meta = _decode_meta(record.get("meta", {}))
                patient_id = _get_patient_id_from_meta(meta)

                if patient_id in allowed_patients:
                    filtered_keys.append(key)

                if idx % 100000 == 0:
                    print(f"  checked {idx}/{len(all_keys)}, selected {len(filtered_keys)}")

        return filtered_keys

    def _read_record(self, key: bytes) -> dict:
        with self.env.begin(write=False) as txn:
            value = txn.get(key)

        if value is None:
            raise KeyError(f"Key not found in LMDB: {key!r}")

        record = msgpack.unpackb(value, raw=False)

        if self.lmdb_field not in record:
            raise KeyError(
                f"Record has no field '{self.lmdb_field}'. key={key!r}"
            )

        return record

    def __getitem__(self, index):
        A_key = self.keys[index % self.size]
        record = self._read_record(A_key)

        A_img = _png_bytes_to_pil_rgb(record[self.lmdb_field])
        A = self.transform(A_img)

        if self.opt.which_direction == "BtoA":
            input_nc = self.opt.output_nc
        else:
            input_nc = self.opt.input_nc

        if input_nc == 1:
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        A_path = A_key.decode("utf-8", errors="ignore")

        return {
            "A": A,
            "A_paths": A_path,
        }

    def __len__(self):
        return self.size

    def name(self):
        return "LMDBSingleDataset"

# from __future__ import annotations

# import io
# import os
# from typing import List

# import lmdb
# import msgpack
# from PIL import Image

# from data.base_dataset import BaseDataset, get_transform


# _META_KEYS = {
#     b"__len__",
#     b"__keys__",
# }


# def _png_bytes_to_pil_rgb(png_bytes: bytes) -> Image.Image:
#     if png_bytes is None or len(png_bytes) == 0:
#         raise ValueError("Empty image bytes in LMDB record.")

#     return Image.open(io.BytesIO(png_bytes)).convert("RGB")


# class LMDBSingleDataset(BaseDataset):
#     """
#     Inference / test dataset.

#     Default:
#         A = record["input"] = H&E patch

#     You may also use:
#         --lmdb_field target
#     if you want to test target domain images.
#     """

#     def initialize(self, opt):
#         self.opt = opt
#         self.transform = get_transform(opt)

#         self.lmdb_path = getattr(opt, "lmdb_path", "")
#         self.lmdb_field = getattr(opt, "lmdb_field", "input")

#         if not self.lmdb_path:
#             raise ValueError("Please specify --lmdb_path")

#         if not os.path.exists(self.lmdb_path):
#             raise FileNotFoundError(f"LMDB path not found: {self.lmdb_path}")

#         self.env = lmdb.open(
#             self.lmdb_path,
#             readonly=True,
#             lock=False,
#             readahead=False,
#             meminit=False,
#             max_readers=2048,
#         )

#         self.keys = self._load_keys()
#         self.size = len(self.keys)

#         if self.size == 0:
#             raise RuntimeError(f"No valid samples found in LMDB: {self.lmdb_path}")

#         print(f"[LMDBSingleDataset] lmdb_path = {self.lmdb_path}")
#         print(f"[LMDBSingleDataset] lmdb_field = {self.lmdb_field}")
#         print(f"[LMDBSingleDataset] num samples = {self.size}")

#     def _load_keys(self) -> List[bytes]:
#         with self.env.begin(write=False) as txn:
#             packed_keys = txn.get(b"__keys__")
#             if packed_keys is not None:
#                 keys = msgpack.unpackb(packed_keys, raw=False)
#                 return [
#                     k if isinstance(k, bytes) else str(k).encode("utf-8")
#                     for k in keys
#                 ]

#             keys: List[bytes] = []
#             with txn.cursor() as cursor:
#                 for key, _ in cursor:
#                     if key in _META_KEYS:
#                         continue
#                     if key.startswith(b"__"):
#                         continue
#                     keys.append(key)

#         return keys

#     def _read_record(self, key: bytes) -> dict:
#         with self.env.begin(write=False) as txn:
#             value = txn.get(key)

#         if value is None:
#             raise KeyError(f"Key not found in LMDB: {key!r}")

#         record = msgpack.unpackb(value, raw=False)

#         if self.lmdb_field not in record:
#             raise KeyError(
#                 f"Record has no field '{self.lmdb_field}'. key={key!r}"
#             )

#         return record

#     def __getitem__(self, index):
#         A_key = self.keys[index % self.size]
#         record = self._read_record(A_key)

#         A_img = _png_bytes_to_pil_rgb(record[self.lmdb_field])
#         A = self.transform(A_img)

#         if self.opt.which_direction == "BtoA":
#             input_nc = self.opt.output_nc
#         else:
#             input_nc = self.opt.input_nc

#         if input_nc == 1:
#             tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
#             A = tmp.unsqueeze(0)

#         A_path = A_key.decode("utf-8", errors="ignore")

#         return {
#             "A": A,
#             "A_paths": A_path,
#         }

#     def __len__(self):
#         return self.size

#     def name(self):
#         return "LMDBSingleDataset"