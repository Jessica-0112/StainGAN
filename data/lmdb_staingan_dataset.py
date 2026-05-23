from __future__ import annotations

import io
import json
import os
import random
from typing import Any, Dict, List, Optional, Set

from collections import defaultdict

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


class LMDBStainGANDataset(BaseDataset):
    """
    Training dataset for your merged LMDB.

    A = record["input"]  = H&E
    B = record["target"] = CK7 / TTF-1 / EVG

    Recommended:
        --split_json ...
        --fold fold0
        --split train

    If split JSON contains train_keys / val_keys / test_keys,
    this dataset directly uses those keys without scanning all LMDB records.
    """

    def initialize(self, opt):
        self.opt = opt
        self.transform = get_transform(opt)

        self.lmdb_path = getattr(opt, "lmdb_path", "")
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

        self.all_keys = self._load_keys_for_current_split()

        self.patient_balanced_train = (
            bool(getattr(opt, "patient_balanced_train", False))
            and bool(getattr(opt, "isTrain", False))
            and self.split == "train"
        )

        self.balance_epoch = 0
        self.patient_to_keys = None

        if self.patient_balanced_train:
            print("[LMDBStainGANDataset] Patient-balanced training is enabled.")

            self.patient_to_keys = self._load_patient_to_keys_from_split_json()

            if self.patient_to_keys is None:
                print("[LMDBStainGANDataset] train_patient_to_keys not found in JSON. Fallback to LMDB grouping.")
                self.patient_to_keys = self._build_patient_to_keys(self.all_keys)
            else:
                print(
                    "[LMDBStainGANDataset] Loaded train_patient_to_keys from split JSON. "
                    f"patients={len(self.patient_to_keys)}"
                )

            self.keys = self._sample_patient_balanced_keys(epoch=0)
        else:
            self.keys = self.all_keys

        self.size = len(self.keys)

        if self.size == 0:
            raise RuntimeError(
                f"No valid samples found. "
                f"lmdb_path={self.lmdb_path}, split_json={self.split_json}, "
                f"fold={self.fold}, split={self.split}"
            )

        print(f"[LMDBStainGANDataset] lmdb_path = {self.lmdb_path}")
        print(f"[LMDBStainGANDataset] split_json = {self.split_json}")
        print(f"[LMDBStainGANDataset] fold = {self.fold}")
        print(f"[LMDBStainGANDataset] split = {self.split}")
        print(f"[LMDBStainGANDataset] num samples = {self.size}")

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
            print(f"[LMDBStainGANDataset] Using key list from JSON: {key_field}")
            return [
                k if isinstance(k, bytes) else str(k).encode("utf-8")
                for k in key_list
            ]

        # fallback：如果 JSON 沒有 train_keys / val_keys / test_keys，
        # 才用 patient list 去掃 LMDB。
        if self.split not in fold_data:
            raise KeyError(f"Split not found in split JSON: {self.split}")

        allowed_patients = set(fold_data[self.split])
        print("[LMDBStainGANDataset] No key list found. Fallback to patient filtering.")

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
    
    def _load_patient_to_keys_from_split_json(self):
        if not self.split_json or not self.fold:
            return None

        with open(self.split_json, "r", encoding="utf-8") as f:
            split_info = json.load(f)

        fold_data = split_info.get("folds", {}).get(self.fold, None)
        if fold_data is None:
            return None

        patient_to_keys = fold_data.get("train_patient_to_keys", None)
        if patient_to_keys is None:
            return None

        output = {}
        for patient_id, keys in patient_to_keys.items():
            output[str(patient_id)] = [
                k if isinstance(k, bytes) else str(k).encode("utf-8")
                for k in keys
            ]

        return output

    def _read_record(self, key: bytes) -> dict:
        with self.env.begin(write=False) as txn:
            value = txn.get(key)

        if value is None:
            raise KeyError(f"Key not found in LMDB: {key!r}")

        record = msgpack.unpackb(value, raw=False)

        if "input" not in record:
            raise KeyError(f"Record has no 'input' field. key={key!r}")

        if "target" not in record:
            raise KeyError(f"Record has no 'target' field. key={key!r}")

        return record
    
    def _get_patient_id_for_key(self, key: bytes) -> str:
        record = self._read_record(key)
        meta = _decode_meta(record.get("meta", {}))
        return _get_patient_id_from_meta(meta)


    def _build_patient_to_keys(self, keys: List[bytes]):
        patient_to_keys = defaultdict(list)

        print("[LMDBStainGANDataset] Grouping training keys by patient_id...")

        for idx, key in enumerate(keys, start=1):
            patient_id = self._get_patient_id_for_key(key)
            patient_to_keys[patient_id].append(key)

            if idx % 100000 == 0:
                print(
                    f"  grouped {idx}/{len(keys)} keys, "
                    f"patients={len(patient_to_keys)}"
                )

        patient_to_keys = dict(patient_to_keys)

        print("[LMDBStainGANDataset] Patient-balanced summary:")
        for patient_id in sorted(patient_to_keys.keys()):
            print(f"  {patient_id}: {len(patient_to_keys[patient_id])} patches")

        return patient_to_keys


    def _sample_patient_balanced_keys(self, epoch: int) -> List[bytes]:
        if self.patient_to_keys is None:
            raise RuntimeError("patient_to_keys is not initialized.")

        target_size = int(getattr(self.opt, "max_dataset_size", 0))

        if target_size <= 0 or target_size > len(self.all_keys):
            target_size = len(self.all_keys)

        patients = sorted(self.patient_to_keys.keys())
        num_patients = len(patients)

        if num_patients == 0:
            raise RuntimeError("No patients found for patient-balanced sampling.")

        rng = random.Random(int(self.opt.train_balance_seed) + int(epoch))

        # 每個 patient 的基本 quota
        base_quota = target_size // num_patients
        remainder = target_size % num_patients

        # 打亂 patient 順序，讓多出來的 remainder 不會永遠落在同幾個 patient
        shuffled_patients = list(patients)
        rng.shuffle(shuffled_patients)

        selected_keys: List[bytes] = []

        for idx, patient_id in enumerate(shuffled_patients):
            quota = base_quota + (1 if idx < remainder else 0)
            keys = list(self.patient_to_keys[patient_id])

            if quota <= 0:
                continue

            if len(keys) >= quota:
                sampled = rng.sample(keys, quota)
            else:
                if bool(getattr(self.opt, "train_balance_with_replacement", False)):
                    sampled = [rng.choice(keys) for _ in range(quota)]
                else:
                    sampled = keys

            selected_keys.extend(sampled)

        # 如果某些 patient patch 不足，導致總數不夠，從全部 keys 裡補齊
        if len(selected_keys) < target_size:
            selected_set = set(selected_keys)
            remaining_pool = [k for k in self.all_keys if k not in selected_set]
            rng.shuffle(remaining_pool)

            need = target_size - len(selected_keys)
            selected_keys.extend(remaining_pool[:need])

        # 如果超過，裁切到 target_size
        if len(selected_keys) > target_size:
            selected_keys = selected_keys[:target_size]

        rng.shuffle(selected_keys)

        print(
            f"[LMDBStainGANDataset] Patient-balanced epoch={epoch}, "
            f"patients={num_patients}, selected_patches={len(selected_keys)}, "
            f"target_size={target_size}, base_quota={base_quota}, remainder={remainder}"
        )

        return selected_keys


    def resample_epoch_keys(self):
        if not self.patient_balanced_train:
            return

        self.balance_epoch += 1
        self.keys = self._sample_patient_balanced_keys(epoch=self.balance_epoch)
        self.size = len(self.keys)

    def __getitem__(self, index):
        A_key = self.keys[index % self.size]

        if self.opt.serial_batches:
            B_key = self.keys[index % self.size]
        else:
            B_key = self.keys[random.randint(0, self.size - 1)]

        A_record = self._read_record(A_key)
        B_record = self._read_record(B_key)

        A_img = _png_bytes_to_pil_rgb(A_record["input"])
        B_img = _png_bytes_to_pil_rgb(B_record["target"])

        A = self.transform(A_img)
        B = self.transform(B_img)

        if self.opt.which_direction == "BtoA":
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if input_nc == 1:
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)

        A_path = A_key.decode("utf-8", errors="ignore")
        B_path = B_key.decode("utf-8", errors="ignore")

        return {
            "A": A,
            "B": B,
            "A_paths": A_path,
            "B_paths": B_path,
        }

    def __len__(self):
        return len(self.keys)

    def name(self):
        return "LMDBStainGANDataset"