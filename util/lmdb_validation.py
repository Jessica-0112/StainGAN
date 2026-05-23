from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import random

import lmdb
import msgpack
import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from torchvision import transforms


def _read_image_from_bytes(img_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


def _build_val_transform(load_size: int, fine_size: int):
    transform_list = []

    if load_size > 0:
        transform_list.append(
            transforms.Resize((load_size, load_size), Image.BICUBIC)
        )

    if fine_size > 0:
        transform_list.append(
            transforms.CenterCrop(fine_size)
        )

    transform_list += [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    return transforms.Compose(transform_list)


def _tensor_to_numpy_01(x: torch.Tensor) -> np.ndarray:
    """
    x: tensor, shape = [3, H, W], range roughly [-1, 1]
    return: numpy, shape = [H, W, 3], range [0, 1]
    """
    x = x.detach().float().cpu()
    x = (x + 1.0) / 2.0
    x = torch.clamp(x, 0.0, 1.0)
    arr = x.permute(1, 2, 0).numpy()
    return arr


def _compute_psnr(fake: np.ndarray, target: np.ndarray) -> float:
    mse = float(np.mean((fake - target) ** 2))
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def _compute_ssim(fake: np.ndarray, target: np.ndarray) -> float:
    return float(
        ssim(
            target,
            fake,
            channel_axis=2,
            data_range=1.0,
        )
    )


class LMDBValidationEvaluator:
    """
    Validation evaluator for paired LMDB records.

    It reads:
        record["input"]  -> H&E
        record["target"] -> real target stain

    Then:
        fake target = model.netG_A(input)

    Metrics:
        val_L1
        val_PSNR
        val_SSIM
    """

    def __init__(self, opt):
        self.opt = opt
        self.lmdb_path = Path(opt.lmdb_path)
        self.split_json = Path(opt.split_json)
        self.fold = opt.fold
        self.val_split = opt.val_split
        self.max_items = int(opt.val_max_dataset_size)

        if not self.lmdb_path.exists():
            raise FileNotFoundError(f"LMDB path not found: {self.lmdb_path}")

        if not self.split_json.exists():
            raise FileNotFoundError(f"split_json not found: {self.split_json}")


        # Flexible fixed-seed validation monitoring subset.
        self.keys = self._load_val_keys()

        # Flexible fixed-seed validation monitoring subset.
        # This supports arbitrary --val_max_dataset_size without regenerating split JSON.
        rng = random.Random(int(getattr(opt, "val_seed", 42)))
        self.keys = list(self.keys)
        rng.shuffle(self.keys)

        if self.max_items > 0 and self.max_items < len(self.keys):
            self.keys = self.keys[: self.max_items]

        if len(self.keys) == 0:
            raise RuntimeError("No validation keys found.")

        self.env = lmdb.open(
            str(self.lmdb_path),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=2048,
        )

        self.transform = _build_val_transform(
            load_size=int(opt.loadSize),
            fine_size=int(opt.fineSize),
        )

        self.metrics_csv = (
            Path(opt.checkpoints_dir)
            / opt.name
            / "validation_metrics.csv"
        )
        self.best_json = (
            Path(opt.checkpoints_dir)
            / opt.name
            / "best_validation_metric.json"
        )
        
        self.best_value = None
        self.bad_epochs = 0

        print("[LMDBValidationEvaluator] enabled")
        print(f"[LMDBValidationEvaluator] val_split = {self.val_split}")
        print(f"[LMDBValidationEvaluator] validation samples = {len(self.keys)}")
        print(f"[LMDBValidationEvaluator] metrics_csv = {self.metrics_csv}")
        
        self.val_monitor_keys_json = (
            Path(opt.checkpoints_dir)
            / opt.name
            / "val_monitor_keys.json"
        )

        self.val_monitor_keys_json.write_text(
            json.dumps(
                {
                    "val_split": self.val_split,
                    "val_seed": int(getattr(opt, "val_seed", 42)),
                    "val_max_dataset_size": int(self.max_items),
                    "num_val_samples": len(self.keys),
                    "keys": self.keys,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        print(f"[LMDBValidationEvaluator] val_monitor_keys_json = {self.val_monitor_keys_json}")

    def _load_val_keys(self) -> List[str]:
        with self.split_json.open("r", encoding="utf-8") as f:
            split_info = json.load(f)

        if self.fold not in split_info["folds"]:
            raise KeyError(f"Fold not found in split JSON: {self.fold}")

        fold_data = split_info["folds"][self.fold]
        key_field = f"{self.val_split}_keys"

        if key_field not in fold_data:
            raise KeyError(
                f"Cannot find '{key_field}' in split JSON. "
                f"Available keys: {list(fold_data.keys())}"
            )

        return list(fold_data[key_field])

    def _read_record(self, key: str) -> dict:
        with self.env.begin(write=False) as txn:
            value = txn.get(key.encode("utf-8"))

        if value is None:
            raise KeyError(f"Key not found in LMDB: {key}")

        return msgpack.unpackb(value, raw=False)

    def evaluate(self, model, epoch: int) -> Dict[str, float]:
        device = next(model.netG_A.parameters()).device
        was_training = model.netG_A.training

        model.netG_A.eval()

        l1_values = []
        psnr_values = []
        ssim_values = []

        with torch.no_grad():
            for idx, key in enumerate(self.keys, start=1):
                record = self._read_record(key)

                input_img = _read_image_from_bytes(record["input"])
                target_img = _read_image_from_bytes(record["target"])

                input_tensor = self.transform(input_img).unsqueeze(0).to(device)
                target_tensor = self.transform(target_img).to(device)

                fake_tensor = model.netG_A(input_tensor)[0]

                val_l1 = torch.mean(torch.abs(fake_tensor - target_tensor)).item()
                l1_values.append(val_l1)

                fake_np = _tensor_to_numpy_01(fake_tensor)
                target_np = _tensor_to_numpy_01(target_tensor)

                psnr_values.append(_compute_psnr(fake_np, target_np))
                ssim_values.append(_compute_ssim(fake_np, target_np))

                if idx % 200 == 0:
                    print(
                        f"[Validation] epoch={epoch}, "
                        f"processed={idx}/{len(self.keys)}"
                    )

        if was_training:
            model.netG_A.train()

        metrics = {
            "epoch": float(epoch),
            "val_L1": float(np.mean(l1_values)),
            "val_PSNR": float(np.mean(psnr_values)),
            "val_SSIM": float(np.mean(ssim_values)),
            "num_val_samples": float(len(self.keys)),
        }

        self._append_metrics_csv(metrics)
        self._update_best_and_maybe_save(model, epoch, metrics)

        return metrics

    def _append_metrics_csv(self, metrics: Dict[str, float]) -> None:
        self.metrics_csv.parent.mkdir(parents=True, exist_ok=True)

        file_exists = self.metrics_csv.exists()

        with self.metrics_csv.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "epoch",
                    "val_L1",
                    "val_PSNR",
                    "val_SSIM",
                    "num_val_samples",
                ],
            )

            if not file_exists:
                writer.writeheader()

            writer.writerow(metrics)

    def _is_improved(self, current: float) -> bool:
        metric = self.opt.early_stop_metric
        mode = self.opt.early_stop_mode
        min_delta = float(self.opt.early_stop_min_delta)

        if self.best_value is None:
            return True

        if mode == "min":
            return current < self.best_value - min_delta

        if mode == "max":
            return current > self.best_value + min_delta

        raise ValueError(f"Unsupported early_stop_mode: {mode}")

    def _update_best_and_maybe_save(
        self,
        model,
        epoch: int,
        metrics: Dict[str, float],
    ) -> None:
        metric = self.opt.early_stop_metric

        if metric not in metrics:
            raise KeyError(
                f"early_stop_metric '{metric}' not in metrics: {list(metrics.keys())}"
            )

        current = float(metrics[metric])
        improved = self._is_improved(current)

        if improved:
            self.best_value = current
            self.bad_epochs = 0

            print(
                f"[Validation] New best {metric}={current:.6f} at epoch {epoch}. "
                f"Saving best checkpoint..."
            )

            model.save("best")

            best_info = {
                "best_epoch": epoch,
                "best_metric": metric,
                "best_value": current,
                "all_metrics": metrics,
            }

            self.best_json.write_text(
                json.dumps(best_info, indent=2),
                encoding="utf-8",
            )
        else:
            self.bad_epochs += 1

            print(
                f"[Validation] No improvement in {metric}. "
                f"current={current:.6f}, best={self.best_value:.6f}, "
                f"bad_epochs={self.bad_epochs}"
            )

    def should_stop(self) -> bool:
        if not bool(self.opt.use_early_stopping):
            return False

        return self.bad_epochs >= int(self.opt.early_stop_patience)