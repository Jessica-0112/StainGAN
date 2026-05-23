from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def plot_metric(df: pd.DataFrame, metric: str, out_path: Path) -> None:
    if metric not in df.columns:
        print(f"[WARN] {metric} not found in CSV.")
        return

    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df[metric], marker="o")
    plt.xlabel("epoch")
    plt.ylabel(metric)
    plt.title(f"Validation {metric}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    df = df.sort_values("epoch").reset_index(drop=True)

    plot_metric(df, "val_L1", out_dir / "val_L1_curve.png")
    plot_metric(df, "val_PSNR", out_dir / "val_PSNR_curve.png")
    plot_metric(df, "val_SSIM", out_dir / "val_SSIM_curve.png")

    # 合併圖：L1 單獨一軸，PSNR/SSIM 另外各自單圖比較清楚
    df.to_csv(out_dir / "validation_metrics_sorted.csv", index=False)

    print(f"[DONE] validation curves saved to: {out_dir}")


if __name__ == "__main__":
    main()