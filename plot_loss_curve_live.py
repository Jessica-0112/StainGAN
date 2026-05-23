from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


LINE_RE = re.compile(
    r"\(epoch:\s*(\d+),\s*iters:\s*(\d+),\s*time:\s*([0-9.]+)\)\s*(.*)"
)

LOSS_RE = re.compile(
    r"([A-Za-z_]+):\s*(-?\d+(?:\.\d+)?)"
)


def parse_loss_log(log_path: Path) -> pd.DataFrame:
    rows = []

    if not log_path.exists():
        raise FileNotFoundError(f"loss_log.txt not found: {log_path}")

    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = LINE_RE.search(line)
            if m is None:
                continue

            epoch = int(m.group(1))
            iters = int(m.group(2))
            time_per_sample = float(m.group(3))
            loss_text = m.group(4)

            row = {
                "epoch": epoch,
                "iters": iters,
                "time": time_per_sample,
            }

            for name, value in LOSS_RE.findall(loss_text):
                row[name] = float(value)

            rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values(["epoch", "iters"]).reset_index(drop=True)
    df["step"] = range(1, len(df) + 1)

    # 估計每個 epoch 的最大 iters，用來建立連續 global_iter
    max_iter_per_epoch = df.groupby("epoch")["iters"].max().max()
    df["global_iter"] = (df["epoch"] - 1) * max_iter_per_epoch + df["iters"]

    return df


def moving_average(series: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return series
    return series.rolling(window=window, min_periods=1).mean()


def plot_group(
    df: pd.DataFrame,
    loss_names: list[str],
    out_path: Path,
    x_col: str,
    smooth: int,
    title: str,
) -> None:
    available = [name for name in loss_names if name in df.columns]

    if not available:
        return

    plt.figure(figsize=(12, 6))

    for name in available:
        y = moving_average(df[name], smooth)
        plt.plot(df[x_col], y, label=name)

    plt.xlabel(x_col)
    plt.ylabel("loss")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--x-axis",
        default="step",
        choices=["step", "iters", "epoch", "global_iter"],
        help="X-axis for plotting.",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=10,
        help="Moving average window. Use 1 for no smoothing.",
    )
    args = parser.parse_args()

    log_path = Path(args.log_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = parse_loss_log(log_path)

    if df.empty:
        print("[WARN] No valid loss records found yet.")
        return

    csv_path = out_dir / "training_loss_parsed.csv"
    df.to_csv(csv_path, index=False)

    # 全部 loss 畫在一張，方便快速看
    all_losses = [
        "D_A", "G_A", "Cyc_A",
        "D_B", "G_B", "Cyc_B",
        "idt_A", "idt_B",
        "Aicha_A", "Aicha_B",
    ]

    plot_group(
        df=df,
        loss_names=all_losses,
        out_path=out_dir / "training_loss_all.png",
        x_col=args.x_axis,
        smooth=args.smooth,
        title=f"Training Loss Curves, smooth={args.smooth}",
    )

    # GAN loss
    plot_group(
        df=df,
        loss_names=["D_A", "G_A", "D_B", "G_B"],
        out_path=out_dir / "training_loss_gan.png",
        x_col=args.x_axis,
        smooth=args.smooth,
        title=f"GAN Loss Curves, smooth={args.smooth}",
    )

    # Cycle / identity loss
    plot_group(
        df=df,
        loss_names=["Cyc_A", "Cyc_B", "idt_A", "idt_B"],
        out_path=out_dir / "training_loss_cycle_identity.png",
        x_col=args.x_axis,
        smooth=args.smooth,
        title=f"Cycle and Identity Loss Curves, smooth={args.smooth}",
    )

    print(f"[DONE] Parsed rows: {len(df)}")
    print(f"[DONE] CSV saved to: {csv_path}")
    print(f"[DONE] Figures saved to: {out_dir}")


if __name__ == "__main__":
    main()