from __future__ import annotations

import argparse
import io
import math
from pathlib import Path
from typing import List, Optional

import lmdb
import msgpack
from PIL import Image, ImageDraw, ImageFont


def open_lmdb(lmdb_path: str):
    return lmdb.open(
        lmdb_path,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=2048,
    )


def read_record(env, key: str) -> Optional[dict]:
    key_bytes = key.encode("utf-8")
    with env.begin(write=False) as txn:
        value = txn.get(key_bytes)

    if value is None:
        return None

    return msgpack.unpackb(value, raw=False)


def bytes_to_rgb(img_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


def parse_key_from_fake_path(fake_path: Path) -> str:
    """
    StainGAN / CycleGAN 通常輸出：
      <key>_fake_B.png
      <key>_fake.png

    這裡把 suffix 拿掉，還原 LMDB key。
    """
    name = fake_path.stem

    suffixes = [
        "_fake_B",
        "_fake",
        "_synthesized",
        "_output",
    ]

    for suffix in suffixes:
        if name.endswith(suffix):
            return name[: -len(suffix)]

    return name


def find_fake_images(images_dir: Path) -> List[Path]:
    candidates = []
    for p in images_dir.rglob("*.png"):
        name = p.name.lower()

        # 排除 grid / triplet 這種後處理圖，其他 png 都視為 fake output
        if "triplet" in name or "grid" in name:
            continue

        candidates.append(p)

    return sorted(candidates)


def make_triplet_canvas(
    input_img: Image.Image,
    fake_img: Image.Image,
    target_img: Image.Image,
    title: str,
    tile_size: int = 256,
    label_h: int = 32,
    image_gap: int = 16,
    border: int = 2,
) -> Image.Image:
    input_img = input_img.resize((tile_size, tile_size))
    fake_img = fake_img.resize((tile_size, tile_size))
    target_img = target_img.resize((tile_size, tile_size))

    labels = ["H&E input", "fake CK7", "real CK7"]
    imgs = [input_img, fake_img, target_img]

    # 每一列 triplet 內部也保留間隔，避免三張圖貼在一起不好比較。
    w = tile_size * 3 + image_gap * 2 + border * 2
    h = tile_size + label_h * 2 + border * 2

    canvas = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(canvas)

    # 外框讓每一組 triplet 更容易跟上下列分開。
    draw.rectangle([0, 0, w - 1, h - 1], outline=(180, 180, 180), width=border)

    for i, (label, img) in enumerate(zip(labels, imgs)):
        x = border + i * (tile_size + image_gap)
        y = border + label_h
        draw.text((x + 8, border + 8), label, fill=(0, 0, 0))
        canvas.paste(img, (x, y))

        # 每張小圖也加細框，讓白背景或淡染色區域更清楚。
        draw.rectangle(
            [x, y, x + tile_size - 1, y + tile_size - 1],
            outline=(210, 210, 210),
            width=1,
        )

    draw.text((border + 8, border + tile_size + label_h + 8), title[:120], fill=(0, 0, 0))

    return canvas


def make_grid(
    triplets: List[Image.Image],
    cols: int = 1,
    pad: int = 40,
    bg_color: tuple[int, int, int] = (245, 245, 245),
) -> Image.Image:
    if not triplets:
        raise ValueError("No triplet images to make grid.")

    cell_w, cell_h = triplets[0].size
    rows = math.ceil(len(triplets) / cols)

    grid_w = cols * cell_w + (cols + 1) * pad
    grid_h = rows * cell_h + (rows + 1) * pad

    grid = Image.new("RGB", (grid_w, grid_h), bg_color)

    for idx, img in enumerate(triplets):
        r = idx // cols
        c = idx % cols
        x = pad + c * (cell_w + pad)
        y = pad + r * (cell_h + pad)
        grid.paste(img, (x, y))

    return grid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmdb-path", required=True)
    parser.add_argument("--images-dir", required=True, help="Directory containing fake_B PNGs.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--cols", type=int, default=1)
    parser.add_argument("--grid-pad", type=int, default=40, help="Padding between triplet rows/columns.")
    parser.add_argument("--image-gap", type=int, default=16, help="Gap between H&E/fake/real images inside each triplet.")
    parser.add_argument("--tile-size", type=int, default=256, help="Resize each patch to this size in the grid.")
    args = parser.parse_args()

    lmdb_path = args.lmdb_path
    images_dir = Path(args.images_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fake_paths = find_fake_images(images_dir)
    print(f"[INFO] fake images found: {len(fake_paths)}")

    if len(fake_paths) == 0:
        raise RuntimeError(f"No fake images found under: {images_dir}")

    env = open_lmdb(lmdb_path)

    triplets = []
    missing = 0

    for fake_path in fake_paths[: args.num_samples]:
        key = parse_key_from_fake_path(fake_path)
        record = read_record(env, key)

        if record is None:
            print(f"[WARN] LMDB key not found: {key}")
            missing += 1
            continue

        input_img = bytes_to_rgb(record["input"])
        target_img = bytes_to_rgb(record["target"])
        fake_img = Image.open(fake_path).convert("RGB")

        triplet = make_triplet_canvas(
            input_img=input_img,
            fake_img=fake_img,
            target_img=target_img,
            title=key,
            tile_size=args.tile_size,
            image_gap=args.image_gap,
        )

        triplet_path = out_dir / f"{key}_triplet.png"
        triplet.save(triplet_path)

        triplets.append(triplet)

    env.close()

    print(f"[INFO] triplets saved: {len(triplets)}")
    print(f"[INFO] missing keys: {missing}")

    if triplets:
        grid = make_grid(triplets, cols=args.cols, pad=args.grid_pad)
        grid_path = out_dir / "triplet_grid.png"
        grid.save(grid_path)
        print(f"[DONE] grid saved to: {grid_path}")


if __name__ == "__main__":
    main()