from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import lmdb
import msgpack
from PIL import Image, ImageDraw


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


def open_lmdb(lmdb_path: str):
    return lmdb.open(
        lmdb_path,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=2048,
    )


def read_record(env, key: str) -> dict:
    with env.begin(write=False) as txn:
        value = txn.get(key.encode("utf-8"))

    if value is None:
        raise KeyError(f"Key not found in LMDB: {key}")

    return msgpack.unpackb(value, raw=False)


def bytes_to_rgb(img_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


def find_png_outputs(images_dir: Path) -> List[Path]:
    pngs = []
    for p in images_dir.rglob("*.png"):
        name = p.name.lower()
        if "triplet" in name or "grid" in name or "mosaic" in name:
            continue
        pngs.append(p)
    return sorted(pngs)


def get_key_from_png(p: Path) -> str:
    name = p.stem
    for suffix in ["_fake_B", "_fake", "_output", "_synthesized"]:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def get_xy(meta: Dict[str, Any]) -> Tuple[int, int]:
    if "x_patch" in meta and "y_patch" in meta:
        return int(meta["x_patch"]), int(meta["y_patch"])

    if "x_lv0" in meta and "y_lv0" in meta:
        return int(meta["x_lv0"]), int(meta["y_lv0"])

    raise KeyError("Cannot find x/y coordinate in meta.")


def make_mosaic(
    items: List[Tuple[int, int, Image.Image]],
    tile_size: int,
    label: str,
) -> Image.Image:
    xs = sorted(set(x for x, _, _ in items))
    ys = sorted(set(y for _, y, _ in items))

    x_to_col = {x: i for i, x in enumerate(xs)}
    y_to_row = {y: i for i, y in enumerate(ys)}

    cols = len(xs)
    rows = len(ys)

    label_h = 36
    canvas = Image.new(
        "RGB",
        (cols * tile_size, rows * tile_size + label_h),
        "white",
    )

    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), label, fill=(0, 0, 0))

    for x, y, img in items:
        col = x_to_col[x]
        row = y_to_row[y]
        img = img.resize((tile_size, tile_size))
        canvas.paste(img, (col * tile_size, label_h + row * tile_size))

    return canvas


def concat_horizontal(images: List[Image.Image], pad: int = 16) -> Image.Image:
    h = max(img.height for img in images)
    w = sum(img.width for img in images) + pad * (len(images) - 1)

    canvas = Image.new("RGB", (w, h), "white")
    x = 0

    for img in images:
        canvas.paste(img, (x, 0))
        x += img.width + pad

    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmdb-path", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tile-size", type=int, default=64)
    parser.add_argument("--max-patches", type=int, default=0)
    args = parser.parse_args()

    lmdb_path = args.lmdb_path
    images_dir = Path(args.images_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pngs = find_png_outputs(images_dir)

    if args.max_patches > 0:
        pngs = pngs[: args.max_patches]

    print(f"[INFO] PNG outputs found: {len(pngs)}")

    if len(pngs) == 0:
        raise RuntimeError(f"No PNG outputs found under: {images_dir}")

    env = open_lmdb(lmdb_path)

    he_items = []
    fake_items = []
    real_items = []

    for idx, png_path in enumerate(pngs, start=1):
        key = get_key_from_png(png_path)
        record = read_record(env, key)
        meta = decode_meta(record.get("meta", {}))
        x, y = get_xy(meta)

        he_img = bytes_to_rgb(record["input"])
        real_img = bytes_to_rgb(record["target"])
        fake_img = Image.open(png_path).convert("RGB")

        he_items.append((x, y, he_img))
        fake_items.append((x, y, fake_img))
        real_items.append((x, y, real_img))

        if idx % 1000 == 0:
            print(f"  processed {idx}/{len(pngs)}")

    env.close()

    he_mosaic = make_mosaic(he_items, args.tile_size, "H&E input")
    fake_mosaic = make_mosaic(fake_items, args.tile_size, "fake target stain")
    real_mosaic = make_mosaic(real_items, args.tile_size, "real target stain")

    he_path = out_dir / "mosaic_he_input.png"
    fake_path = out_dir / "mosaic_fake_target.png"
    real_path = out_dir / "mosaic_real_target.png"
    triplet_path = out_dir / "mosaic_triplet_he_fake_real.png"

    he_mosaic.save(he_path)
    fake_mosaic.save(fake_path)
    real_mosaic.save(real_path)

    triplet = concat_horizontal([he_mosaic, fake_mosaic, real_mosaic])
    triplet.save(triplet_path)

    print("[DONE]")
    print(f"H&E mosaic   : {he_path}")
    print(f"fake mosaic  : {fake_path}")
    print(f"real mosaic  : {real_path}")
    print(f"triplet      : {triplet_path}")


if __name__ == "__main__":
    main()