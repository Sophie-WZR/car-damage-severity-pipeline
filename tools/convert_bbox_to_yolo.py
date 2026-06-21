#!/usr/bin/env python3
"""Convert a CSV of pixel bbox coordinates to YOLO .txt labels.

CSV format (header required):
filename,x_min,y_min,x_max,y_max,class

Example:
000001.jpg,100,150,300,450,1

Usage:
python tools/convert_bbox_to_yolo.py --csv annotations/bboxes.csv --images-dir data/images --labels-dir data/labels --create-empty
"""
import argparse
import csv
from pathlib import Path
from PIL import Image


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def convert(csv_path: Path, images_dir: Path, labels_dir: Path, create_empty: bool = False, min_side: int = 0):
    ensure_dir(labels_dir)
    rows_by_file = {}
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            fname = r['filename']
            x_min = float(r['x_min'])
            y_min = float(r['y_min'])
            x_max = float(r['x_max'])
            y_max = float(r['y_max'])
            cls = int(r['class'])
            rows_by_file.setdefault(fname, []).append((x_min, y_min, x_max, y_max, cls))

    # Optionally create empty label files for all images
    if create_empty:
        for img_path in images_dir.glob('*'):
            if img_path.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'):
                continue
            out_file = labels_dir / (img_path.stem + '.txt')
            if not out_file.exists():
                out_file.write_text('')

    for fname, boxes in rows_by_file.items():
        img_path = images_dir / fname
        if not img_path.exists():
            print(f'Warning: image not found {img_path}, skipping')
            continue
        with Image.open(img_path) as img:
            w, h = img.size
        out_file = labels_dir / (Path(fname).stem + '.txt')
        with open(out_file, 'w') as out:
            for x_min, y_min, x_max, y_max, cls in boxes:
                bw = x_max - x_min
                bh = y_max - y_min
                if min(bw, bh) < min_side:
                    continue
                x_center = (x_min + x_max) / 2.0 / w
                y_center = (y_min + y_max) / 2.0 / h
                width = bw / w
                height = bh / h
                out.write(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


def main():
    parser = argparse.ArgumentParser(description='Convert pixel bboxes CSV to YOLO .txt labels')
    parser.add_argument('--csv', required=True, help='Path to CSV file with bbox annotations')
    parser.add_argument('--images-dir', required=True, help='Directory containing images')
    parser.add_argument('--labels-dir', required=True, help='Output directory for YOLO labels')
    parser.add_argument('--create-empty', action='store_true', help='Create empty .txt for images without boxes')
    parser.add_argument('--min-side', type=int, default=0, help='Ignore boxes with short side < min-side pixels')
    args = parser.parse_args()

    convert(Path(args.csv), Path(args.images_dir), Path(args.labels_dir), args.create_empty, args.min_side)


if __name__ == '__main__':
    main()
