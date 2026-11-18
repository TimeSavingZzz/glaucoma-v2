"""数据自检：打印目录、CSV 头、样本数与标签分布、图片可读性。"""
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

root = Path(sys.argv[1] if len(sys.argv) > 1 else "/mnt/glaucoma/data")
print("ROOT", root)
for p in sorted(root.iterdir()):
    print(" ", "DIR " if p.is_dir() else "FILE", p.name,
          len(list(p.iterdir())) if p.is_dir() else p.stat().st_size)

for csv in root.rglob("*.csv"):
    try:
        df = pd.read_csv(csv)
    except Exception as e:  # noqa: BLE001
        print("CSV", csv, "ERR", e)
        continue
    print("CSV", csv.relative_to(root), "rows", len(df), "cols", list(df.columns)[:6])
    if "True_Label" in df.columns:
        print("   labels", df["True_Label"].value_counts().to_dict())
    names = df["Image_name"].astype(str).head(3).tolist() if "Image_name" in df.columns else []
    print("   head", names)

for d in root.rglob("*"):
    if d.is_dir() and any(f.suffix.lower() in (".jpg", ".jpeg", ".png") for f in d.iterdir()):
        files = [f for f in d.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
        print("IMGDIR", d.relative_to(root), len(files), files[0].name)
        try:
            with Image.open(files[0]) as im:
                print("   size", im.size, im.mode)
        except Exception as e:  # noqa: BLE001
            print("   ERR", e)
