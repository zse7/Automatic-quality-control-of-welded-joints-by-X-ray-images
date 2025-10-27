from pathlib import Path
import csv
import json

dataset2_split_root = Path("data/dataset_2_split")  
out_dir = Path("data/imagelevel")
out_dir.mkdir(parents=True, exist_ok=True)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()   # для относительных путей

EXT_IMG = {'.jpg','.jpeg','.png','.bmp','.tif','.tiff','webp'}

def collect_rows_for_split(split_name):
    split_root = dataset2_split_root / split_name
    images_dir = split_root / "images"
    labels_dir = split_root / "labels"
    rows = []
    if not images_dir.exists():
        print(f"[WARN] images dir not found for split '{split_name}': {images_dir}")
        return rows

    img_map = {p.stem: p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in EXT_IMG}

    for stem, img_path in sorted(img_map.items()):
        img_rel_path = img_path.resolve().relative_to(PROJECT_ROOT)
        label_file = labels_dir / f"{stem}.txt"
        classes_str = ""
        if label_file.exists():
            txt = label_file.read_text(encoding='utf-8', errors='ignore').strip()
            if txt != "":
                cls_set = set()
                for ln in txt.splitlines():
                    parts = ln.strip().split()
                    if not parts:
                        continue
                    try:
                        cls_set.add(str(int(float(parts[0]))))
                    except:
                        continue
                if cls_set:
                    classes_str = ",".join(sorted(cls_set, key=lambda x:int(x)))
        rows.append((str(img_rel_path), classes_str))
    return rows

def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path","classes"])
        writer.writerows(rows)

if __name__ == "__main__":
    splits = ["train","val","test"]
    all_rows = {}
    for s in splits:
        rows = collect_rows_for_split(s)
        all_rows[s] = rows
        print(f"{s}: images={len(rows)}")

    write_csv(out_dir / "train.csv", all_rows.get("train", []))
    write_csv(out_dir / "val.csv", all_rows.get("val", []))
    write_csv(out_dir / "test.csv", all_rows.get("test", []))
    print("Saved CSVs to", out_dir)

    classes_set = set()
    classes_count = {s:{} for s in splits} 

    for split_name, rows in all_rows.items():
        for _, cls_str in rows:
            if cls_str:
                for c in cls_str.split(","):
                    classes_set.add(c)
                    classes_count[split_name][c] = classes_count[split_name].get(c, 0) + 1

    classes_list = sorted(classes_set, key=lambda x:int(x)) if classes_set else []

    classes_info = {
        "classes": classes_list,
        "counts": classes_count
    }

    with open(out_dir / "classes_list.json", "w", encoding='utf-8') as f:
        json.dump(classes_info, f, ensure_ascii=False, indent=2)

    print("classes_list.json saved with counts:", out_dir / "classes_list.json")
    print("Classes found with counts per split:", classes_info)
