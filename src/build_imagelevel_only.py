#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_imagelevel_merged_fixed.py

Объединяет dataset_2 (labels .txt -> первые токены) и dataset_1 (папки -> class ids)
в единый CSV для multi-label классификатора: image_path,classes

Жёстко прописаны пути внутри скрипта.
"""

from pathlib import Path
import csv

# ----------------------------- ПУТИ -----------------------------
dataset2_root = Path(r"../data/dataset_2")
dataset1_root = Path(r"../data/dataset_1")
out_csv = Path(r"../data/imagelevel/image_labels.csv")

# ----------------------------- МАППИНГ dataset_1 -----------------------------
# ключи — имена папок, значения — list of class_ids
mapping_folder_to_classids = {
    "непровар": [12],     # сопоставление с id 12
    "трещины": [4],       # сопоставление с id 4
    # добавляй при необходимости другие папки
}
excluded_folders = set()  # папки, которые полностью пропускаем, если нужно

# ----------------------------- Функции -----------------------------
def build_from_dataset2(root):
    """
    Возвращает dict: image_full_path (str) -> set(class_id_str)
    """
    root = Path(root)
    out = {}
    missed_txt = []
    for split in ("train",):
        images_dir = root / split / "images"
        labels_dir = root / split / "labels"
        if not images_dir.exists() or not labels_dir.exists():
            print(f"[WARN] Для split {split} не найдены папки: {images_dir} или {labels_dir}")
            continue
        # map stems to image path (учитываем расширения)
        img_map = {}
        for p in images_dir.iterdir():
            if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.tif','.tiff','webp'}:
                img_map[p.stem] = str(p.resolve())
        # parse labels
        for txt in labels_dir.glob("*.txt"):
            stem = txt.stem
            img_full = img_map.get(stem)
            if img_full is None:
                missed_txt.append(str(txt))
                continue
            txt_text = txt.read_text(encoding='utf-8', errors='ignore').strip()
            if txt_text == "":
                out[img_full] = set()
                continue
            cls_set = set()
            for ln in txt_text.splitlines():
                parts = ln.strip().split()
                if not parts:
                    continue
                try:
                    cls_set.add(str(int(float(parts[0]))))
                except:
                    continue
            out[img_full] = cls_set
    return out, missed_txt

def build_from_dataset1(root, mapping_folder_to_classids, excluded_folders=None):
    """
    Возвращает dict: image_full_path (str) -> set(class_id_str)
    Обрабатывает только папки из mapping_folder_to_classids keys.
    Если mapping value == [] -> означает "no defect" (пустой set).
    """
    root = Path(root)
    out = {}
    excluded_folders = set(excluded_folders or [])
    for split in ("training",):
        split_dir = root / split
        if not split_dir.exists():
            continue
        for sub in split_dir.iterdir():
            if not sub.is_dir():
                continue
            name = sub.name
            if name in excluded_folders:
                print(f"[INFO] Пропускаем папку (excluded): {name}")
                continue
            if name not in mapping_folder_to_classids:
                print(f"[INFO] Папка {name} не в mapping, пропускаем")
                continue
            mapped = mapping_folder_to_classids[name]
            cls_set = set(str(int(x)) for x in mapped) if mapped else set()
            for p in sub.iterdir():
                if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.tif','.tiff','webp'}:
                    out[str(p.resolve())] = set(cls_set)
    return out

def merge_maps(map2, map1):
    """Объединить два словаря image->set, отдавая union"""
    out = {}
    keys = set(map2.keys()) | set(map1.keys())
    for k in keys:
        s = set()
        if k in map2:
            s |= set(map2[k])
        if k in map1:
            s |= set(map1[k])
        out[k] = s
    return out

# ----------------------------- Основной блок -----------------------------
if __name__ == "__main__":
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    print("Чтение dataset_2...")
    d2_map, missed_txt = build_from_dataset2(dataset2_root)
    print(f"Найдено изображений из dataset_2: {len(d2_map)}; .txt без image: {len(missed_txt)}")
    if missed_txt:
        print("Примеры .txt без image:", missed_txt[:10])

    print("Чтение dataset_1 (по mapping)...")
    d1_map = build_from_dataset1(dataset1_root, mapping_folder_to_classids, excluded_folders=excluded_folders)
    print(f"Найдено изображений из dataset_1 (после маппинга): {len(d1_map)}")

    merged = merge_maps(d2_map, d1_map)
    print("Всего изображений в merged:", len(merged))

    # статистика
    count_no_labels = sum(1 for s in merged.values() if len(s)==0)
    classes_present = sorted({c for s in merged.values() for c in s}, key=lambda x:int(x))
    print("Изображений без меток (no-defect):", count_no_labels)
    print("Классы, найденные в merged:", classes_present)

    # write CSV
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path","classes"])
        for img_path, clsset in merged.items():
            cls_str = ",".join(sorted(clsset, key=lambda x:int(x))) if clsset else ""
            writer.writerow([img_path, cls_str])

    print("CSV сохранён в:", out_csv)
    print("Готово.")
