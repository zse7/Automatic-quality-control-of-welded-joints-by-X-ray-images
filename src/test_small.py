#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
balance_dataset2_split.py

Создаёт новый датасет из dataset_2_split с ограничением количества изображений
по классам для train/val/test.

LIMITS задаются для каждого split.
Файлы копируются в новую папку, исходные не трогаются.
"""

import random
from pathlib import Path
import shutil
import json

# ---------------- CONFIG ----------------
dataset2_split_root = Path("../data/dataset_2_split")
output_root = Path("../data/dataset_2_split_balanced")
output_root.mkdir(parents=True, exist_ok=True)

LIMITS = {
    "train": 1200,
    "val": 700,
    "test": 550
}

EXT_IMG = {'.jpg','.jpeg','.png','.bmp','.tif','.tiff','webp'}

# ---------------- FUNCTIONS ----------------
def collect_images_by_class(split_name):
    """
    Возвращает словарь: class_id -> список файлов для данного split
    """
    split_root = dataset2_split_root / split_name
    images_dir = split_root / "images"
    labels_dir = split_root / "labels"
    
    img_map = {p.stem: p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in EXT_IMG}
    
    class_to_files = {}
    
    for stem, img_path in img_map.items():
        label_file = labels_dir / f"{stem}.txt"
        if not label_file.exists():
            cls_list = ["no_defect"]  # пустая метка
        else:
            txt = label_file.read_text(encoding='utf-8', errors='ignore').strip()
            if txt == "":
                cls_list = ["no_defect"]
            else:
                cls_list = []
                for ln in txt.splitlines():
                    parts = ln.strip().split()
                    if not parts:
                        continue
                    try:
                        cls_list.append(str(int(float(parts[0]))))
                    except:
                        continue
                if not cls_list:
                    cls_list = ["no_defect"]
        
        for c in cls_list:
            class_to_files.setdefault(c, []).append((img_path, label_file))
    
    return class_to_files

def copy_limited(class_to_files, split_name, limit):
    """
    Копирует файлы в output_root, соблюдая лимит на класс
    """
    split_out = output_root / split_name
    images_out = split_out / "images"
    labels_out = split_out / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)
    
    counts = {}
    moved_files = 0
    
    for cls, files in class_to_files.items():
        selected = files
        if len(files) > limit:
            selected = random.sample(files, limit)
        counts[cls] = len(selected)
        for img_path, label_file in selected:
            shutil.copy2(img_path, images_out / img_path.name)
            if label_file.exists():
                shutil.copy2(label_file, labels_out / label_file.name)
            moved_files += 1
    
    return moved_files, counts

# ---------------- MAIN ----------------
if __name__ == "__main__":
    random.seed(42)  # для воспроизводимости
    summary = {}
    
    for split in ["train","val","test"]:
        print(f"Processing split: {split}")
        class_to_files = collect_images_by_class(split)
        moved_files, class_counts = copy_limited(class_to_files, split, LIMITS[split])
        summary[split] = {
            "moved_files": moved_files,
            "class_counts": class_counts
        }
        print(f"{split}: moved_files={moved_files}, classes={class_counts}")
    
    # Сохраняем статистику
    with open(output_root / "summary.json", "w", encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print("Done. New balanced dataset saved in:", output_root)
