#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
split_dataset2_stratified.py

Создает копию dataset_2 в новое место и делит train на новый train + test.
Сохраняет val без изменений. Стратифицированное разбиение по классам (multi-label приближенно).

Не модифицирует исходный dataset_2.
"""
from pathlib import Path
import shutil
import random
from collections import defaultdict

random.seed(42)

# === НАСТРОЙКИ ===
dataset2_root = Path("../data/dataset_2")       # исходный dataset_2
output_root = Path("../data/dataset_2_split")  # куда сохраняем копию
test_ratio = 0.2                                # доля test из train
min_train_size = 10000                          # минимум оставшихся данных в train

splits = ["train", "val"]

for split in splits:
    print(f"--- {split.upper()} ---")
    images_dir = dataset2_root / split / "images"
    labels_dir = dataset2_root / split / "labels"

    out_images_dir = output_root / split / "images"
    out_labels_dir = output_root / split / "labels"
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    # копируем все файлы на новое место (сохраняя структуру)
    for p in images_dir.iterdir():
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            shutil.copy2(p, out_images_dir / p.name)
    for p in labels_dir.iterdir():
        if p.is_file() and p.suffix.lower() == ".txt":
            shutil.copy2(p, out_labels_dir / p.name)

# === теперь разбиваем train на новый train + test ===
train_images_dir = output_root / "train" / "images"
train_labels_dir = output_root / "train" / "labels"

all_stems = [p.stem for p in train_images_dir.iterdir() if p.is_file()]
train_map = {}

# строим словарь stem -> set классов
for stem in all_stems:
    label_file = train_labels_dir / f"{stem}.txt"
    if not label_file.exists() or label_file.stat().st_size == 0:
        train_map[stem] = {"no_defect"}
    else:
        lines = label_file.read_text(encoding='utf-8', errors='ignore').splitlines()
        cls_set = set()
        for ln in lines:
            parts = ln.strip().split()
            if parts:
                try:
                    cls_set.add(parts[0])
                except:
                    continue
        train_map[stem] = cls_set if cls_set else {"no_defect"}

# строим class -> list of stems
class_to_stems = defaultdict(list)
for stem, clsset in train_map.items():
    for c in clsset:
        class_to_stems[c].append(stem)

# выбираем тестовые стемы
test_stems = set()
for c, stems in class_to_stems.items():
    n_c = max(1, int(len(stems) * test_ratio))
    selected = random.sample(stems, min(n_c, len(stems)))
    test_stems.update(selected)

# корректируем, чтобы в train осталось хотя бы min_train_size
if len(all_stems) - len(test_stems) < min_train_size:
    to_remove = len(test_stems) - (len(all_stems) - min_train_size)
    test_stems = set(list(test_stems)[:-to_remove])

# создаем папку test
test_images_dir = output_root / "test" / "images"
test_labels_dir = output_root / "test" / "labels"
test_images_dir.mkdir(parents=True, exist_ok=True)
test_labels_dir.mkdir(parents=True, exist_ok=True)

# перемещаем файлы в test
for stem in test_stems:
    img_path = train_images_dir / f"{stem}.jpg"
    if not img_path.exists():
        img_path = train_images_dir / f"{stem}.png"
    if img_path.exists():
        shutil.move(str(img_path), test_images_dir / img_path.name)
    label_path = train_labels_dir / f"{stem}.txt"
    if label_path.exists():
        shutil.move(str(label_path), test_labels_dir / label_path.name)

print("=== Разделение dataset_2 завершено ===")
print(f"Train: {len(list(train_images_dir.iterdir()))} изображений")
print(f"Val: {len(list(output_root / 'val' / 'images'))} изображений")
print(f"Test: {len(list(test_images_dir.iterdir()))} изображений")
