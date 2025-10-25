#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
remap_labels_polygons.py
Создаёт новый датасет с переразмеченными классами для YOLO (полигоны).
Старые файлы остаются нетронутыми.
"""

from pathlib import Path
import shutil

# пути к старому и новому датасету
old_dataset = Path("../data/dataset_2")
new_dataset = Path("../data/dataset_2_remap")

splits = ["train", "val"]

# карта переназначения классов
# 0: Пора -> 0
# 1: Включение -> 1
# 2: Подрез -> 4
# 3: Прожог -> 5
# 4: Трещина -> 2
# 5: Наплыв -> 5
# 6: Эталон1 -> 5
# 7: Эталон2 -> 5
# 8: Эталон3 -> 5
# 9: Пора-скрытая -> 5
# 10: Утяжина -> 5
# 11: Несплавление -> 3
# 12: Непровар корня -> 5
remap_classes = {
    0:0, 1:1, 2:4, 3:5, 4:2,
    5:5, 6:5, 7:5, 8:5, 9:5,
    10:5, 11:3, 12:5
}

for split in splits:
    old_labels = old_dataset / split / "labels"
    new_labels = new_dataset / split / "labels"
    new_labels.mkdir(parents=True, exist_ok=True)

    for txt_file in old_labels.glob("*.txt"):
        new_lines = []
        with open(txt_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    # пустой файл или некорректная строка — оставляем пустым
                    continue
                cls_id = int(parts[0])
                new_cls = remap_classes.get(cls_id, 5)  # всё остальное -> 5 (Прочие дефекты)
                # сохраняем полигоны без изменений
                new_lines.append(" ".join([str(new_cls)] + parts[1:]))

        # сохраняем в новый файл
        out_file = new_labels / txt_file.name
        with open(out_file, "w", encoding="utf-8") as f:
            f.write("\n".join(new_lines))
    print(f"{split}: переразмечено {len(list(old_labels.glob('*.txt')))} файлов")
