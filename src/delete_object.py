import os
import random
from pathlib import Path
from collections import defaultdict

dataset_path = Path("data/dataset_2_split")
max_per_class = 1000 

for split in ["train", "val", "test"]:
    images_path = dataset_path / split / "images"
    labels_path = dataset_path / split / "labels"

    class_files = defaultdict(list)  # {class_id: [(label_file, lines_of_class, all_lines)]}

    for label_file in labels_path.glob("*.txt"):
        with open(label_file, "r", encoding="utf-8") as f:
            all_lines = [line.strip() for line in f if line.strip()]
        
        classes_in_file = defaultdict(list)
        for line in all_lines:
            cls_id = line.split()[0]
            classes_in_file[cls_id].append(line)
        
        for cls_id, cls_lines in classes_in_file.items():
            class_files[cls_id].append((label_file, cls_lines, all_lines))

    for cls_id, files in class_files.items():
        total_objects = sum(len(f[1]) for f in files)
        print(f"{split.upper()} - класс {cls_id}: найдено {total_objects} объектов")

        if total_objects <= max_per_class:
            print(f"{split.upper()} - класс {cls_id}: уже меньше {max_per_class}, ничего удалять не нужно")
            continue

        to_remove = total_objects - max_per_class
        print(f"{split.upper()} - класс {cls_id}: нужно удалить {to_remove} объектов")

        random.shuffle(files)
        removed_count = 0

        for label_file, cls_lines, all_lines in files:
            remaining_to_remove = to_remove - removed_count
            if remaining_to_remove <= 0:
                break

            if len(cls_lines) <= remaining_to_remove:
                new_lines = [line for line in all_lines if line.split()[0] != cls_id]
                removed_count += len(cls_lines)
            else:
                cls_to_keep = len(cls_lines) - remaining_to_remove
                new_lines = [line for line in all_lines if line.split()[0] != cls_id]
                new_lines.extend(cls_lines[:cls_to_keep])
                removed_count += remaining_to_remove

            if not new_lines:
                label_file.unlink()
                base_stem = label_file.stem
                for ext in [".jpg", ".jpeg", ".png"]:
                    img_file = images_path / f"{base_stem}{ext}"
                    if img_file.exists():
                        img_file.unlink()
            else:
                with open(label_file, "w", encoding="utf-8") as f:
                    f.write("\n".join(new_lines) + "\n")

        print(f"{split.upper()} - класс {cls_id}: удалено объектов: {removed_count}\n")
