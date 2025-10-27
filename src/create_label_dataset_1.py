import os
import shutil
from pathlib import Path

def create_defect_annotations_fixed():
    """Создание аннотаций с фиксированными координатами для Дефект 1 и Дефект 3"""

    defect_folders = {
        "непровар": {
            "class_id": 12,
            "paths": {
                "train": ["data/data/training/непровар"],
                "test": ["data/data/testing/непровар"],
                "val": ["data/data/validation/непровар"]
            }
        },
        "трещины": {
            "class_id": 4,
            "paths": {
                "train": ["data/data/training/трещины"],
                "test": ["data/data/testing/трещины"],
                "val": ["data/data/validation/трещины"]
            }
        }
    }

    output_dirs = {
        "train": {
            "labels": Path("data/dataset_2_split/train/labels"),
            "images": Path("data/dataset_2_split/train/images")
        },
        "test": {
            "labels": Path("data/dataset_2_split/test/labels"),
            "images": Path("data/dataset_2_split/test/images")
        },
        "val": {
            "labels": Path("data/dataset_2_split/val/labels"),
            "images": Path("data/dataset_2_split/val/images")
        }
    }

    for split_dirs in output_dirs.values():
        for dir_path in split_dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)

    for defect_name, defect_info in defect_folders.items():
        class_id = defect_info["class_id"]
        print(f"\nОбработка дефекта: {defect_name} (класс {class_id})")

        for split, folders in defect_info["paths"].items():
            for folder in folders:
                folder_path = Path(folder)
                if not folder_path.exists():
                    print(f"Папка не найдена: {folder_path}")
                    continue

                img_out_dir = output_dirs[split]["images"]
                lbl_out_dir = output_dirs[split]["labels"]

                image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                print(f"{split.upper()}: найдено {len(image_files)} файлов")

                for img_name in image_files:
                    src_img = folder_path / img_name
                    dst_img = img_out_dir / img_name
                    dst_lbl = lbl_out_dir / (Path(img_name).stem + ".txt")

                    # Копируем изображение
                    shutil.copy2(src_img, dst_img)

                    # Создаём фиксированную аннотацию
                    with open(dst_lbl, "w", encoding="utf-8") as f:
                        f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")

                print(f"{split.upper()}: создано {len(image_files)} аннотаций")

    print("\nВсе аннотации успешно созданы.")

if __name__ == "main":
    create_defect_annotations_fixed()