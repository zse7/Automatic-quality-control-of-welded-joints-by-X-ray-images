from pathlib import Path
import shutil

dataset_path = Path("../data/dataset_2")
splits = ["train", "val"]

notebooks_dir = Path("../notebooks")

for split in splits:
    img_folder = dataset_path / split / "images"
    label_folder = dataset_path / split / "labels"

    img_names = {f.stem.lower(): f for f in img_folder.glob("*") if f.suffix.lower() in [".jpg", ".jpeg", ".png"]}
    label_names = {f.stem.lower() for f in label_folder.glob("*.txt")}

    images_without_label = [img_names[name] for name in img_names if name not in label_names]

    # создаём папку orphans для изображений
    target_dir = notebooks_dir / "orphans" / split / "images"
    target_dir.mkdir(parents=True, exist_ok=True)

    # перемещаем
    for img_path in images_without_label:
        dst = target_dir / img_path.name
        shutil.move(str(img_path), str(dst))

    print(f"{split}: перемещено {len(images_without_label)} изображений без меток")
