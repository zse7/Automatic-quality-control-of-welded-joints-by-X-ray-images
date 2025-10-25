from pathlib import Path
import shutil

dataset_path = Path("../data/dataset_2")
splits = ["train", "val"]

notebooks_dir = Path("../notebooks")

for split in splits:
    img_folder = dataset_path / split / "images"
    label_folder = dataset_path / split / "labels"

    img_names = {f.stem.lower() for f in img_folder.glob("*") if f.suffix.lower() in [".jpg", ".jpeg", ".png"]}
    label_files = list(label_folder.glob("*.txt"))

    # orphan .txt — нет соответствующего изображения
    orphan_txt = [f for f in label_files if f.stem.lower() not in img_names]

    target_dir = notebooks_dir / "orphans" / split / "labels"
    target_dir.mkdir(parents=True, exist_ok=True)

    # перемещаем orphan .txt
    for lf in orphan_txt:
        dst = target_dir / lf.name
        shutil.move(str(lf), str(dst))

    print(f"{split}: перемещено {len(orphan_txt)} orphan .txt")
