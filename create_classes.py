import os
from pathlib import Path
from shutil import copy2

images_dir = Path("data/train/images")
labels_dir = Path("data/train/labels")
new_labels_dir = Path("data/train/labels_6classes")
new_labels_dir.mkdir(parents=True, exist_ok=True)

map_to_case_classes = {
    0: 0,  # Porosity
    1: 1,  # Slag Inclusion
    2: 4,  # Undercut
    3: 5,  # Other
    4: 2,  # Crack
    5: 5,  # Other
    6: 5,  # Other
    7: 5,  # Other
    8: 5,  # Other
    9: 0,  # Porosity
    10:5,  # Other
    11:3,  # Lack of Fusion
    12:3   # Lack of Fusion
}

new_class_names = ["Porosity","Slag Inclusion","Crack","Lack of Fusion","Undercut","Other"]

for txt_file in labels_dir.glob("*.txt"):
    new_lines = []
    with open(txt_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts: continue
            old_class = int(parts[0])
            coords = parts[1:]
            new_class = map_to_case_classes.get(old_class, 5) 
            new_lines.append(f"{new_class} " + " ".join(coords))

    new_txt_file = new_labels_dir / txt_file.name
    with open(new_txt_file, "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines))

print(f"Saved: {new_labels_dir}")

data_yaml = Path("data/train/data_6classes.yaml")
data_yaml.write_text(f"""
train: images
val: ../val/images
nc: 6
names: {new_class_names}
""".strip())
print(f"data.yaml создан: {data_yaml}")