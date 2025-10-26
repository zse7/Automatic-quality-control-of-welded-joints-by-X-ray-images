import os
from collections import Counter, defaultdict

def count_dataset(dataset_path="data/dataset_2_split"):
    splits = ["train", "val", "test"]

    for split in splits:
        images_dir = os.path.join(dataset_path, split, "images")
        labels_dir = os.path.join(dataset_path, split, "labels")

        if not os.path.exists(images_dir):
            print(f"\n--- {split.upper()} ---")
            print("Папка не найдена")
            continue

        image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        total_images = len(image_files)

        txt_files = []
        empty_txt = 0
        missing_txt = 0
        class_counter = Counter()

        for img_file in image_files:
            base = os.path.splitext(img_file)[0]
            label_path = os.path.join(labels_dir, base + ".txt")

            if os.path.exists(label_path):
                txt_files.append(label_path)

                if os.path.getsize(label_path) == 0:
                    empty_txt += 1
                else:
                    with open(label_path, "r", encoding="utf-8") as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) > 0:
                                cls = parts[0]
                                class_counter[cls] += 1
            else:
                missing_txt += 1

        print(f"\n--- {split.upper()} ---")
        print(f"Изображений: {total_images}")
        print(f"Файлов с метками: {len(txt_files)}")
        print(f"Пустых файлов с метками: {empty_txt}")
        print(f"Изображений без .txt: {missing_txt}")

        if len(class_counter) > 0:
            print("\nКлассы и количество объектов:")
            for cls, count in sorted(class_counter.items(), key=lambda x: int(x[0])):
                print(f"  Класс {cls}: {count}")
        else:
            print("Нет размеченных объектов.")

if __name__ == "__main__":
    dataset_path = "data/dataset_2_split" 
    count_dataset(dataset_path)
