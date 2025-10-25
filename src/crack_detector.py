import os
import cv2
from pathlib import Path
import json
import random
import shutil

def create_crack_annotations_for_train_val_test():
    """Создание аннотаций для трещин с распределением по train/val/test + копирование изображений"""
    
    # Пути к папкам с трещинами (включая тестовую выборку)
    crack_folders = {
        "train": ["data//data//training//Дефект 3"],
        "test": ["data//data//testing//Дефект 3"],
        "val": ["data//data//validation//Дефект 3"]
    }
    
    # Выходные папки для аннотаций и изображений
    output_dirs = {
        "train": {
            "labels": Path("data//dataset_2_split//train//labels"),
            "images": Path("data//dataset_2_split//train//images")
        },
        "test": {
            "labels": Path("data//dataset_2_split//test//labels"),
            "images": Path("data//dataset_2_split//test//images")
        },
        "val": {
            "labels": Path("data//dataset_2_split//val//labels"),
            "images": Path("data//dataset_2_split//val//images")
        }
    }
    
    # Создаем выходные директории
    for split_type, dirs in output_dirs.items():
        dirs["labels"].mkdir(parents=True, exist_ok=True)
        dirs["images"].mkdir(parents=True, exist_ok=True)
    
    # Статистика
    stats = {"train": 0, "val": 0, "test": 0}
    
    for split_type, folders in crack_folders.items():
        print(f"\nОБРАБОТКА {split_type.upper()} ДАННЫХ:")
        
        for crack_folder in folders:
            crack_path = Path(crack_folder)
            
            if not crack_path.exists():
                print(f"Папка не найдена: {crack_path}")
                continue
                
            print(f"Обработка папки: {crack_path}")
            
            # Ищем все изображения в папке
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
            image_files = []
            
            for ext in image_extensions:
                image_files.extend(crack_path.glob(ext))
                image_files.extend(crack_path.glob(ext.upper()))
            
            print(f" Найдено изображений: {len(image_files)}")
            
            for image_path in image_files:
                # Создаем аннотацию
                annotation_content = create_full_image_annotation(image_path)
                
                # Имя файла аннотации
                annotation_filename = image_path.stem + '.txt'
                annotation_path = output_dirs[split_type]["labels"] / annotation_filename
                
                # Сохраняем аннотацию
                with open(annotation_path, 'w', encoding='utf-8') as f:
                    f.write(annotation_content)
                
                dst_image_path = output_dirs[split_type]["images"] / image_path.name
                shutil.copy2(image_path, dst_image_path)
                
                stats[split_type] += 1
                print(f"Создана аннотация и скопировано изображение: {image_path.name} -> {split_type}")
    
    print(f"Статистика:")
    print(f"   Train: {stats['train']} аннотаций")
    print(f"   Val: {stats['val']} аннотаций")
    print(f"   Test: {stats['test']} аннотаций")
    
    print(f"Данные сохранены в:")
    print(f"   Train: {output_dirs['train']['images']}")
    print(f"   Val: {output_dirs['val']['images']}")
    print(f"   Test: {output_dirs['test']['images']}")

def create_crack_annotations_with_split():
    """Альтернативный вариант: автоматическое разделение на train/val/test + копирование изображений"""
    
    # Все папки с трещинами
    all_crack_folders = [
        "data//training//Дефект 3",
        "data//testing//Дефект 3", 
        "data//validation//Дефект 3"
    ]
    
    # Выходные папки
    train_labels_dir = Path("data//dataset_2_split//train//labels")
    val_labels_dir = Path("data//dataset_2_split//val//labels")
    test_labels_dir = Path("data//dataset_2_split//test//labels")
    
    train_images_dir = Path("data//dataset_2_split//train//images")
    val_images_dir = Path("data//dataset_2_split//val//images")
    test_images_dir = Path("data//dataset_2_split//test//images")
    
    for dir_path in [train_labels_dir, val_labels_dir, test_labels_dir,
                     train_images_dir, val_images_dir, test_images_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Собираем все изображения
    all_images = []
    for crack_folder in all_crack_folders:
        crack_path = Path(crack_folder)
        if crack_path.exists():
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                all_images.extend(crack_path.glob(ext))
    
    print(f"Всего найдено изображений с трещинами: {len(all_images)}")
    
    # Перемешиваем и разделяем 70/15/15
    random.shuffle(all_images)
    train_split_idx = int(0.7 * len(all_images))
    val_split_idx = int(0.85 * len(all_images))
    
    train_images = all_images[:train_split_idx]
    val_images = all_images[train_split_idx:val_split_idx]
    test_images = all_images[val_split_idx:]
    
    print(f"Разделение: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test")
    
    # Создаем аннотации и копируем изображения
    for image_path in train_images:
        annotation_content = create_full_image_annotation(image_path)
        annotation_filename = image_path.stem + '.txt'
        annotation_path = train_labels_dir / annotation_filename
        
        with open(annotation_path, 'w', encoding='utf-8') as f:
            f.write(annotation_content)
        
        shutil.copy2(image_path, train_images_dir / image_path.name)
    
    for image_path in val_images:
        annotation_content = create_full_image_annotation(image_path)
        annotation_filename = image_path.stem + '.txt'
        annotation_path = val_labels_dir / annotation_filename
        
        with open(annotation_path, 'w', encoding='utf-8') as f:
            f.write(annotation_content)
        
        shutil.copy2(image_path, val_images_dir / image_path.name)
    
    for image_path in test_images:
        annotation_content = create_full_image_annotation(image_path)
        annotation_filename = image_path.stem + '.txt'
        annotation_path = test_labels_dir / annotation_filename
        
        with open(annotation_path, 'w', encoding='utf-8') as f:
            f.write(annotation_content)
        
        shutil.copy2(image_path, test_images_dir / image_path.name)
    
    print(f"Создано аннотаций и скопировано изображений:")
    print(f"   Train: {len(train_images)}")
    print(f"   Val: {len(val_images)}")
    print(f"   Test: {len(test_images)}")


def create_full_image_annotation(image_path):
    """Создает аннотацию где трещина занимает всю центральную область изображения"""
    image = cv2.imread(str(image_path))
    if image is None:
        return "2 0.5 0.5 0.8 0.8"
    
    h, w = image.shape[:2]
    
    # Создаем bounding box в центре изображения
    x_center = 0.5
    y_center = 0.5
    width = 0.8
    height = 0.8
    
    return f"2 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

class SmartCrackDetector:
    """Умный детектор трещин для рентгеновских снимков"""
    
    def detect_cracks(self, image_path):
        """Детекция трещин с помощью морфологического анализа"""
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return []
        
        h, w = image.shape
        
        # Улучшение контраста
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(image)
        
        # Бинаризация
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Морфологические операции для выделения линейных структур
        kernel_line = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        morph = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_line)
        
        # Поиск контуров
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bboxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Фильтр по площади
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                if aspect_ratio > 2.0 or (aspect_ratio < 0.5 and h > 0):
                    bboxes.append([x, y, w, h])
        
        return bboxes
    
    def bboxes_to_yolo_format(self, bboxes, image_path):
        """Конвертация bbox в YOLO формат"""
        image = cv2.imread(image_path)
        if image is None:
            return "2 0.5 0.5 0.8 0.8"
        
        h, w = image.shape[:2]
        yolo_lines = []
        
        for bbox in bboxes:
            x, y, bbox_w, bbox_h = bbox
            x_center = (x + bbox_w / 2) / w
            y_center = (y + bbox_h / 2) / h
            width = bbox_w / w
            height = bbox_h / h
            yolo_lines.append(f"2 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        if not yolo_lines:
            return "2 0.5 0.5 0.8 0.8"
        
        return "\n".join(yolo_lines)

def create_smart_annotations_for_all():
    """Умное создание аннотаций с детекцией для train, val и test + копирование изображений"""
    
    crack_folders = {
        "train": ["data//training//Дефект 3"],
        "val": ["data//validation//Дефект 3"],
        "test": ["data//testing//Дефект 3"]
    }
    
    output_dirs = {
        "train": {
            "labels": Path("data//dataset_2_split//train//labels"),
            "images": Path("data//dataset_2_split//train//images")
        },
        "val": {
            "labels": Path("data//dataset_2_split//val//labels"),
            "images": Path("data//dataset_2_split//val//images")
        },
        "test": {
            "labels": Path("data//dataset_2_split//test//labels"),
            "images": Path("data//dataset_2_split//test//images")
        }
    }
    
    for dirs in output_dirs.values():
        dirs["labels"].mkdir(parents=True, exist_ok=True)
        dirs["images"].mkdir(parents=True, exist_ok=True)
    
    detector = SmartCrackDetector()
    stats = {"train": 0, "val": 0, "test": 0}
    
    for split_type, folders in crack_folders.items():
        
        for crack_folder in folders:
            crack_path = Path(crack_folder)
            if not crack_path.exists():
                print(f"Папка не найдена: {crack_path}")
                continue
                
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                image_files.extend(crack_path.glob(ext))
            
            print(f"🔍 Обработка папки: {crack_path}")
            print(f"   Найдено изображений: {len(image_files)}")
            
            for image_path in image_files:
                crack_bboxes = detector.detect_cracks(str(image_path))
                annotation_content = detector.bboxes_to_yolo_format(crack_bboxes, str(image_path))
                annotation_filename = image_path.stem + '.txt'
                annotation_path = output_dirs[split_type]["labels"] / annotation_filename
                
                with open(annotation_path, 'w', encoding='utf-8') as f:
                    f.write(annotation_content)
                
                shutil.copy2(image_path, output_dirs[split_type]["images"] / image_path.name)
                
                stats[split_type] += len(crack_bboxes) if crack_bboxes else 1
                print(f" {image_path.name} -> {split_type} ({len(crack_bboxes)} трещин)")
    
    print("\nИтоговая статистика:")
    for k, v in stats.items():
        print(f"   {k}: {v} аннотаций")


def check_all_annotations():
    """Проверка аннотаций во всех папках (train, val, test)"""
    train_dir = Path("data//dataset_2_split//train//labels")
    val_dir = Path("data//dataset_2_split//val//labels")
    test_dir = Path("data//dataset_2_split//test//labels")
    
    print("ПРОВЕРКА АННОТАЦИЙ:")
    
    for split_type, dir_path in [("TRAIN", train_dir), ("VAL", val_dir), ("TEST", test_dir)]:
        if dir_path.exists():
            txt_files = list(dir_path.glob("*.txt"))
            print(f"📊 {split_type}: {len(txt_files)} аннотаций")
            for txt_file in txt_files[:2]:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                lines = content.split('\n') if content else []
                print(f"   📄 {txt_file.name}: {len(lines)} трещин")
        else:
            print(f"{split_type}: папка не найдена")

if name == "main":
    check_all_annotations()
    print("\nВАРИАНТ 1: Соответствие папкам (training -> train, validation -> val, testing -> test)")
    create_crack_annotations_for_train_val_test()
    check_all_annotations()