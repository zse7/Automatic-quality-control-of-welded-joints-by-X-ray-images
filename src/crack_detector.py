import os
import cv2
from pathlib import Path
import json
import random

def create_crack_annotations_for_train_val_test():
    """Создание аннотаций для трещин с распределением по train/val/test"""
    
    # Пути к папкам с трещинами (включая тестовую выборку)
    crack_folders = {
        "train": ["data//data//training//Дефект 3"],
        "test": ["data//data//testing//Дефект 3"],
        "val": ["data//data//validation//Дефект 3"]
    }
    
    # Выходные папки для аннотаций
    output_dirs = {
        "train": Path("data//dataset_2_split//train//labels"),
        "test": Path("data//dataset_2_split//test//labels"),
        "val": Path("data//dataset_2_split//val//labels")
    }
    
    # Создаем выходные директории
    for dir_path in output_dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Статистика
    stats = {"train": 0, "val": 0, "test": 0}
    
    for split_type, folders in crack_folders.items():
        print(f"\n🎯 ОБРАБОТКА {split_type.upper()} ДАННЫХ:")
        
        for crack_folder in folders:
            crack_path = Path(crack_folder)
            
            if not crack_path.exists():
                print(f"❌ Папка не найдена: {crack_path}")
                continue
                
            print(f"🔍 Обработка папки: {crack_path}")
            
            # Ищем все изображения в папке
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
            image_files = []
            
            for ext in image_extensions:
                image_files.extend(crack_path.glob(ext))
                image_files.extend(crack_path.glob(ext.upper()))
            
            print(f"   Найдено изображений: {len(image_files)}")
            
            for image_path in image_files:
                # Создаем аннотацию
                annotation_content = create_full_image_annotation(image_path)
                
                # Имя файла аннотации
                annotation_filename = image_path.stem + '.txt'
                annotation_path = output_dirs[split_type] / annotation_filename
                
                # Сохраняем аннотацию
                with open(annotation_path, 'w', encoding='utf-8') as f:
                    f.write(annotation_content)
                
                stats[split_type] += 1
                print(f"   ✅ Создана аннотация: {annotation_filename} -> {split_type}")
    
    # Отчет
    print(f"\n🎉 ВЫПОЛНЕНО!")
    print(f"📊 Статистика:")
    print(f"   Train: {stats['train']} аннотаций")
    print(f"   Val: {stats['val']} аннотаций")
    print(f"   Test: {stats['test']} аннотаций")
    
    print(f"💾 Аннотации сохранены в:")
    print(f"   Train: {output_dirs['train']}")
    print(f"   Val: {output_dirs['val']}")
    print(f"   Test: {output_dirs['test']}")

def create_crack_annotations_with_split():
    """Альтернативный вариант: автоматическое разделение на train/val/test"""
    
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
    
    for dir_path in [train_labels_dir, val_labels_dir, test_labels_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Собираем все изображения
    all_images = []
    for crack_folder in all_crack_folders:
        crack_path = Path(crack_folder)
        if crack_path.exists():
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                all_images.extend(crack_path.glob(ext))
    
    print(f"📊 Всего найдено изображений с трещинами: {len(all_images)}")
    
    # Перемешиваем и разделяем 70/15/15
    random.shuffle(all_images)
    train_split_idx = int(0.7 * len(all_images))
    val_split_idx = int(0.85 * len(all_images))
    
    train_images = all_images[:train_split_idx]
    val_images = all_images[train_split_idx:val_split_idx]
    test_images = all_images[val_split_idx:]
    
    print(f"🎯 Разделение: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test")
    
    # Создаем аннотации для train
    for image_path in train_images:
        annotation_content = create_full_image_annotation(image_path)
        annotation_filename = image_path.stem + '.txt'
        annotation_path = train_labels_dir / annotation_filename
        
        with open(annotation_path, 'w', encoding='utf-8') as f:
            f.write(annotation_content)
    
    # Создаем аннотации для val
    for image_path in val_images:
        annotation_content = create_full_image_annotation(image_path)
        annotation_filename = image_path.stem + '.txt'
        annotation_path = val_labels_dir / annotation_filename
        
        with open(annotation_path, 'w', encoding='utf-8') as f:
            f.write(annotation_content)
    
    # Создаем аннотации для test
    for image_path in test_images:
        annotation_content = create_full_image_annotation(image_path)
        annotation_filename = image_path.stem + '.txt'
        annotation_path = test_labels_dir / annotation_filename
        
        with open(annotation_path, 'w', encoding='utf-8') as f:
            f.write(annotation_content)
    
    print(f"✅ Создано аннотаций:")
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
    
    # Класс 2 = Crack
    return f"2 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

def create_smart_annotations_for_all():
    """Умное создание аннотаций с детекцией для train, val и test"""
    
    crack_folders = {
        "train": ["data//training//Дефект 3"],
        "val": ["data//validation//Дефект 3"],
        "test": ["data//testing//Дефект 3"]
    }
    
    output_dirs = {
        "train": Path("data//dataset_2_split//train//labels"),
        "val": Path("data//dataset_2_split//val//labels"),
        "test": Path("data//dataset_2_split//test//labels")
    }
    
    # Создаем директории
    for dir_path in output_dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    detector = SmartCrackDetector()
    stats = {"train": 0, "val": 0, "test": 0}
    
    for split_type, folders in crack_folders.items():
        print(f"\n🎯 УМНАЯ ОБРАБОТКА {split_type.upper()}:")
        
        for crack_folder in folders:
            crack_path = Path(crack_folder)
            if not crack_path.exists():
                print(f"❌ Папка не найдена: {crack_path}")
                continue
                
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                image_files.extend(crack_path.glob(ext))
            
            print(f"🔍 Обработка папки: {crack_path}")
            print(f"   Найдено изображений: {len(image_files)}")
            
            for image_path in image_files:
                # Детекция трещин
                crack_bboxes = detector.detect_cracks(str(image_path))
                
                # Создаем аннотации
                annotation_content = detector.bboxes_to_yolo_format(crack_bboxes, str(image_path))
                
                # Сохраняем
                annotation_filename = image_path.stem + '.txt'
                annotation_path = output_dirs[split_type] / annotation_filename
                
                with open(annotation_path, 'w', encoding='utf-8') as f:
                    f.write(annotation_content)
                
                stats[split_type] += len(crack_bboxes) if crack_bboxes else 1
                print(f"   ✅ {image_path.name} -> {len(crack_bboxes) if crack_bboxes else 1} трещин -> {split_type}")

class SmartCrackDetector:
    """Умный детектор трещин для рентгеновских снимков"""
    
    def detect_cracks(self, image_path):
        """Детекция трещин с помощью морфологического анализа"""
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return []
        
        h, w = image.shape
        
        # 1. Улучшение контраста
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(image)
        
        # 2. Бинаризация
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 3. Морфологические операции для выделения линейных структур
        kernel_line = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        morph = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_line)
        
        # 4. Поиск контуров
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bboxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Фильтр по площади
                x, y, w, h = cv2.boundingRect(contour)
                
                # Фильтр по соотношению сторон (трещины вытянутые)
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
            
            # Нормализация
            x_center = (x + bbox_w / 2) / w
            y_center = (y + bbox_h / 2) / h
            width = bbox_w / w
            height = bbox_h / h
            
            yolo_lines.append(f"2 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        if not yolo_lines:
            return "2 0.5 0.5 0.8 0.8"
        
        return "\n".join(yolo_lines)

def check_all_annotations():
    """Проверка аннотаций во всех папках (train, val, test)"""
    train_dir = Path("data//dataset_2_split//train//labels")
    val_dir = Path("data//dataset_2_split//val//labels")
    test_dir = Path("data//dataset_2_split//test//labels")
    
    print("🔍 ПРОВЕРКА АННОТАЦИЙ:")
    
    for split_type, dir_path in [("TRAIN", train_dir), ("VAL", val_dir), ("TEST", test_dir)]:
        if dir_path.exists():
            txt_files = list(dir_path.glob("*.txt"))
            print(f"📊 {split_type}: {len(txt_files)} аннотаций")
            
            # Показываем несколько примеров
            for txt_file in txt_files[:2]:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                lines = content.split('\n') if content else []
                print(f"   📄 {txt_file.name}: {len(lines)} трещин")
        else:
            print(f"❌ {split_type}: папка не найдена")

# ЗАПУСК
if __name__ == "__main__":
    print("🚀 СОЗДАНИЕ АННОТАЦИЙ ДЛЯ TRAIN, VAL И TEST")
    
    # Проверяем существующие аннотации
    check_all_annotations()
    
    print("\n🎯 ВАРИАНТ 1: Соответствие папкам (training -> train, validation -> val, testing -> test)")
    create_crack_annotations_for_train_val_test()
    
    # Финальная проверка
    check_all_annotations()