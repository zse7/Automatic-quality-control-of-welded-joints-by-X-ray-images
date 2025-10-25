import os
import shutil
import pandas as pd
import random
from collections import defaultdict

def move_orphan_images(dataset_path, orphans_dir='notebooks/orphans'):
    """
    Перемещает изображения без соответствующих .txt файлов в папку orphans
    """
    print("=== Перемещение изображений без меток ===")
    
    os.makedirs(orphans_dir, exist_ok=True)
    
    splits = ['train', 'val', 'test']
    moved_count = 0
    
    for split in splits:
        images_path = os.path.join(dataset_path, split, 'images')
        labels_path = os.path.join(dataset_path, split, 'labels')
        
        if not os.path.exists(images_path):
            continue
            
        for image_file in os.listdir(images_path):
            if image_file.endswith(('.jpg', '.jpeg', '.png')):
                base_name = os.path.splitext(image_file)[0]
                label_file = base_name + '.txt'
                label_path = os.path.join(labels_path, label_file)
                
                # Если нет соответствующего .txt файла
                if not os.path.exists(label_path):
                    src_image_path = os.path.join(images_path, image_file)
                    dst_image_path = os.path.join(orphans_dir, image_file)
                    
                    shutil.move(src_image_path, dst_image_path)
                    moved_count += 1
                    print(f"Перемещено: {image_file}")
    
    print(f"Всего перемещено изображений без меток: {moved_count}")
    return moved_count

def move_orphan_labels(dataset_path, orphans_dir='notebooks/orphans'):
    """
    Перемещает .txt файлы без соответствующих изображений в папку orphans
    """
    print("\n=== Перемещение orphan .txt файлов ===")
    
    os.makedirs(orphans_dir, exist_ok=True)
    
    splits = ['train', 'val', 'test']
    moved_count = 0
    
    for split in splits:
        images_path = os.path.join(dataset_path, split, 'images')
        labels_path = os.path.join(dataset_path, split, 'labels')
        
        if not os.path.exists(labels_path):
            continue
            
        for label_file in os.listdir(labels_path):
            if label_file.endswith('.txt'):
                base_name = os.path.splitext(label_file)[0]
                
                # Проверяем все возможные расширения изображений
                image_found = False
                for ext in ['.jpg', '.jpeg', '.png']:
                    image_path = os.path.join(images_path, base_name + ext)
                    if os.path.exists(image_path):
                        image_found = True
                        break
                
                # Если нет соответствующего изображения
                if not image_found:
                    src_label_path = os.path.join(labels_path, label_file)
                    dst_label_path = os.path.join(orphans_dir, label_file)
                    
                    shutil.move(src_label_path, dst_label_path)
                    moved_count += 1
                    print(f"Перемещено: {label_file}")
    
    print(f"Всего перемещено orphan .txt файлов: {moved_count}")
    return moved_count


def remap_labels_polygons(source_dataset_path, target_dataset_path, class_mapping=None):
    """
    Создает новый датасет с переразмеченными классами
    class_mapping: словарь {старый_класс: новый_класс}
    """
    print("\n=== Переразметка классов ===")
    
    if class_mapping is None:
        # Пример маппинга классов - настрой под свои нужды
        class_mapping = {
            '0': '0',  # старый класс 0 -> новый класс 0
            '1': '1',  # старый класс 1 -> новый класс 1
            '2': '1',  # старый класс 2 -> новый класс 1 (объединение классов)
            '3': '2',  # старый класс 3 -> новый класс 2
        }
    
    splits = ['train', 'val', 'test']
    
    # Создаем структуру целевого датасета
    for split in splits:
        os.makedirs(os.path.join(target_dataset_path, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(target_dataset_path, split, 'labels'), exist_ok=True)
    
    total_remapped = 0
    
    for split in splits:
        source_images_path = os.path.join(source_dataset_path, split, 'images')
        source_labels_path = os.path.join(source_dataset_path, split, 'labels')
        target_images_path = os.path.join(target_dataset_path, split, 'images')
        target_labels_path = os.path.join(target_dataset_path, split, 'labels')
        
        if not os.path.exists(source_images_path):
            continue
        
        # Копируем изображения
        for image_file in os.listdir(source_images_path):
            if image_file.endswith(('.jpg', '.jpeg', '.png')):
                src_image_path = os.path.join(source_images_path, image_file)
                dst_image_path = os.path.join(target_images_path, image_file)
                shutil.copy2(src_image_path, dst_image_path)
        
        # Обрабатываем и копируем labels с переразметкой
        for label_file in os.listdir(source_labels_path):
            if label_file.endswith('.txt'):
                src_label_path = os.path.join(source_labels_path, label_file)
                dst_label_path = os.path.join(target_labels_path, label_file)
                
                with open(src_label_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                new_lines = []
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) >= 5:  # YOLO формат: class x_center y_center width height
                        old_class = parts[0]
                        new_class = class_mapping.get(old_class, old_class)  # применяем маппинг
                        
                        if new_class != old_class:
                            total_remapped += 1
                        
                        new_line = f"{new_class} " + " ".join(parts[1:]) + "\n"
                        new_lines.append(new_line)
                
                # Записываем переразмеченные аннотации
                with open(dst_label_path, 'w', encoding='utf-8') as f:
                    f.writelines(new_lines)
    
    print(f"Создан новый датасет: {target_dataset_path}")
    print(f"Всего переразмечено аннотаций: {total_remapped}")
    print(f"Маппинг классов: {class_mapping}")
    
    return total_remapped

# ==================== СКРИПТ 4: Стратифицированное разделение ====================

def create_stratified_split(dataset_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """
    Создает стратифицированное разделение на train/val/test
    """
    print("\n=== Стратифицированное разделение ===")
    
    random.seed(random_state)
    
    # Проверяем соотношения
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        raise ValueError(f"Сумма соотношений должна быть 1.0, получено {total_ratio}")
    
    # Создаем временную папку для работы
    temp_dir = os.path.join(dataset_path, 'temp_reorganization')
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    
    # Создаем структуру во временной папке
    splits = ['train', 'val', 'test']
    for split in splits:
        os.makedirs(os.path.join(temp_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(temp_dir, split, 'labels'), exist_ok=True)
    
    # Собираем ВСЕ файлы из исходных папок в один список
    all_files = []
    source_folders = ['train', 'val']
    
    for folder in source_folders:
        images_path = os.path.join(dataset_path, folder, 'images')
        labels_path = os.path.join(dataset_path, folder, 'labels')
        
        if not os.path.exists(images_path):
            continue
            
        for file in os.listdir(images_path):
            if file.endswith(('.jpg', '.jpeg', '.png')):
                base_name = os.path.splitext(file)[0]
                label_file = base_name + '.txt'
                label_path = os.path.join(labels_path, label_file)
                
                # Определяем категорию
                if os.path.exists(label_path):
                    if os.path.getsize(label_path) == 0:
                        category = 'empty'
                    else:
                        category = 'with_annotations'
                else:
                    category = 'no_file'
                
                all_files.append({
                    'source_folder': folder,
                    'base_name': base_name,
                    'image_name': file,
                    'category': category,
                    'image_path': os.path.join(images_path, file),
                    'label_path': label_path if os.path.exists(label_path) else None
                })
    
    # Группируем по категориям для стратификации
    files_by_category = defaultdict(list)
    for file_info in all_files:
        files_by_category[file_info['category']].append(file_info)
    
    # Статистика исходных данных
    total_files = len(all_files)
    print("Исходная статистика датасета:")
    for category, files in files_by_category.items():
        percentage = (len(files) / total_files) * 100
        print(f"  {category}: {len(files)} файлов ({percentage:.1f}%)")
    
    # СТРАТИФИЦИРОВАННОЕ РАЗДЕЛЕНИЕ
    train_files = []
    val_files = []
    test_files = []
    
    for category, category_files in files_by_category.items():
        # Перемешиваем файлы категории
        random.shuffle(category_files)
        
        # Вычисляем количество файлов для каждого сплита
        n_total = len(category_files)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        n_test = n_total - n_train - n_val  # Оставшиеся файлы идут в test
        
        # Разделяем файлы
        train_files.extend(category_files[:n_train])
        val_files.extend(category_files[n_train:n_train + n_val])
        test_files.extend(category_files[n_train + n_val:])
        
        print(f"\nРазделение категории '{category}':")
        print(f"  Train: {n_train} файлов ({n_train/n_total*100:.1f}%)")
        print(f"  Val: {n_val} файлов ({n_val/n_total*100:.1f}%)") 
        print(f"  Test: {n_test} файлов ({n_test/n_total*100:.1f}%)")
    
    # Функция для копирования файлов во ВРЕМЕННЫЕ папки
    def copy_files(files_list, target_split):
        for file_info in files_list:
            # Пути к исходным файлам
            src_image_path = file_info['image_path']
            src_label_path = file_info['label_path']
            
            # Пути к целевым файлам во ВРЕМЕННОЙ папке
            dst_image_path = os.path.join(temp_dir, target_split, 'images', file_info['image_name'])
            dst_label_path = os.path.join(temp_dir, target_split, 'labels', file_info['base_name'] + '.txt')
            
            # Копируем изображение
            shutil.copy2(src_image_path, dst_image_path)
            
            # Копируем txt файл если он существует
            if src_label_path and os.path.exists(src_label_path):
                shutil.copy2(src_label_path, dst_label_path)
    
    # Копируем файлы во ВРЕМЕННЫЕ папки
    print("\nКопирование файлов во временную структуру...")
    copy_files(train_files, 'train')
    copy_files(val_files, 'val') 
    copy_files(test_files, 'test')
    
    # Удаляем оригинальные папки и заменяем их временными
    print("Заменяем оригинальные папки...")
    for split in splits:
        original_split_path = os.path.join(dataset_path, split)
        temp_split_path = os.path.join(temp_dir, split)
        
        # Удаляем оригинальную папку если существует
        if os.path.exists(original_split_path):
            shutil.rmtree(original_split_path)
        
        # Перемещаем временную папку на место оригинальной
        shutil.move(temp_split_path, original_split_path)
    
    # Удаляем временную папку
    shutil.rmtree(temp_dir)
    
    return {
        'train': train_files,
        'val': val_files, 
        'test': test_files
    }

# ==================== СКРИПТ 5: Расчет стратификации ====================

def calculate_final_stratification(dataset_path):
    """
    Расчет финальной стратификации после разделения
    """
    splits = ['train', 'val', 'test']
    results = {}
    
    for split in splits:
        images_path = os.path.join(dataset_path, split, 'images')
        labels_path = os.path.join(dataset_path, split, 'labels')
        
        if not os.path.exists(images_path):
            continue
            
        categories = defaultdict(int)
        total_files = 0
        
        for image_file in os.listdir(images_path):
            if image_file.endswith(('.jpg', '.jpeg', '.png')):
                base_name = os.path.splitext(image_file)[0]
                label_path = os.path.join(labels_path, base_name + '.txt')
                
                if os.path.exists(label_path):
                    if os.path.getsize(label_path) == 0:
                        category = 'empty'
                    else:
                        category = 'with_annotations'
                else:
                    category = 'no_file'
                
                categories[category] += 1
                total_files += 1
        
        results[split] = {
            'total': total_files,
            'categories': dict(categories),
            'percentages': {k: (v/total_files)*100 for k, v in categories.items()}
        }
    
    return results

# ==================== СКРИПТ 6: Конвертация в CSV ====================

def dataset_to_csv(dataset_path, output_csv='dataset_stratified.csv'):
    """
    Конвертирует весь датасет в CSV
    """
    print("\n=== Конвертация в CSV ===")
    
    data = []
    
    splits = ['train', 'val', 'test']
    
    for split in splits:
        images_path = os.path.join(dataset_path, split, 'images')
        labels_path = os.path.join(dataset_path, split, 'labels')
        
        if not os.path.exists(images_path):
            continue
            
        for image_file in os.listdir(images_path):
            if image_file.endswith(('.jpg', '.jpeg', '.png')):
                base_name = os.path.splitext(image_file)[0]
                image_path = os.path.join(images_path, image_file)
                label_path = os.path.join(labels_path, base_name + '.txt')
                
                # Определяем статус аннотаций
                label_status = "no_file"
                annotations = ""
                file_size = 0
                
                if os.path.exists(label_path):
                    file_size = os.path.getsize(label_path)
                    if file_size == 0:
                        label_status = "empty"
                    else:
                        label_status = "has_annotations"
                        with open(label_path, 'r', encoding='utf-8') as f:
                            annotations = f.read().strip()
                
                data.append({
                    'split': split,
                    'image_path': image_path,
                    'label_path': label_path,
                    'label_status': label_status,
                    'file_size_bytes': file_size,
                    'annotations': annotations,
                    'filename': base_name,
                    'has_annotations': label_status == "has_annotations",
                    'has_empty_txt': label_status == "empty",
                    'no_txt_file': label_status == "no_file"
                })
    
    # Создаем DataFrame и сохраняем в CSV
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False, encoding='utf-8')
    
    print(f"Датасет сохранен в {output_csv}")
    print(f"Всего записей: {len(df)}")
    
    return df

# ==================== ОСНОВНОЙ БЛОК ИСПОЛНЕНИЯ ====================

if __name__ == "__main__":
    dataset_path = "data/data"
    
    print("🚀 ЗАПУСК ВСЕХ СКРИПТОВ ПРЕПРОЦЕССИНГА ДАННЫХ\n")
    
    # 1. Очистка от orphan файлов
    print("1. Очистка от orphan файлов...")
    move_orphan_images(dataset_path)
    move_orphan_labels(dataset_path)
    
    # 2. Стратифицированное разделение
    print("\n2. Стратифицированное разделение...")
    splits = create_stratified_split(
        dataset_path, 
        train_ratio=0.7, 
        val_ratio=0.15, 
        test_ratio=0.15
    )
    
    # 3. Проверка стратификации
    print("\n3. Проверка стратификации...")
    stratification = calculate_final_stratification(dataset_path)
    
    for split, stats in stratification.items():
        print(f"\n{split.upper()}: {stats['total']} файлов")
        for category, count in stats['categories'].items():
            percentage = stats['percentages'][category]
            print(f"  {category}: {count} файлов ({percentage:.1f}%)")
    
    # 4. Создание CSV
    print("\n4. Создание CSV файла...")
    df = dataset_to_csv(dataset_path, "dataset_stratified.csv")
    
    # 5. Опционально: переразметка классов (раскомментировать если нужно)
    # print("\n5. Переразметка классов...")
    remap_labels_polygons(
        source_dataset_path=dataset_path,
        target_dataset_path="data/dataset_2_remap",
        class_mapping={'0': '0', '1': '1', '2': '1', '3': '2'}  # настрой под свои классы
    )
    
    print("\n✅ ВСЕ СКРИПТЫ ВЫПОЛНЕНЫ УСПЕШНО!")
    print(f"   Train: {len(splits['train'])} файлов (70%)")
    print(f"   Val: {len(splits['val'])} файлов (15%)")
    print(f"   Test: {len(splits['test'])} файлов (15%)")