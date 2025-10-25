from ultralytics import YOLO
import json
import cv2
from pathlib import Path
import numpy as np

class GOSTDefectAnalyzer:
    """Анализатор дефектов по ГОСТ нормам"""
    
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.pixel_to_mm = 0.1  # Коэффициент преобразования пикселей в мм
    
    def analyze_image(self, image_path):
        """Анализирует изображение и применяет ГОСТ нормы"""
        results = self.model(image_path)
        defects = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    defect = {
                        'class': self.model.names[int(box.cls)],
                        'class_id': int(box.cls),
                        'confidence': float(box.conf),
                        'bbox': [round(x, 1) for x in box.xyxy[0].tolist()],
                        'bbox_mm': self._pixels_to_mm(box.xyxy[0].tolist()),
                        'severity': self._assess_gost_severity(self.model.names[int(box.cls)], 
                                                             box.xyxy[0].tolist())
                    }
                    defects.append(defect)
        
        return defects, results[0].plot() if results else None
    
    def _pixels_to_mm(self, bbox):
        """Конвертирует bbox из пикселей в мм"""
        x1, y1, x2, y2 = bbox
        width_mm = (x2 - x1) * self.pixel_to_mm
        height_mm = (y2 - y1) * self.pixel_to_mm
        return [round(width_mm, 2), round(height_mm, 2)]
    
    def _assess_gost_severity(self, defect_type, bbox):
        """Оценивает серьезность дефекта по ГОСТ"""
        width_mm, height_mm = self._pixels_to_mm(bbox)
        area_mm2 = width_mm * height_mm
        
        if defect_type == "Porosity":
            # Поры: диаметр отдельной поры ≤ 1.5 мм - допустимо, > 3 мм - недопустимо
            diameter = max(width_mm, height_mm)
            if diameter <= 1.5:
                return "ДОПУСТИМО"
            elif diameter > 3.0:
                return "НЕДОПУСТИМО"
            else:
                return "ТРЕБУЕТ ПРОВЕРКИ"
                
        elif defect_type == "Slag Inclusion":
            # Шлаковые включения: длина ≤ 3 мм, ширина ≤ 0.5 мм - допустимо
            if width_mm <= 3.0 and height_mm <= 0.5:
                return "ДОПУСТИМО"
            elif width_mm > 5.0 or height_mm > 1.0:
                return "НЕДОПУСТИМО"
            else:
                return "ТРЕБУЕТ ПРОВЕРКИ"
                
        elif defect_type == "Crack":
            # Трещины: любая трещина - недопустимо
            return "НЕДОПУСТИМО"
            
        elif defect_type == "Lack of Fusion":
            # Непровар: длина ≤ 10 мм - допустимо, > 25 мм - недопустимо
            if width_mm <= 10.0:
                return "ДОПУСТИМО"
            elif width_mm > 25.0:
                return "НЕДОПУСТИМО"
            else:
                return "ТРЕБУЕТ ПРОВЕРКИ"
                
        elif defect_type == "Undercut":
            # Подрез: глубина ≤ 0.5 мм - допустимо, > 1 мм - недопустимо
            depth = height_mm  # предполагаем что высота = глубина подреза
            if depth <= 0.5:
                return "ДОПУСТИМО"
            elif depth > 1.0:
                return "НЕДОПУСТИМО"
            else:
                return "ТРЕБУЕТ ПРОВЕРКИ"
                
        else:  # Other defects
            return "ТРЕБУЕТ ПРОВЕРКИ"
    
    def generate_gost_report(self, image_path, defects):
        """Генерирует отчет по ГОСТ"""
        report = {
            "image": str(image_path),
            "total_defects": len(defects),
            "gost_assessment": {
                "acceptable": 0,
                "requires_inspection": 0, 
                "unacceptable": 0
            },
            "defects_by_type": {},
            "defects_details": defects
        }
        
        # Подсчет по ГОСТ категориям
        for defect in defects:
            severity = defect['severity']
            if severity == "ДОПУСТИМО":
                report["gost_assessment"]["acceptable"] += 1
            elif severity == "ТРЕБУЕТ ПРОВЕРКИ":
                report["gost_assessment"]["requires_inspection"] += 1
            elif severity == "НЕДОПУСТИМО":
                report["gost_assessment"]["unacceptable"] += 1
            
            # Подсчет по типам
            defect_type = defect['class']
            if defect_type not in report["defects_by_type"]:
                report["defects_by_type"][defect_type] = 0
            report["defects_by_type"][defect_type] += 1
        
        # Общая оценка шва
        if report["gost_assessment"]["unacceptable"] > 0:
            report["weld_quality"] = "БРАК"
        elif report["gost_assessment"]["requires_inspection"] > 0:
            report["weld_quality"] = "ТРЕБУЕТ ДОПОЛНИТЕЛЬНОЙ ПРОВЕРКИ"
        else:
            report["weld_quality"] = "ГОДЕН"
        
        return report

def main():
    model_path = "models/trained/welding_defects/weights/best.pt" 
    image_path = "data//data//val//images/example.jpg"
    
    # Ищем любое изображение для теста
    test_images = list(Path("data//data//val//images").glob("*.jpg")) + \
                  list(Path("data//data//val//images").glob("*.png"))
        
    image_path = test_images[0]
    print(f"Анализируем: {image_path}")
    
    # Анализ
    analyzer = GOSTDefectAnalyzer(model_path)
    defects, annotated_img = analyzer.analyze_image(image_path)
    
    # Сохраняем результат
    if annotated_img is not None:
        output_img = "gost_analysis_result.jpg"
        cv2.imwrite(output_img, annotated_img)
        print(f"Результат сохранен: {output_img}")
    
    # Генерируем ГОСТ отчет
    report = analyzer.generate_gost_report(image_path, defects)
    
    # Сохраняем JSON отчет
    with open("gost_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # Выводим сводку
    print(f"\nРЕЗУЛЬТАТЫ АНАЛИЗА ПО ГОСТ:")
    print(f"Всего дефектов: {report['total_defects']}")
    print(f"Качество шва: {report['weld_quality']}")
    print(f"\nОценка по ГОСТ:")
    print(f"  Допустимо: {report['gost_assessment']['acceptable']}")
    print(f"  Требует проверки: {report['gost_assessment']['requires_inspection']}")
    print(f"  Недопустимо: {report['gost_assessment']['unacceptable']}")
    
    print(f"\nДетали по типам дефектов:")
    for defect_type, count in report['defects_by_type'].items():
        print(f"  {defect_type}: {count}")
    
    print(f"\nОтчет сохранен: gost_report.json")

if __name__ == "__main__":
    main()