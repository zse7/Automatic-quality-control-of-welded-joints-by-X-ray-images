from ultralytics import YOLO

model_path = "models/m60.pt"
model = YOLO(model_path)

print("Названия классов:")
for class_id, class_name in model.names.items():
    print(f"{class_id}: {class_name}")