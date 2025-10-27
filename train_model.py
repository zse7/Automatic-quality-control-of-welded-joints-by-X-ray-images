import torch
from ultralytics import YOLO
import os
from pathlib import Path

class FastYOLOTrainer:
    def __init__(self):
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
        self.device = 'cpu'
        self.paths = {
            'dataset_root': 'data/dataset_2_split',
            'pretrained_model': 'yolov8s.pt', 
            'output_dir': 'models/fast_training',  
        }
        Path(self.paths['output_dir']).mkdir(parents=True, exist_ok=True)
    
    def get_fast_config(self):
        return {
            'data': str(Path(self.paths['dataset_root']) / 'data.yaml'),
            'model': self.paths['pretrained_model'],
            'epochs': 10,         
            'patience': 10,
            'val': True , 
            'project': self.paths['output_dir'],
            'name': 'fast',
            'device': 'mps',
            'batch': 32,          
            'workers': 0,          
            'imgsz': 320,           
            'amp': True,
            'cache': False,        
            'close_mosaic': 0,    
            'mosaic': 0.0,        
            'mixup': 0.0,
            'copy_paste': 0.0,
            'hsv_h': 0.0,
            'hsv_s': 0.0,
            'hsv_v': 0.0,
            'degrees': 0.0,
            'translate': 0.0,
            'scale': 0.0,
            'fliplr': 0.0,
            'optimizer': 'SGD',     
            'lr0': 0.02,           
            'verbose': False,       
            'plots': False,         
            'save_period': 5,
            'exist_ok': True,
        }
    
    def create_data_yaml(self):
        yaml_content = {
            'path': self.paths['dataset_root'],
            'train': 'train/images',
            'val': 'val/images', 
            'test': 'test/images',
            'names': {0: 'пора', 1: 'вклюение', 2: 'подрез', 3: 'прожог', 4: 'трещина', 5: 'наплыв', 6: 'эталон 1', 7: 'эталон 2', 8: 'эталон 3', 9: 'пора-скрытая', 10: 'утяжина', 11: 'несплавление', 12: 'непровар корня'},
            'nc': 13,
        }
        yaml_path = Path(self.paths['dataset_root']) / 'data.yaml'
        with open(yaml_path, 'w', encoding='utf-8') as f:
            import yaml
            yaml.dump(yaml_content, f, allow_unicode=True, default_flow_style=False)
    
    def train(self):
        print("Запуск быстрого обучения")
        self.create_data_yaml()
        
        if self.device == 'mps':
            torch.mps.empty_cache()
        
        model = YOLO(self.paths['pretrained_model'])
        config = self.get_fast_config()
        
        results = model.train(**config)
        return model, results

if __name__ == "__main__":
    trainer = FastYOLOTrainer()
    model, results = trainer.train()