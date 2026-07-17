import os
import yaml
from ultralytics import YOLO

# Setup directory paths for data loading
root_dir = os.getcwd()
train_path = os.path.join(root_dir, 'images', 'train')
val_path = os.path.join(root_dir, 'images', 'val')

# Configure dataset specs and 29 target classes
data_config = {
    'path': root_dir,
    'train': os.path.relpath(train_path, root_dir),
    'val': os.path.relpath(val_path, root_dir),
    'nc': 29,
    'names': [
        'Cherry leaf', 'Peach leaf', 'Corn leaf blight', 'Apple rust leaf', 
        'Potato leaf late blight', 'Strawberry leaf', 'Corn rust leaf', 
        'Tomato leaf late blight', 'Tomato mold leaf', 'Potato leaf early blight', 
        'Apple leaf', 'Tomato leaf yellow virus', 'Blueberry leaf', 
        'Tomato leaf mosaic virus', 'Raspberry leaf', 'Tomato leaf bacterial spot', 
        'Squash Powdery mildew leaf', 'grape leaf', 'Corn leaf', 
        'Tomato early blight leaf', 'grape leaf black rot', 'Potato leaf', 
        'Tomato Septoria leaf spot', 'Tomato leaf', 'Soybean leaf', 
        'Bell_pepper leaf spot', 'Bell_pepper leaf', 'Apple Scab Leaf', 
        'Tomato two spotted spider mites leaf'
    ]
}

# Export configuration dictionary to yaml file
with open('data.yaml', 'w') as outfile:
    yaml.dump(data_config, outfile, default_flow_style=False)

if __name__ == '__main__':
    # Initialize YOLOv8 nano model weights
    model = YOLO('yolov8n.pt') 

    # Execute training loop with hardware optimizations
    model.train(
        data='data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        workers=0,
        cache=True,
        amp=True,
        name='crop_blight_full'
    )