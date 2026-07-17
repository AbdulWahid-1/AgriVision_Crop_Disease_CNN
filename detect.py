import os
import pandas as pd
import cv2
import glob
import matplotlib.pyplot as plt
from ultralytics import YOLO

if __name__ == '__main__':
    # Load fine-tuned weights
    model = YOLO('runs/detect/crop_blight_full/weights/best.pt')

    # Run validation metrics evaluation
    metrics = model.val(workers=0)

    print(f"mAP_50-95: {metrics.box.map:.4f}")
    print(f"mAP_50:    {metrics.box.map50:.4f}")
    print(f"Recall:    {metrics.box.map75:.4f}")

    # Process training logs and generate performance curves
    log_path = 'runs/detect/crop_blight_full/results.csv'
    if os.path.exists(log_path):
        df = pd.read_csv(log_path)
        df.columns = df.columns.str.strip()
        os.makedirs('assets', exist_ok=True)
        
        # Plot training loss progression
        plt.figure(figsize=(8, 5))
        plt.plot(df['epoch'], df['train/box_loss'], label='Box Loss', color='orange')
        plt.plot(df['epoch'], df['train/cls_loss'], label='Class Loss', color='blue')
        plt.title('Loss Curves')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig('assets/training_loss_graph.png', dpi=300)
        plt.close()
        
        # Plot mAP@0.5 score trajectory
        plt.figure(figsize=(8, 5))
        plt.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@0.5', color='green', marker='o')
        plt.title('Accuracy Progression')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig('assets/map_performance_graph.png', dpi=300)
        plt.close()

    # Run single image inference with confidence threshold filter
    val_images = glob.glob('images/val/*.jpg')
    if val_images:
        preds = model(val_images[0], conf=0.5)
        annotated = preds[0].plot()
        cv2.imwrite('assets/github_visual_output.jpg', annotated)