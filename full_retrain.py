#!/usr/bin/env python3
"""
Full retrain script - Creates dataset with all 25 classes and retrains the model
"""

import os
import pickle
import cv2
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("Full Retrain - Creating dataset with ALL 25 classes...")
print("This will take a few minutes...\n")

# Use new MediaPipe API
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Check if hand landmarker model exists
MODEL_PATH = 'hand_landmarker.task'
if not os.path.exists(MODEL_PATH):
    print("Error: hand_landmarker.task not found!")
    exit(1)

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3
)
detector = vision.HandLandmarker.create_from_options(options)

DATA_DIR = './src/data'
data = []
labels = []

# Process ALL 25 classes
num_classes = 25
max_images_per_class = 100  # Use more images for better accuracy

print(f"Processing {num_classes} classes...")

for actual_label in range(num_classes):
    print(f'Processing class {actual_label} (letter: {chr(65 + actual_label) if actual_label < 24 else "SPACE"})...', end=' ')
    count = 0
    
    for dir_ in os.listdir(DATA_DIR):
        DATA_DIR_path = os.path.join(DATA_DIR, dir_)
        if not os.path.isdir(DATA_DIR_path):
            continue
        
        class_path = os.path.join(DATA_DIR_path, str(actual_label))
        if not os.path.exists(class_path):
            continue
        
        for img_path in os.listdir(class_path):
            if count >= max_images_per_class:
                break
                
            if not img_path.endswith(('.jpg', '.jpeg', '.png')):
                continue
            
            img = cv2.imread(os.path.join(class_path, img_path))
            if img is None:
                continue
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            
            results = detector.detect(mp_image)
            if results.hand_landmarks:
                for hand_landmarks in results.hand_landmarks:
                    data_aux = []
                    x_ = []
                    y_ = []
                    
                    for landmark in hand_landmarks:
                        x_.append(landmark.x)
                        y_.append(landmark.y)
                    
                    for landmark in hand_landmarks:
                        data_aux.append(landmark.x - min(x_))
                        data_aux.append(landmark.y - min(y_))
                    
                    if len(data_aux) == 42:  # Ensure correct size
                        data.append(data_aux)
                        labels.append(actual_label)
                        count += 1
    
    print(f"{count} samples")

print(f"\n✓ Dataset created with {len(data)} total samples")

# Save dataset
DATASET_DIR = './src/data_pickle'
Path(DATASET_DIR).mkdir(parents=True, exist_ok=True)
with open(f'{DATASET_DIR}/full_dataset.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
print(f"✓ Dataset saved to {DATASET_DIR}/full_dataset.pickle")

# Train model
print("\nTraining Random Forest model...")
data_array = np.asarray(data)
labels_array = np.asarray(labels)

x_train, x_test, y_train, y_test = train_test_split(
    data_array, labels_array, test_size=0.2, shuffle=True, stratify=labels_array
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)

print(f'✓ Accuracy: {score*100:.2f}%')

# Save model
MODEL_DIR = './src/models'
Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
with open(f'{MODEL_DIR}/model_f.p', 'wb') as f:
    pickle.dump({'model': model}, f)

print(f"✓ Model saved to {MODEL_DIR}/model_f.p")
print("\n" + "="*60)
print("✓ DONE! Full model trained with 25 classes!")
print("="*60)
print("\nRun: python -m src.cli_enhanced infer --model forest")
