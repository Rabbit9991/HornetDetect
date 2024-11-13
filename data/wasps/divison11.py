import os
import random
import shutil

# 원본 디렉토리
train_images_dir = 'train\images'
train_labels_dir = 'train/labels'

# 타겟 디렉토리
valid_images_dir = 'valid/images'
valid_labels_dir = 'valid/labels'

# 디렉토리 생성
os.makedirs(valid_images_dir, exist_ok=True)
os.makedirs(valid_labels_dir, exist_ok=True)

# 이미지 파일 목록 가져오기
image_files = [f for f in os.listdir(train_images_dir) if os.path.isfile(os.path.join(train_images_dir, f))]

# 섞기
random.shuffle(image_files)

# 나누기 비율 (예: 80% train, 20% valid)
split_ratio = 0.8
split_index = int(len(image_files) * split_ratio)

# train과 valid로 나누기
train_files = image_files[:split_index]
valid_files = image_files[split_index:]

# train 디렉토리에 남아있는 이미지 및 라벨 정리
for f in valid_files:
    image_path = os.path.join(train_images_dir, f)
    label_path = os.path.join(train_labels_dir, f.replace('.jpg', '.txt'))  # 이미지 확장자에 맞게 수정

    shutil.move(image_path, os.path.join(valid_images_dir, f))
    shutil.move(label_path, os.path.join(valid_labels_dir, f.replace('.jpg', '.txt')))  # 이미지 확장자에 맞게 수정

print(f"Total images: {len(image_files)}")
print(f"Training images: {len(train_files)}")
print(f"Validation images: {len(valid_files)}")
