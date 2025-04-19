"""
Improved Data Preprocessing Script for WHM CAPTCHA Project with Albumentations
"""

import os
import shutil
import numpy as np
from PIL import Image
from imutils import paths

import cv2
import albumentations as A

import object.config as config

# Albumentations augmentation pipeline
AUGMENTER = A.Compose([
    A.Affine(translate_percent=(0.05, 0.1), scale=(0.9, 1.1), shear=(-10, 10), p=1),
    A.RandomBrightnessContrast(p=0.5),
    A.GaussianBlur(blur_limit=(3, 7), p=0.3),
    A.GaussNoise(var_limit=(10.0, 50), p=0.3),
])


def ensure_dir(path):
    """Create or clear a directory."""
    if os.path.exists(path):
        print(f"⚠️  Removing existing directory: {path}")
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def save_images_to_folder(foldername, images, labels):
    """Save PIL images into class-labeled subfolders."""
    ensure_dir(foldername)
    counts = {}

    for image, label in zip(images, labels):
        label_path = os.path.join(foldername, label)
        os.makedirs(label_path, exist_ok=True)

        count = counts.get(label, 1)
        filename = os.path.join(label_path, f"{str(count).zfill(6)}.png")
        image.save(filename, "PNG")
        counts[label] = count + 1

    print(f"✅ Saved {sum(counts.values())} images in: {foldername}")


def load_images_and_labels(raw_dir):
    """Load PNG images and extract class labels."""
    print(f"🔍 Scanning directory: {raw_dir}")
    image_paths = list(paths.list_images(raw_dir))
    print(f"🖼️  Found {len(image_paths)} image(s)")

    image_list, label_list = [], []

    for file in image_paths:
        try:
            im = Image.open(file)
            im.load()
            rgba_image = im.convert('RGBA')
            label = os.path.basename(os.path.dirname(file))
            image_list.append(rgba_image)
            label_list.append(label)
        except Exception as e:
            print(f"⚠️  Skipping {file}: {e}")

    return image_list, label_list


def pil_to_cv2(pil_image):
    """Convert PIL RGBA image to OpenCV BGR image."""
    rgb = pil_image.convert("RGB")
    return cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)


def cv2_to_pil(cv2_image):
    """Convert OpenCV BGR image to PIL RGBA image."""
    rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb).convert("RGBA")


def create():
    image_list, label_list = load_images_and_labels(config.RAW_DIR)

    if not image_list:
        print("❌ No valid training images found. Please check your raw_training_data directory structure.")
        return

    print("🔀 Shuffling and splitting...")
    permutation = np.random.permutation(len(image_list))
    image_list = [image_list[i] for i in permutation]
    label_list = [label_list[i] for i in permutation]

    split_idx = int(config.SPLIT_RATE * len(image_list))
    X_train, Y_train = image_list[:split_idx], label_list[:split_idx]
    X_val, Y_val = image_list[split_idx:], label_list[split_idx:]

    save_images_to_folder(config.VAL_DIR, X_val, Y_val)

    X_augmented, Y_augmented = [], []

    print("🌀 Applying rotation and augmentations...")
    for image, label in zip(X_train, Y_train):
        for degree in range(0, 360, 30):  # Rotate every 30 degrees
            rotated = image.rotate(degree)
            X_augmented.append(rotated)
            Y_augmented.append(label)

            # Albumentations augmentation
            cv_image = pil_to_cv2(rotated)
            augmented_cv = AUGMENTER(image=cv_image)["image"]
            augmented_pil = cv2_to_pil(augmented_cv)

            X_augmented.append(augmented_pil)
            Y_augmented.append(label)

    print(f"🖼️  Total training images after augmentation: {len(X_augmented)}")
    save_images_to_folder(config.TRAIN_DIR, X_augmented, Y_augmented)


if __name__ == '__main__':
    create()
