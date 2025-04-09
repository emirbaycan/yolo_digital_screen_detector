import os
import random
import shutil

IMG_DIR = "dataset/images/train"
LABEL_DIR = "dataset/labels/train"

VAL_RATIO = 0.2  # 20% for validation
img_files = sorted([f for f in os.listdir(IMG_DIR) if f.endswith(".png")])
random.shuffle(img_files)
val_count = int(len(img_files) * VAL_RATIO)

for img in img_files[:val_count]:
    shutil.move(os.path.join(IMG_DIR, img), "dataset/images/val/")
    shutil.move(os.path.join(LABEL_DIR, img.replace(".png", ".txt")), "dataset/labels/val/")