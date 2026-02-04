import kagglehub
import random
from pathlib import Path
import os
import shutil

# # Download latest version
# path = kagglehub.dataset_download("abdallahalidev/plantvillage-dataset")

# print("Path to dataset files:", path)



# CONFIG
PLANT_NAME = "Corn_(maize)___"   # change to "Potato___", "Pepper___", etc.
IMAGES_PER_CLASS = 300

SOURCE_DIR = Path(r"C:\Users\Frogg\.cache\kagglehub\datasets\abdallahalidev\plantvillage-dataset\versions\3\plantvillage dataset\color")
TARGET_DIR = Path("data/Corn_raw")

random.seed(42)
TARGET_DIR.mkdir(parents=True, exist_ok=True)

for class_dir in SOURCE_DIR.iterdir():
    if class_dir.is_dir() and class_dir.name.startswith(PLANT_NAME):
        images = list(class_dir.glob("*.jpg"))
        random.shuffle(images)
        images = images[:IMAGES_PER_CLASS]

        out_dir = TARGET_DIR / class_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)

        for img in images:
            shutil.copy(img, out_dir / img.name)