import os
from PIL import Image
dataset_path = "dataset/raw"

classes = os.listdir(dataset_path)
print(classes)

for class_name in classes:
    class_folder = os.path.join(dataset_path, class_name)
    num_images = len(os.listdir(class_folder))
    print(class_name, "->", num_images, "images")

print("\nChecking for corrupted images...")

corrupted_files = []

for class_name in classes:
    class_folder = os.path.join(dataset_path, class_name)
    for filename in os.listdir(class_folder):
        file_path = os.path.join(class_folder, filename)
        try:
            img = Image.open(file_path)
            img.verify()
        except Exception:
            corrupted_files.append(file_path)

print("Total corrupted images found:", len(corrupted_files))
for f in corrupted_files:
    print(f)    