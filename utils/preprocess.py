from PIL import Image
import os

IMAGE_SIZE = (128, 128)

def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.convert("RGB")
    img = img.resize(IMAGE_SIZE)
    return img

test_img = preprocess_image("dataset/raw/Tomato_healthy/" + os.listdir("dataset/raw/Tomato_healthy")[0])
print("New size:", test_img.size)
print("New mode:", test_img.mode)

dataset_path = "dataset/raw"
output_path = "dataset/processed"

classes = os.listdir(dataset_path)

for class_name in classes:
    output_class_folder = os.path.join(output_path, class_name)
    os.makedirs(output_class_folder, exist_ok=True)

for class_name in classes:
    input_class_folder = os.path.join(dataset_path, class_name)
    output_class_folder = os.path.join(output_path, class_name)

    for filename in os.listdir(input_class_folder):
        input_file_path = os.path.join(input_class_folder, filename)
        output_file_path = os.path.join(output_class_folder, filename)

        processed_img = preprocess_image(input_file_path)
        processed_img.save(output_file_path)

    print("Done:", class_name)    