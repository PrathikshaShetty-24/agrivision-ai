import tensorflow as tf

dataset_path = "dataset/processed"

train_data = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(128, 128),
    batch_size=32
)
class_names = train_data.class_names
print(class_names)

class_counts = {
    'Corn_Common_Rust': 1306,
    'Corn_Gray_Leaf_Spot': 574,
    'Corn_Northern_Leaf_Blight': 1146,
    'Corn_healthy': 1162,
    'Potato___Early_blight': 1000,
    'Potato___Late_blight': 1000,
    'Potato___healthy': 152,
    'Tomato_Early_blight': 1000,
    'Tomato_Late_blight': 626,
    'Tomato_Leaf_Mold': 952,
    'Tomato_healthy': 1591
}

total_samples = sum(class_counts.values())
num_of_classes = len(class_counts)

class_weight = {}
for index, name in enumerate(class_names):
    count = class_counts[name]
    weight = (total_samples / (num_of_classes * count))**0.5
    class_weight[index] = weight

print(class_weight)

val_data = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(128, 128),
    batch_size=32
)
num_classes = 11

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])

model = tf.keras.Sequential([
    data_augmentation,
    tf.keras.layers.Rescaling(1./255, input_shape=(128, 128, 3)),
    tf.keras.layers.Conv2D(16, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(num_classes)
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    class_weight=class_weight
)

import numpy as np

y_true = []
y_pred = []

for images, labels in val_data:
    predictions = model.predict(images, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    y_true.extend(labels.numpy())
    y_pred.extend(predicted_classes)

from sklearn.metrics import classification_report

print(classification_report(y_true, y_pred, target_names=class_names))  

model.save("model/leafintel_model.keras")
print("Model saved successfully!")