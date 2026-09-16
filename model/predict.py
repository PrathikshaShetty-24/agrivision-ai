import tensorflow as tf

model = tf.keras.models.load_model("model/leafintel_model.keras")
print("Model loaded successfully!")

from PIL import Image
import numpy as np

class_names = ['Corn_Common_Rust', 'Corn_Gray_Leaf_Spot', 'Corn_Northern_Leaf_Blight', 'Corn_healthy',
               'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
               'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_healthy']

symptom_explanations = {
    'Corn_Common_Rust': "Small, reddish-brown circular to elongated pustules appear on both leaf surfaces, often giving the leaf a rusty, powdery look. Common in humid conditions.",
    'Corn_Gray_Leaf_Spot': "Rectangular, gray-to-tan lesions run parallel to the leaf veins. Spots may merge in severe cases, causing large areas of dead tissue.",
    'Corn_Northern_Leaf_Blight': "Long, cigar-shaped gray-green or tan lesions form on the leaves, typically starting on lower leaves and moving upward as the disease spreads.",
    'Corn_healthy': "No visible disease symptoms. Leaf shows normal green color and healthy structure.",
    'Potato___Early_blight': "Dark brown spots with concentric rings (like a target) appear, usually starting on older, lower leaves first.",
    'Potato___Late_blight': "Dark, water-soaked patches appear on leaves, often with a pale green-yellow border. Can spread rapidly in cool, wet weather and is historically one of the most destructive potato diseases.",
    'Potato___healthy': "No visible disease symptoms. Leaf shows normal green color and healthy structure.",
    'Tomato_Early_blight': "Dark brown spots with concentric rings (a 'target' pattern) form on older leaves first, often with yellowing around the spot.",
    'Tomato_Late_blight': "Irregular, water-soaked, dark green to brown patches appear, often with white fungal growth visible on the underside of the leaf in humid conditions.",
    'Tomato_Leaf_Mold': "Pale green or yellow spots appear on the upper leaf surface, with a corresponding olive-green to grayish-purple fuzzy mold visible on the underside.",
    'Tomato_healthy': "No visible disease symptoms. Leaf shows normal green color and healthy structure."
}

def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((128, 128))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)
    return predictions
test_image = "dataset/raw/Corn_Gray_Leaf_Spot/" + __import__("os").listdir("dataset/raw/Corn_Gray_Leaf_Spot")[0]


predictions = predict_image(test_image)
print("Raw predictions:", predictions)

probabilities = tf.nn.softmax(predictions[0])
predicted_index = np.argmax(probabilities)
predicted_class = class_names[predicted_index]
confidence = float(probabilities[predicted_index]) * 100

print("Predicted class:", predicted_class)
print("Confidence:", round(confidence, 2), "%")
print("Explanation:", symptom_explanations[predicted_class])