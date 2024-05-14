from PIL import ImageOps, Image
import numpy as np
import os
from PIL import ImageOps, Image
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

def classify(image, model, class_names):

    ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)

    image_array = np.array(image)

    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    prediction = model.predict(data)
    # index = np.argmax(prediction)
    index = 1 if prediction[0][1] > 0.95 else 0
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    return class_name, confidence_score

def load_images_from_directory(directory, class_name):
    images = []
    labels = []
    for filename in os.listdir(directory):
        img_path = os.path.join(directory, filename)
        img = Image.open(img_path).convert('RGB')
        img = img.resize((224, 224))
        images.append(img)
        labels.append(class_name)
    return images, labels

if __name__ == '__main__':
    # Load your Teachable Machine model
    model = tf.keras.models.load_model('classification model/model13_05.h5')
    class_names = ['NORMAL', 'PNEUMONIA']  # Match your model's output

    # Data loading and prediction
    val_pneumonia_images, pneumonia_labels = load_images_from_directory("chest_xray/val/PNEUMONIA", 'PNEUMONIA')
    val_normal_images, normal_labels = load_images_from_directory("chest_xray/val/NORMAL", 'NORMAL')

    val_images = val_pneumonia_images + val_normal_images
    val_labels = pneumonia_labels + normal_labels

    y_true = []
    y_pred = []

    for i, image in enumerate(val_images):
        class_name, confidence_score = classify(image, model, class_names)
        y_true.append(val_labels[i])
        y_pred.append(class_name)

    # Performance Evaluation
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    print("Confusion Matrix:\n", cm)
    print(classification_report(y_true, y_pred))

