from PIL import ImageOps, Image
import numpy as np

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
