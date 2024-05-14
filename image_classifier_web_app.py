import streamlit as st
from keras.models import load_model
from PIL import Image
from util import classify

st.markdown(
    """
<style>
body {
    font-size: 18px; /* Adjust the value as needed */
}
</style>
    """,
    unsafe_allow_html=True,
)

st.title('Pneumonia Classification')

st.header('Upload images')

files = st.file_uploader('Upload multiple images', type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

model = load_model('./classification model/model13_05.h5')

with open('./classification model/labels.txt', 'r') as f:
    class_names = [a[:-1].split(' ')[1] for a in f.readlines()]

if files:
    for file in files:
        image = Image.open(file).convert('RGB')
        image = image.resize((224, 224))
        st.image(image, use_column_width=True)

        class_name, confidence_score = classify(image, model, class_names)
        st.write(f'## Prediction: {class_name}')
        st.write(f'## Confidence: {confidence_score}')
