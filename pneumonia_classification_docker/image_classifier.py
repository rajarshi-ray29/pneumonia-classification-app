import streamlit as st
from keras.models import load_model
from PIL import Image
import os
from util import classify
import json

def process_image(image_path, model, class_names):
    """Processes a single image file"""
    image = Image.open(image_path).convert('RGB')
    image = image.resize((224, 224))
    class_name, confidence_score = classify(image, model, class_names)
    return class_name, confidence_score

def main(input_folder, output_file):
    """Processes images in a folder and generates the output text file"""
    print("Input Folder:", input_folder) 
    print("Output File:", output_file)

    model = load_model('./classification model/model13_05.h5')
    with open('./classification model/labels.txt', 'r') as f:
        class_names = [a[:-1].split(' ')[1] for a in f.readlines()]

    results = {}
    for filename in os.listdir(input_folder):
        if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
            image_path = os.path.join(input_folder, filename)
            class_name, confidence_score = process_image(image_path, model, class_names)
            results[filename] = {
                "prediction": class_name,
                "confidence": float(confidence_score) 
            }

    # Save to JSON file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Classify images using a pneumonia model.')
    parser.add_argument('input_folder', help='Path to the folder containing images.')
    parser.add_argument('output_file', help='Path to the output JSON file.')
    args = parser.parse_args()

    main(args.input_folder, args.output_file)
