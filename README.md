---
title: Deepfake Detector App
emoji: 😻
colorFrom: purple
colorTo: red
sdk: gradio
sdk_version: 5.23.1
app_file: app.py
pinned: false
license: mit
---
# Deepfake Image Detector

## Overview
This is a deepfake image detection application built using TensorFlow and Gradio. The app allows users to upload an image and check whether it is real or fake based on a trained deep learning model.

## Dataset
The model is trained on the dataset available at [Kaggle: Deepfake and Real Images](https://www.kaggle.com/datasets/deepfake-and-real-images). This dataset contains labeled images of both real and deepfake-generated faces, which were used to train the deep learning model.

## Features
- Upload an image to analyze whether it is real or fake.
- Uses a pre-trained deep learning model (`deepfake_model.h5`).
- Provides a confidence score for the classification.
- Interactive UI built using Gradio.

## Installation
Ensure that you have Python installed (preferably Python 3.8 or higher). Install the required dependencies by running:
```bash
pip install tensorflow numpy matplotlib gradio
```

## Usage
1. Ensure you have the trained model file (`deepfake_model.h5`) in the same directory as the script.
2. Run the application using:
```bash
python app.py
```
3. The application will launch a web interface. Upload an image and click "Analyze Image" to get the result.

## Model Details
- The model is a Convolutional Neural Network (CNN) trained on deepfake and real images.
- Input images are resized to 150x150 pixels and normalized before being passed to the model.
- The model outputs a probability score indicating whether an image is likely to be real or fake.

## Gradio Interface
The application provides a user-friendly interface where you can:
- Upload an image.
- Specify the model path (default: `deepfake_model.h5`).
- View the analyzed image with the prediction result.

## Deployment
This app can be deployed to Hugging Face Spaces using Gradio:
1. Create a new space on Hugging Face.
2. Upload `app.py` and `deepfake_model.h5` to the repository.
3. Ensure that `requirements.txt` includes:
```
tensorflow
numpy
matplotlib
gradio
```
4. Set the runtime to Python and the entry point to `app.py`.
5. Deploy and test the application.

## Author
This project is developed as an open-source initiative to help detect deepfake images effectively.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
