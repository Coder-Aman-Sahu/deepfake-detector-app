import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import os
import matplotlib.pyplot as plt
import gradio as gr

# Load your trained model (path will need to be configured)
def load_deepfake_model(model_path="deepfake_model.h5"):
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Function to preprocess and predict on a single image
def predict_image(image, model, threshold=0.5):
    try:
        # Preprocess the image
        img_array = img_to_array(image)
        img_array = img_array / 255.0  # Normalize like during training
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
        # Make prediction
        prediction = model.predict(img_array)
        
        # Get prediction score
        score = float(prediction[0][0])
        
        # Interpret prediction
        if score > threshold:
            result = "Fake"
            confidence = score
        else:
            result = "Real"
            confidence = 1 - score
            
        return result, confidence
    except Exception as e:
        return f"Error during prediction: {e}", 0

# Gradio interface function
def analyze_image(image, model_path):
    if not image:
        return None, "Please upload an image"
    
    try:
        # Load the model
        model = load_deepfake_model(model_path)
        if model is None:
            return image, f"Failed to load model from {model_path}. Check the path and try again."
        
        # Resize image to match model's expected input
        target_size = (150, 150)
        resized_img = image.resize(target_size)
        
        # Make prediction
        result, confidence = predict_image(resized_img, model)
        
        # Create labeled image
        fig, ax = plt.subplots(1)
        ax.imshow(image)
        ax.set_title(f"Prediction: {result} (Confidence: {confidence:.2%})")
        ax.axis('off')
        plt.close(fig)  # Close the figure to prevent display
        
        # Return results
        result_text = f"Prediction: {result}\nConfidence: {confidence:.2%}"
        return image, result_text
    except Exception as e:
        return image, f"Error: {str(e)}"

# Create the Gradio interface
def create_deepfake_detector_app():
    with gr.Blocks(title="Deepfake Image Detector") as app:
        gr.Markdown("# Deepfake Image Detector")
        gr.Markdown("Upload an image to check if it's real or a deepfake.")
        
        with gr.Row():
            with gr.Column():
                # Input components
                input_image = gr.Image(type="pil", label="Upload Image")
                model_path = gr.Textbox(
                    label="Model Path", 
                    value="deepfake_model.h5",
                    info="Path to your trained model file (.h5)"
                )
                analyze_btn = gr.Button("Analyze Image", variant="primary")
            
            with gr.Column():
                # Output components
                output_image = gr.Image(type="pil", label="Analyzed Image")
                result_text = gr.Textbox(label="Result")
        
        # Set up the button click event
        analyze_btn.click(
            fn=analyze_image,
            inputs=[input_image, model_path],
            outputs=[output_image, result_text]
        )
        
        gr.Markdown("### How to use")
        gr.Markdown("""
        1. Ensure you have installed the required libraries: `pip install tensorflow numpy matplotlib gradio`
        2. Upload an image using the panel on the left
        3. Provide the path to your trained model file (default is "deepfake_model.h5" in the current directory)
        4. Click "Analyze Image" to see the results
        """)
    
    return app

# Run the app
if __name__ == "__main__":
    app = create_deepfake_detector_app()
    app.launch(share=True)