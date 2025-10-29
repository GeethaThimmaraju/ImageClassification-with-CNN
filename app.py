import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# ✅ CIFAR-10 class labels
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# ✅ Load your trained model 
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('cnn_cifar10_model.h5')
    return model

model = load_model()

# ✅ Streamlit UI
st.title("CIFAR-10 Image Classification App")
st.write("Upload an image and let the CNN predict its class!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = np.array(image)
    img = cv2.resize(img, (32, 32))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    predictions = model.predict(img)
    class_idx = np.argmax(predictions)
    confidence = np.max(predictions) * 100

    st.write("### 🎯 Prediction:")
    st.success(f"**{class_names[class_idx]}** ({confidence:.2f}% confidence)")
    st.sidebar.header("📊 Model Info")
    st.sidebar.write("Model: Custom CNN trained on CIFAR-10")
    st.sidebar.write("Accuracy: ~85%")
