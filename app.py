import streamlit as st
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation
from PIL import Image
import numpy as np
import h5py
import urllib.request

st.set_page_config(page_title="Egyptian Money Classifier", page_icon="🇪🇬")

# --- CONFIGURATION ---
# The File ID extracted from your provided link 
FILE_ID = '1L6dFo2LkNAFQm68WvY_B0Zlk22vGhPSg' 
MODEL_URL = f'https://drive.google.com/uc?export=download&id={FILE_ID}'
MODEL_PATH = 'egyptian_money_custom_cnn.h5'

# A function to build the EXACT same architecture you used in training
def build_my_cnn(num_classes=9):
    model = Sequential()
    # Block 1
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(128, 128, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    # Block 2
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))
    # Block 3
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))
    # Block 4
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))
    # Classification Head
    model.add(Flatten())
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model

@st.cache_resource
def load_trained_model():
    # --- DRIVE DOWNLOAD LOGIC ---
    if not os.path.exists(MODEL_PATH):
        try:
            with st.spinner('Downloading model from Google Drive (this happens once)...'):
                urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        except Exception as e:
            st.error(f"Error downloading from Drive: {e}")
            return None

    # Build the skeleton manually
    my_model = build_my_cnn(num_classes=9)
    
    try:
        # Load weights into the skeleton
        my_model.load_weights(MODEL_PATH)
        return my_model
    except Exception as e:
        # Fallback Force Load if headers mismatch
        st.warning("Attempting Legacy Force Load...")
        try:
            with h5py.File(MODEL_PATH, 'r') as f:
                my_model.load_weights(MODEL_PATH, by_name=True, skip_mismatch=True)
            return my_model
        except Exception as e2:
            st.error(f"Final loading failed: {e2}")
            return None

model = load_trained_model()

class_names = [
    '1 Pound', '10 Pounds (Old)', '10 Pounds (New 2023)', 
    '100 Pounds', '20 Pounds (Old)', '20 Pounds (New 2023)', 
    '200 Pounds', '5 Pounds', '50 Pounds'
]

# --- UI INTERFACE ---
st.title("🇪🇬 Egyptian Currency Classifier")
st.write("Upload a photo of Egyptian currency to identify it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # Pre-processing
    img = image.resize((128, 128))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if st.button('Classify'):
        if model is not None:
            with st.spinner('Analyzing...'):
                predictions = model.predict(img_array)
                result_index = np.argmax(predictions[0])
                result_class = class_names[result_index]
                confidence = predictions[0][result_index] * 100

                st.divider()
                st.success(f"Prediction: **{result_class}**")
                st.info(f"Confidence: **{confidence:.2f}%**")
        else:
            st.error("Model is not loaded. Please verify your Drive file permissions.")