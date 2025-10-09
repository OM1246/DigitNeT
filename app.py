import streamlit as st
import numpy as np
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import os
import time
import pandas as pd


MODEL_PATH = "mnist_cnn.h5"


@st.cache_resource
def load_model():
    """
    Loads the pre-trained model from disk. If the model doesn't exist,
    it triggers the training process. This is cached so it only runs once.
    """
    if not os.path.exists(MODEL_PATH):
        with st.spinner(
            "Training model for the first time... This may take a few minutes."
        ):
            train_and_save_model()
        st.success("Model trained and saved!")

    return tf.keras.models.load_model(MODEL_PATH)


def train_and_save_model():
    """
    Trains a simple CNN on the MNIST dataset and saves it.
    This function is intended to be run only once.
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype("float32") / 255
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype("float32") / 255

    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(
                32, (3, 3), activation="relu", input_shape=(28, 28, 1)
            ),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    model.fit(
        x_train, y_train, epochs=10, batch_size=128, validation_data=(x_test, y_test)
    )
    model.save(MODEL_PATH)


# --- IMAGE PREPROCESSING (No changes from previous version) ---
def preprocess_image(image_data):
    """
    Takes the raw image data from the canvas, converts it to grayscale,
    resizes it to 28x28, and normalizes it.
    """
    img_rgba = Image.fromarray(image_data.astype("uint8"), "RGBA")
    # Invert colors: The model was trained on black background, white digits.
    # Canvas gives white background, black digits. So we invert the grayscale image.
    img_gray = img_rgba.convert("L")
    img_inverted = Image.eval(img_gray, lambda x: 255 - x)
    img_resized = img_inverted.resize((28, 28), Image.LANCZOS)

    img_array = np.array(img_resized)
    img_reshaped = img_array.reshape(1, 28, 28, 1)
    img_normalized = img_reshaped.astype("float32") / 255.0
    return img_normalized, img_array


# --- NEW MINIMALIST STREAMLIT UI ---

st.set_page_config(
    page_title="DigitNet | Clean UI",
    page_icon="🎨",
    layout="centered",  # Use 'centered' layout for a cleaner look
)

# --- Custom CSS for the new Dark Theme ---
st.markdown(
    """
<style>
    /* Main app background */
    .stApp {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    /* Main title */
    .title {
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        padding-top: 1rem;
        color: #FFFFFF;
    }
    /* Subtitle */
    .subtitle {
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 1.2rem;
        text-align: center;
        color: #AAAAAA;
        padding-bottom: 2rem;
    }
    /* Canvas styling */
    .stCanvas {
        border-radius: 15px;
        border: 2px solid #555555;
    }
    /* Button styling */
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        border: 2px solid #00A3FF;
        color: #00A3FF;
        background-color: transparent;
        font-size: 1.1rem;
        font-weight: 600;
        padding: 0.75rem 0;
        transition: all 0.3s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #00A3FF;
        color: #1E1E1E;
        border-color: #00A3FF;
    }
    .stButton>button:active {
        transform: scale(0.98);
    }
    /* Prediction output styling */
    .prediction-header {
        font-size: 1.5rem;
        font-weight: 600;
        text-align: center;
        margin-top: 2rem;
        color: #FFFFFF;
    }
    .predicted-digit {
        font-size: 8rem;
        font-weight: 800;
        text-align: center;
        color: #00FFA3;
        line-height: 1;
    }
    .confidence {
        font-size: 1.1rem;
        text-align: center;
        color: #AAAAAA;
        margin-bottom: 2rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# --- App Layout ---

# Load the trained model
model = load_model()

# Initialize session state variables
if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = "canvas_initial"
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

# Title and Subtitle
st.markdown('<p class="title">DigitNet 🎨</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Draw a single digit (0-9) on the canvas below.</p>',
    unsafe_allow_html=True,
)

# Center the canvas using columns
_, col_canvas, _ = st.columns([1, 4, 1])
with col_canvas:
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=25,
        stroke_color="#000000",
        background_color="#FFFFFF",
        height=350,
        width=350,
        drawing_mode="freedraw",
        key=st.session_state.canvas_key,
    )

# Control Buttons (Predict & Clear)
btn_col1, btn_col2 = st.columns(2)
predict_button = btn_col1.button("Predict")
clear_button = btn_col2.button("Clear")

# --- Button Logic ---
if clear_button:
    # Change the canvas key to force a re-render (clearing it)
    st.session_state.canvas_key = f"canvas_{int(time.time())}"
    # Clear the previous prediction
    st.session_state.prediction_result = None
    st.rerun()

if predict_button:
    if canvas_result.image_data is not None and np.any(canvas_result.image_data):
        # Preprocess the image and make a prediction
        processed_image, display_image = preprocess_image(canvas_result.image_data)
        prediction = model.predict(processed_image)

        # Store results in session state
        st.session_state.prediction_result = {
            "prediction": prediction,
            "display_image": display_image,
        }
    else:
        st.warning("Please draw a digit before predicting.", icon="⚠️")
        st.session_state.prediction_result = None


# --- Display Prediction Results ---
if st.session_state.prediction_result is not None:
    result = st.session_state.prediction_result
    prediction = result["prediction"]
    display_image = result["display_image"]

    # Get top 3 predictions
    top_indices = np.argsort(prediction[0])[-3:][::-1]
    top_confidences = prediction[0][top_indices] * 100
    top_digits = top_indices

    st.markdown(
        '<p class="prediction-header">Prediction Result</p>', unsafe_allow_html=True
    )

    # Display the top guess prominently
    st.markdown(
        f'<p class="predicted-digit">{top_digits[0]}</p>', unsafe_allow_html=True
    )
    st.markdown(
        f'<p class="confidence">Confidence: {top_confidences[0]:.2f}%</p>',
        unsafe_allow_html=True,
    )

    # Display the Top 3 predictions and the processed image side-by-side
    res_col1, res_col2 = st.columns([0.6, 0.4])
    with res_col1:
        st.write("Top 3 Guesses:")
        df = pd.DataFrame(
            {"Digit": top_digits, "Confidence": [f"{c:.2f}%" for c in top_confidences]}
        )
        st.dataframe(df, use_container_width=True, hide_index=True)

    with res_col2:
        st.write("Model Input:")
        st.image(display_image, caption="28x28 Grayscale", width=150)
