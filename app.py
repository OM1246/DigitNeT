import streamlit as st
import numpy as np
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import os
import time
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json

MODEL_PATH = "mnist_cnn.h5"
HISTORY_FILE = "prediction_history.json"


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


def preprocess_image(image_data):
    """
    Takes the raw image data from the canvas, converts it to grayscale,
    resizes it to 28x28, and normalizes it.
    """
    img_rgba = Image.fromarray(image_data.astype("uint8"), "RGBA")
    img_gray = img_rgba.convert("L")
    img_inverted = Image.eval(img_gray, lambda x: 255 - x)
    img_resized = img_inverted.resize((28, 28), Image.LANCZOS)

    img_array = np.array(img_resized)
    img_reshaped = img_array.reshape(1, 28, 28, 1)
    img_normalized = img_reshaped.astype("float32") / 255.0
    return img_normalized, img_array


def save_prediction_history(digit, confidence, all_predictions):
    """Save prediction to history file"""
    history = load_prediction_history()
    history.append(
        {
            "timestamp": datetime.now().isoformat(),
            "predicted_digit": int(digit),
            "confidence": float(confidence),
            "all_predictions": [float(p) for p in all_predictions],
        }
    )

    # Keep only last 100 predictions
    if len(history) > 100:
        history = history[-100:]

    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f)


def load_prediction_history():
    """Load prediction history from file"""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        except:
            return []
    return []


def create_confidence_chart(predictions):
    """Create an interactive confidence chart"""
    digits = list(range(10))
    confidences = [float(p) * 100 for p in predictions[0]]

    fig = go.Figure(
        data=[
            go.Bar(
                x=digits,
                y=confidences,
                marker=dict(
                    color=confidences,
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Confidence %"),
                ),
                text=[f"{c:.1f}%" for c in confidences],
                textposition="outside",
            )
        ]
    )

    fig.update_layout(
        title="Confidence Distribution Across All Digits",
        xaxis_title="Digit",
        yaxis_title="Confidence (%)",
        template="plotly_dark",
        height=400,
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    return fig


def create_history_chart(history):
    """Create a chart showing prediction history"""
    if not history:
        return None

    df = pd.DataFrame(history)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Count predictions by digit
    digit_counts = df["predicted_digit"].value_counts().sort_index()

    fig = go.Figure(
        data=[
            go.Bar(
                x=digit_counts.index,
                y=digit_counts.values,
                marker=dict(
                    color=digit_counts.values,
                    colorscale="Plasma",
                    showscale=True,
                    colorbar=dict(title="Count"),
                ),
                text=digit_counts.values,
                textposition="outside",
            )
        ]
    )

    fig.update_layout(
        title="Your Drawing History (Last 100 Predictions)",
        xaxis_title="Digit",
        yaxis_title="Number of Times Drawn",
        template="plotly_dark",
        height=350,
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    return fig


def create_heatmap(image_array):
    """Create a heatmap visualization of the drawn digit"""
    fig = px.imshow(
        image_array,
        color_continuous_scale="Viridis",
        aspect="equal",
    )

    fig.update_layout(
        title="Pixel Intensity Heatmap",
        template="plotly_dark",
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=True,
    )

    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    return fig


# --- Page Configuration ---
st.set_page_config(
    page_title="DigitNet Pro | AI Digit Recognition",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS for High-Tech Dark Theme ---
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;500;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0a0e27 100%);
        color: #FFFFFF;
    }
    
    /* Animated background */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: 
            radial-gradient(circle at 20% 50%, rgba(0, 212, 255, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 80% 80%, rgba(124, 58, 237, 0.1) 0%, transparent 50%);
        animation: bgShift 15s ease infinite;
        z-index: -1;
        pointer-events: none;
    }
    
    @keyframes bgShift {
        0%, 100% { transform: translate(0, 0); }
        50% { transform: translate(30px, -30px); }
    }
    
    .main-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 3.5rem;
        font-weight: 900;
        text-align: center;
        padding: 1.5rem 0;
        background: linear-gradient(135deg, #00d4ff, #7c3aed, #f59e0b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: titleGlow 3s ease infinite;
        text-shadow: 0 0 30px rgba(0, 212, 255, 0.5);
    }
    
    @keyframes titleGlow {
        0%, 100% { filter: drop-shadow(0 0 10px rgba(0, 212, 255, 0.5)); }
        50% { filter: drop-shadow(0 0 20px rgba(0, 212, 255, 0.8)); }
    }
    
    .subtitle {
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.3rem;
        text-align: center;
        color: #94a3b8;
        padding-bottom: 2rem;
        letter-spacing: 2px;
    }
    
    .stCanvas {
        border-radius: 20px;
        border: 3px solid #00d4ff;
        box-shadow: 0 0 30px rgba(0, 212, 255, 0.4);
        transition: all 0.3s ease;
    }
    
    .stCanvas:hover {
        box-shadow: 0 0 40px rgba(0, 212, 255, 0.6);
        transform: scale(1.02);
    }
    
    .stButton>button {
        width: 100%;
        border-radius: 15px;
        border: 2px solid #00d4ff;
        color: #00d4ff;
        background: rgba(0, 212, 255, 0.1);
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.2rem;
        font-weight: 700;
        padding: 0.8rem 0;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    .stButton>button:hover {
        background: #00d4ff;
        color: #0a0e27;
        box-shadow: 0 0 30px rgba(0, 212, 255, 0.6);
        transform: translateY(-3px);
    }
    
    .stButton>button:active {
        transform: translateY(-1px);
    }
    
    .prediction-header {
        font-family: 'Orbitron', sans-serif;
        font-size: 1.8rem;
        font-weight: 700;
        text-align: center;
        margin-top: 2rem;
        color: #00d4ff;
        text-shadow: 0 0 15px rgba(0, 212, 255, 0.5);
    }
    
    .predicted-digit {
        font-family: 'Orbitron', sans-serif;
        font-size: 10rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(135deg, #00ffa3, #00d4ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1;
        animation: digitPulse 2s ease infinite;
        text-shadow: 0 0 50px rgba(0, 255, 163, 0.5);
    }
    
    @keyframes digitPulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    .confidence {
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.3rem;
        text-align: center;
        color: #94a3b8;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: rgba(0, 212, 255, 0.05);
        border: 2px solid rgba(0, 212, 255, 0.3);
        border-radius: 15px;
        padding: 1.5rem;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        border-color: #00d4ff;
        box-shadow: 0 0 25px rgba(0, 212, 255, 0.3);
        transform: translateY(-5px);
    }
    
    .stDataFrame {
        background: rgba(0, 212, 255, 0.05);
        border-radius: 10px;
        border: 1px solid rgba(0, 212, 255, 0.2);
    }
    
    /* Sidebar styling */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0e27 0%, #1a1f3a 100%);
        border-right: 2px solid rgba(0, 212, 255, 0.3);
    }
    
    .sidebar-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 1.5rem;
        color: #00d4ff;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 2px solid rgba(0, 212, 255, 0.3);
        margin-bottom: 1rem;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(0, 212, 255, 0.1);
        border-radius: 10px;
        border: 1px solid rgba(0, 212, 255, 0.3);
        font-family: 'Rajdhani', sans-serif;
        font-weight: 700;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #00d4ff, #7c3aed);
        border-radius: 10px;
        box-shadow: 0 0 10px rgba(0, 212, 255, 0.5);
    }
</style>
""",
    unsafe_allow_html=True,
)

# --- Sidebar ---
with st.sidebar:
    st.markdown('<p class="sidebar-title">⚙️ Settings</p>', unsafe_allow_html=True)

    stroke_width = st.slider("Brush Size", 10, 50, 25, 5)
    stroke_color = st.color_picker("Brush Color", "#000000")

    st.markdown("---")

    show_heatmap = st.checkbox("Show Pixel Heatmap", value=True)
    show_confidence_chart = st.checkbox("Show Confidence Chart", value=True)
    show_history = st.checkbox("Show Prediction History", value=True)

    st.markdown("---")

    st.markdown('<p class="sidebar-title">📊 Statistics</p>', unsafe_allow_html=True)

    history = load_prediction_history()
    total_predictions = len(history)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Predictions", total_predictions)
    with col2:
        if history:
            avg_confidence = np.mean([h["confidence"] for h in history])
            st.metric("Avg Confidence", f"{avg_confidence:.1f}%")

    if st.button("🗑️ Clear History", use_container_width=True):
        if os.path.exists(HISTORY_FILE):
            os.remove(HISTORY_FILE)
            st.success("History cleared!")
            st.rerun()

# --- Main App ---
model = load_model()

# Initialize session state
if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = "canvas_initial"
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

# Title and Subtitle
st.markdown('<p class="main-title">🎨 DigitNet Pro</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Advanced AI-Powered Digit Recognition System</p>',
    unsafe_allow_html=True,
)

# Main layout
col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown("### 🖊️ Drawing Canvas")
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color="#FFFFFF",
        height=400,
        width=400,
        drawing_mode="freedraw",
        key=st.session_state.canvas_key,
    )

    # Control Buttons
    btn_col1, btn_col2, btn_col3 = st.columns(3)
    predict_button = btn_col1.button("🔮 Predict", use_container_width=True)
    clear_button = btn_col2.button("🧹 Clear", use_container_width=True)
    random_button = btn_col3.button("🎲 Random", use_container_width=True)

with col_right:
    st.markdown("### 🎯 Prediction Results")

    if st.session_state.prediction_result is not None:
        result = st.session_state.prediction_result
        prediction = result["prediction"]
        display_image = result["display_image"]

        # Get top prediction
        top_digit = np.argmax(prediction[0])
        top_confidence = prediction[0][top_digit] * 100

        # Display the prediction
        st.markdown(
            f'<p class="predicted-digit">{top_digit}</p>', unsafe_allow_html=True
        )
        st.markdown(
            f'<p class="confidence">Confidence: {top_confidence:.2f}%</p>',
            unsafe_allow_html=True,
        )

        # Top 3 predictions
        top_indices = np.argsort(prediction[0])[-3:][::-1]
        top_confidences = prediction[0][top_indices] * 100
        top_digits = top_indices

        st.markdown("#### 🏆 Top 3 Predictions")
        df = pd.DataFrame(
            {
                "Rank": ["🥇", "🥈", "🥉"],
                "Digit": top_digits,
                "Confidence": [f"{c:.2f}%" for c in top_confidences],
            }
        )
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Model input visualization
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### 📥 Model Input (28x28)")
            st.image(display_image, width=200)

        with col_b:
            if show_heatmap:
                st.markdown("#### 🌡️ Pixel Intensity")
                heatmap_fig = create_heatmap(display_image)
                st.plotly_chart(heatmap_fig, use_container_width=True)
    else:
        st.info("👆 Draw a digit on the canvas and click Predict!")

# --- Button Logic ---
if clear_button:
    st.session_state.canvas_key = f"canvas_{int(time.time())}"
    st.session_state.prediction_result = None
    st.rerun()

if random_button:
    # Load a random MNIST sample
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    idx = np.random.randint(0, len(x_test))
    sample_image = x_test[idx]

    # Create a fake canvas result
    processed_image = sample_image.reshape(1, 28, 28, 1).astype("float32") / 255
    prediction = model.predict(processed_image)

    st.session_state.prediction_result = {
        "prediction": prediction,
        "display_image": sample_image,
    }
    st.rerun()

if predict_button:
    if canvas_result.image_data is not None and np.any(canvas_result.image_data):
        with st.spinner("🔄 Analyzing your drawing..."):
            processed_image, display_image = preprocess_image(canvas_result.image_data)
            prediction = model.predict(processed_image)

            # Save to history
            top_digit = np.argmax(prediction[0])
            top_confidence = prediction[0][top_digit] * 100
            save_prediction_history(top_digit, top_confidence, prediction[0])

            st.session_state.prediction_result = {
                "prediction": prediction,
                "display_image": display_image,
            }
        st.rerun()
    else:
        st.warning("⚠️ Please draw a digit before predicting!")

# --- Additional Visualizations ---
if st.session_state.prediction_result is not None:
    st.markdown("---")

    if show_confidence_chart:
        st.markdown("### 📊 Confidence Distribution")
        confidence_fig = create_confidence_chart(
            st.session_state.prediction_result["prediction"]
        )
        st.plotly_chart(confidence_fig, use_container_width=True)

if show_history and history:
    st.markdown("---")
    st.markdown("### 📈 Your Drawing History")
    history_fig = create_history_chart(history)
    if history_fig:
        st.plotly_chart(history_fig, use_container_width=True)

    # Recent predictions table
    with st.expander("📋 View Recent Predictions"):
        recent_history = history[-10:][::-1]
        df_history = pd.DataFrame(
            [
                {
                    "Time": datetime.fromisoformat(h["timestamp"]).strftime("%H:%M:%S"),
                    "Digit": h["predicted_digit"],
                    "Confidence": f"{h['confidence']:.2f}%",
                }
                for h in recent_history
            ]
        )
        st.dataframe(df_history, use_container_width=True, hide_index=True)

# --- Footer ---
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #94a3b8; font-family: Rajdhani;'>"
    "Powered by TensorFlow & Streamlit | Built with ❤️ for AI Enthusiasts"
    "</p>",
    unsafe_allow_html=True,
)
