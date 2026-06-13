import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models
from PIL import Image
import numpy as np
import pandas as pd
import time
import os
import datetime
import random
from streamlit_js_eval import get_geolocation

# 1. PAGE CONFIG
st.set_page_config(page_title="Live Analysis", page_icon="🔮", layout="wide")

# 2. STYLING (Fixed Title Color & Layout)
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; }

    /* FIX: Target only the span class, NOT the whole H1. 
       This keeps the Emoji Purple and the Text Green/Blue */
    .gradient-text {
        background: -webkit-linear-gradient(45deg, #4CAF50, #2196F3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }

    h1 {
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0px;
    }

    /* Headers - Blue Outline Style */
    .stInfo {
        border-left: 5px solid #2196F3 !important;
    }

    /* Button Styling */
    div.stButton > button {
        background: linear-gradient(90deg, #00C853, #64DD17);
        color: white; border: none; border-radius: 8px; 
        font-size: 16px; font-weight: bold; width: 100%; padding: 12px;
        transition: all 0.3s ease;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.3);
    }
    div.stButton > button:hover {
        background: linear-gradient(90deg, #00E676, #76FF03);
        transform: translateY(-2px);
        box-shadow: 0px 6px 15px rgba(0, 230, 118, 0.4);
    }

    div[data-testid="stMetricValue"] { font-size: 28px; color: #4CAF50; }
    </style>
""", unsafe_allow_html=True)

# 3. SESSION STATE
if 'scan_complete' not in st.session_state:
    st.session_state['scan_complete'] = False
if 'result_data' not in st.session_state:
    st.session_state['result_data'] = {}


# 4. MODEL LOADER
@st.cache_resource
def get_model():
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'biomass_model.keras'))
    if not os.path.exists(model_path):
        return None
    try:
        tf.keras.backend.clear_session()
        # Load the trained model directly. Only rebuild the EfficientNet-B0
        # architecture (which downloads ImageNet weights) as a fallback if the
        # saved file turns out to hold weights rather than a full model.
        try:
            return tf.keras.models.load_model(model_path, compile=False)
        except Exception as load_err:
            st.sidebar.warning(f"Full-model load failed, rebuilding architecture: {load_err}")
            base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
            base_model.trainable = False
            model = models.Sequential([
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(1, activation='linear')
            ])
            model.load_weights(model_path)
            return model
    except Exception as e:
        st.sidebar.error(f"Model load error: {e}")
        return None


# SIDEBAR
st.sidebar.header("🛠 Scanner Settings")
input_mode = st.sidebar.radio("Input Source", ["📂 File Upload", "📷 Live Camera"])
# Removed the divider line here as requested

st.sidebar.subheader("📍 Location Services")
use_manual_gps = st.sidebar.checkbox("Override Auto-GPS")
if use_manual_gps:
    manual_lat = st.sidebar.number_input("Latitude", value=6.9271, format="%.4f")
    manual_lng = st.sidebar.number_input("Longitude", value=79.8612, format="%.4f")
    st.sidebar.info("Manual Mode Active")

model = get_model()
if model:
    st.sidebar.success("✅ Model: EfficientNet-B0")
else:
    st.sidebar.error("❌ Model: Offline")

# MAIN CONTENT - FIXED TITLE HTML
st.markdown('<h1>🔮 <span class="gradient-text">Pro-Grade Field Diagnostics</span></h1>', unsafe_allow_html=True)
st.caption("Real-time Biomass Quantification Engine | CSIRO Protocol | v3.2.0")

col1, col2 = st.columns([1, 1.5], gap="large")

# --- LEFT COLUMN ---
with col1:
    st.info("Step 1: Acquire Imagery")

    image = None
    uploaded_file = None
    camera_file = None
    if input_mode == "📂 File Upload":
        uploaded_file = st.file_uploader("Drop Field Sample Here", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
    else:
        camera_file = st.camera_input("Take Field Photo")
        if camera_file:
            image = Image.open(camera_file).convert("RGB")

    if image is not None:
        # Reset any stale result when the source image changes, so the displayed
        # numbers always correspond to the currently loaded image.
        src = uploaded_file or camera_file
        img_sig = getattr(src, "file_id", None) or getattr(src, "name", None) or id(image)
        if st.session_state.get("img_sig") != img_sig:
            st.session_state["img_sig"] = img_sig
            st.session_state["scan_complete"] = False
            st.session_state["result_data"] = {}

        st.image(image, caption="Source Imagery", use_container_width=True)

        # Spectral Analysis
        st.write("**Spectral Analysis**")
        # Resize small for instant load
        img_small = image.resize((150, 150))
        img_np = np.array(img_small)

        r_hist, _ = np.histogram(img_np[:, :, 0], bins=256, range=(0, 256))
        g_hist, _ = np.histogram(img_np[:, :, 1], bins=256, range=(0, 256))
        b_hist, _ = np.histogram(img_np[:, :, 2], bins=256, range=(0, 256))

        chart_data = pd.DataFrame({'Red': r_hist, 'Green': g_hist, 'Blue': b_hist})
        st.line_chart(chart_data, height=150, color=["#FF0000", "#00FF00", "#0000FF"])

# --- RIGHT COLUMN ---
with col2:
    st.info("Step 2: AI Inference")

    if image and model:
        # GHOSTING FIX: We use st.status instead of st.spinner.
        # st.status creates a box that stays in place, so the UI doesn't jump up and down.
        if st.button("Initialize Scan", type="primary"):
            status = st.status("Initializing Neural Engine...", expanded=True)
            try:
                # Preprocessing
                status.write("Processing Tensor Data...")
                img_resized = image.resize((224, 224))
                # NOTE: Keras EfficientNet-B0 includes its own preprocessing /
                # normalization layer and expects raw [0, 255] float inputs.
                # Do NOT divide by 255 here. If your training pipeline fed 0-1
                # scaled inputs instead, revert this line to `/ 255.0`.
                img_array = np.array(img_resized).astype(np.float32)
                img_array = np.expand_dims(img_array, axis=0)

                # Inference (direct call is faster than .predict() for one image)
                start_time = time.time()
                prediction = model(img_array, training=False).numpy()
                end_time = time.time()

                # Update Memory
                st.session_state['scan_complete'] = True
                st.session_state['result_data'] = {
                    'biomass': float(prediction[0][0]),
                    'latency': (end_time - start_time) * 1000,
                    'confidence': random.uniform(94.0, 99.9)
                }
                status.update(label="Scan Complete", state="complete", expanded=False)

            except Exception as e:
                status.update(label="Error", state="error")
                st.error(f"Inference Failed: {e}")

        # Display Results
        if st.session_state['scan_complete']:
            data = st.session_state['result_data']

            # Using a container for stability
            with st.container():
                r1, r2, r3 = st.columns(3)
                r1.metric("Biomass Density", f"{data['biomass']:.2f} g")
                r2.metric("Signal Quality (demo)", f"{data['confidence']:.1f}%")
                r3.metric("Latency", f"{data['latency']:.0f}ms")

                st.divider()

                if data['biomass'] < 20:
                    st.error("🔴 **CRITICAL: LOW YIELD**\n\nPasture depleted.")
                elif data['biomass'] < 60:
                    st.warning("🟡 **WARNING: MEDIUM YIELD**\n\nGrazing permitted with caution.")
                else:
                    st.success("🟢 **OPTIMAL: HIGH YIELD**\n\nIntensive grazing sustainable.")

                # --- GPS ---
                st.write("**📍 Geo-Tagging**")

                lat, lng = None, None

                if use_manual_gps:
                    lat, lng = manual_lat, manual_lng
                    st.info(f"Manual Lock: {lat}, {lng}")
                else:
                    loc = get_geolocation()
                    if loc:
                        lat = loc['coords']['latitude']
                        lng = loc['coords']['longitude']
                        st.success(f"Auto-GPS Locked: {lat:.4f}, {lng:.4f}")
                    else:
                        st.warning("GPS searching... (Allow Browser Permission)")

                if lat and lng:
                    map_data = pd.DataFrame({
                        'lat': [lat],
                        'lon': [lng],
                        'size': [20]
                    })
                    st.map(map_data, zoom=15, size='size')
                    loc_str = f"{lat}, {lng}"
                else:
                    loc_str = "No Signal"

                report_text = f"Scan Report\nDate: {datetime.datetime.now()}\nBiomass: {data['biomass']:.2f}g\nLocation: {loc_str}"
                st.download_button("💾 Export Analysis Log", report_text, file_name="scan_log.txt")