import streamlit as st

# 1. PAGE CONFIG
st.set_page_config(page_title="User Guide", page_icon="❓", layout="wide")

# 2. PRO CSS (Consistent Theme)
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; }
    h1 {
        background: -webkit-linear-gradient(45deg, #4CAF50, #2196F3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    h3 { color: #E0E0E0; border-left: 5px solid #2196F3; padding-left: 15px; }
    .guide-card {
        background-color: #161B22; padding: 20px; border-radius: 10px;
        border: 1px solid #30363D; margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# 3. CONTENT
st.title("❓ System Documentation & User Guide")
st.caption("CSIRO Biomass Estimation Protocol | v2.5")

st.markdown("---")

st.markdown("### 📘 Application Overview")
st.write("""
This application is designed for **Agronomists and Farmers** to estimate pasture biomass without destructive sampling. 
It uses a **Convolutional Neural Network (EfficientNet-B0)** to analyze the texture and density of grass from a simple photo.
""")

# SPLIT SECTIONS
tab1, tab2, tab3 = st.tabs(["🔮 Live Prediction", "📊 Analytics", "⚙️ Settings"])

with tab1:
    st.markdown('<div class="guide-card">', unsafe_allow_html=True)
    st.markdown("### How to use the Scanner")
    st.info("This is the main tool for field work.")

    st.markdown("""
    1.  **Select Input Source:**
        *   **File Upload:** Use for high-res drone or camera images saved on disk.
        *   **Live Camera:** Use for instant analysis in the field (requires webcam/phone permissions).
    2.  **Spectral Analysis:**
        *   The graph below the image shows the **RGB Distribution**.
        *   **Healthy Grass:** Should show a strong **Green** peak.
        *   **Dead/Dry Grass:** Will show balanced Red/Green (Brown) levels.
    3.  **Initialize Scan:**
        *   Click the **Green Button**. The AI processes the image in ~50ms.
    4.  **Interpret Results:**
        *   **< 20g:** Critical Low Yield (Do not graze).
        *   **20g - 60g:** Medium Yield (Caution).
        *   **> 60g:** High Yield (Optimal).
    """)
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="guide-card">', unsafe_allow_html=True)
    st.markdown("### Understanding the AI")
    st.warning("This section is for Technical Auditors and Data Scientists.")

    st.markdown("""
    *   **Architecture:** Explains the layers of the Neural Network.
    *   **Training Performance:** Shows the 'Loss Curve'.
        *   *If the Blue Line goes down:* The model is learning.
        *   *If lines diverge:* The model might be overfitting.
    *   **Residual Analysis:** Shows the error margin (approx ±22g).
    """)
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="guide-card">', unsafe_allow_html=True)
    st.markdown("### Troubleshooting")
    st.error("""
    **Common Errors:**
    *   **'Model Not Found':** Ensure `biomass_model.keras` is in the root folder.
    *   **Inaccurate Predictions:** Ensure the image contains **ONLY** grass. Faces, cars, or sky will confuse the AI.
    *   **Location Failed:** Ensure your device has Internet access for IP Geolocation.
    """)
    st.markdown('</div>', unsafe_allow_html=True)