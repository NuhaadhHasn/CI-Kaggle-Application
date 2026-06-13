import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Model Analytics", page_icon="📊", layout="wide")

st.title("📊 Architecture & Performance Metrics")
st.markdown("Detailed technical evaluation of the EfficientNet regression model.")

st.info("ℹ️ The training curves and error metrics on this page are illustrative "
        "figures from the development run, hard-coded for presentation purposes.")

# Tabbed Interface for cleaner look
tab1, tab2, tab3 = st.tabs(["🧠 Model Architecture", "📈 Training Performance", "🔬 Confusion Matrix"])

# TAB 1: ARCHITECTURE
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Why EfficientNet?")
        st.write("""
        We selected **EfficientNetB0** over ResNet50 or VGG16 for three reasons:
        1.  **Parameter Efficiency:** It uses only 5.3M parameters (vs 25M for ResNet), making it faster for deployment.
        2.  **Compound Scaling:** It balances depth, width, and resolution optimally.
        3.  **Transfer Learning:** Pre-trained weights from ImageNet allow the model to recognize textures (like grass blades) without needing millions of pasture images.
        """)
    with col2:
        st.info("**Layer Configuration:**")
        st.code("""
Layer (type)                Output Shape         Param #   
=========================================================
efficientnetb0 (Functional) (None, 7, 7, 1280)   4,049,571 
_________________________________________________________
global_avg_pool (GlobalAvg) (None, 1280)         0         
_________________________________________________________
dense_head (Dense)          (None, 128)          163,968   
_________________________________________________________
dropout (Dropout)           (None, 128)          0         
_________________________________________________________
output_layer (Dense)        (None, 1)            129       
=========================================================
Total params: 4,213,668
Trainable params: 164,097 (Fine-tuning Head)
        """, language="text")

# TAB 2: TRAINING GRAPHS
with tab2:
    st.subheader("Loss Convergence (MSE)")
    st.write("The graph below demonstrates the reduction in Mean Squared Error (MSE) over 15 epochs.")

    # Simulated Data (Matches your Kaggle Run)
    chart_data = pd.DataFrame({
        'Epoch': list(range(1, 16)),
        'Training Loss': [3591, 2814, 2186, 1711, 1557, 918, 884, 947, 885, 870, 860, 855, 850, 848, 845],
        'Validation Loss': [1729, 1720, 1127, 1094, 703, 607, 622, 603, 590, 588, 585, 582, 580, 578, 575]
    })

    st.line_chart(chart_data.set_index('Epoch'))

    st.caption(
        "Observation: The Validation Loss (Blue) stabilizes around Epoch 5, indicating the model is not overfitting significantly.")

# TAB 3: ERROR ANALYSIS
with tab3:
    st.subheader("Residual Analysis")
    st.write("In regression tasks, we analyze the 'Residuals' (Difference between True and Predicted values).")

    col_metrics1, col_metrics2 = st.columns(2)
    col_metrics1.metric("Mean Absolute Error (MAE)", "22.13 g", delta="-1.2 g")
    col_metrics2.metric("R-Squared (R²)", "0.89", delta="+0.05")

    st.warning(
        "Note: The model performs best in the 0-150g range. Extremely dense pastures (>300g) may have higher variance due to data scarcity.")