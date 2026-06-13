# 🌿 CI-Kaggle-Application — Intelligent Pasture Biomass Estimation

A deep-learning **computer-vision** application that estimates **pasture (grass) biomass** from a single photograph, removing the need for destructive field sampling. The app is built with **Streamlit** and uses an **EfficientNet-B0** convolutional neural network (transfer learning on ImageNet) with a custom regression head.

> Coursework context: **CIS 6005 — Computational Intelligence**. The underlying regression model was trained on a Kaggle pasture-imagery dataset and exported to `biomass_model.keras`.

---

## ✨ Features

- **Live Prediction** — upload a field image or use the live camera, run inference, and get an estimated biomass density (grams) with a yield classification (Low / Medium / High).
- **Spectral Analysis** — per-image RGB histogram to inspect colour distribution (healthy green vs. dry/brown grass).
- **Geo-tagging** — automatic browser geolocation or manual latitude/longitude override, plotted on a map, with an exportable scan report.
- **Model Analytics** — architecture rationale, layer configuration, training loss curves, and residual error metrics (MAE, R²).
- **User Guide** — in-app documentation covering usage, interpretation, and troubleshooting.

---

## 🧠 Model Architecture

| Layer | Output Shape | Notes |
|-------|--------------|-------|
| EfficientNet-B0 (frozen) | (None, 7, 7, 1280) | ImageNet weights, feature extractor |
| GlobalAveragePooling2D | (None, 1280) | — |
| Dense (ReLU) | (None, 128) | Regression head |
| Dropout (0.3) | (None, 128) | Regularization |
| Dense (Linear) | (None, 1) | Biomass output (grams) |

Inputs are **224×224 RGB** tensors. Only the regression head is fine-tuned; the EfficientNet backbone is frozen. If `biomass_model.keras` is present it is loaded directly; otherwise the architecture above is rebuilt and weights are loaded.

---

## 📁 Project Structure

```
.
├── app.py                      # Home / landing page (system overview)
├── main.py                     # PyCharm starter script (not used by the app)
├── biomass_model.keras         # Trained Keras regression model
├── requirements.txt            # Python dependencies
├── pages/
│   ├── 1_🔮_Live_Prediction.py # Image upload/camera → inference → result + map
│   ├── 2_📊_Model_Analytics.py # Architecture, training curves, error analysis
│   └── 3_❓_User_Guide.py      # In-app documentation
└── *.jpg                       # Sample grass images for testing
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/CI-Kaggle-Application.git
cd CI-Kaggle-Application
```

### 2. Create a virtual environment & install dependencies

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

> Requires **Python 3.10+**. Key dependencies: `streamlit`, `tensorflow`, `keras`, `pillow`, `numpy`, `pandas`, `streamlit-js-eval`.

### 3. Run the app

```bash
streamlit run app.py
```

Streamlit opens the app in your browser (default `http://localhost:8501`). Use the sidebar to navigate between the Home, Live Prediction, Model Analytics, and User Guide pages.

---

## 🧪 Usage

1. Open **🔮 Live Prediction**.
2. Choose an input source — **File Upload** (drone/camera images) or **Live Camera**.
3. Review the **Spectral Analysis** histogram (a strong green peak indicates healthy grass).
4. Click **Initialize Scan** to run inference.
5. Interpret the result:
   - **< 20 g** → 🔴 Critical low yield (do not graze)
   - **20–60 g** → 🟡 Medium yield (graze with caution)
   - **> 60 g** → 🟢 High yield (intensive grazing sustainable)
6. Optionally geo-tag the scan and export the analysis log.

For best accuracy, ensure the image contains **only grass** — faces, sky, or vehicles will degrade predictions.

---

## ⚙️ Tech Stack

- **Framework:** Streamlit
- **Deep Learning:** TensorFlow / Keras (EfficientNet-B0)
- **Imaging & Data:** Pillow, NumPy, pandas
- **Geolocation:** streamlit-js-eval

---

## 📝 Notes

- The home page and parts of the analytics page use illustrative/demo figures (e.g., server logs, headline metrics) for presentation; the **Live Prediction** page performs real inference using the loaded Keras model.
- The trained model file and sample images are committed so the app runs out of the box. Adjust `.gitignore` if you prefer to exclude large binaries.

---

## 📄 License

No license has been specified. All rights reserved by the author unless a license file is added.
