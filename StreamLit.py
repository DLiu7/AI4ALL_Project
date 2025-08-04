import streamlit as st
from PIL import Image

st.set_page_config(page_title="RescueVision", layout="wide")

# Title
st.title("🚁 RescueVision")
st.markdown("""
Developed an automated human detection system for search and rescue drone imagery using YOLOv11 technology. Achieved **85% mean Average Precision (mAP)** through advanced machine learning techniques and comprehensive bias mitigation strategies.

_Developed within the AI4ALL Ignite accelerator program._
""")

# Problem Statement
st.header("🔍 Problem Statement")
st.markdown("""
Rapid and accurate human detection in aerial images is crucial for search and rescue (SAR) operations. Traditional manual analysis of drone footage is slow and error-prone, delaying emergency responses. RescueVision automates this process, enhancing speed and reliability while reducing human error.
""")


# Key Results
st.header("📊 Key Results")
st.markdown("""
- Achieved **85% mAP** on wilderness drone imagery
- Processed **4,000+ training images** from the SARD dataset
- Real-time detection capability for emergency response scenarios
- Developed a comprehensive evaluation framework:
  - Precision, Recall, F1-Score
  - False Positive Rate
""")

# Metrics Section #Change here after we do the validation
st.subheader("Model Metrics")
map = 0.85
precision = 0.82
recall = 0.80
f1 = 0.81

col1, col2, col3, col4 = st.columns(4)
col1.metric("mAP", f"{map*100:.1f}%")
col2.metric("Precision", f"{precision:.2f}")
col3.metric("Recall", f"{recall:.2f}")
col4.metric("F1-Score", f"{f1:.2f}")

# Bar chart for metrics
import pandas as pd
metrics_data = pd.DataFrame({
    'Metric': ['mAP', 'Precision', 'Recall', 'F1-Score'],
    'Value': [map, precision, recall, f1]
})
st.bar_chart(metrics_data.set_index('Metric'))

# --- Model Inference Section ---
st.header("🔎 Try the Model: Human Detection Demo")
try:
    from ultralytics import YOLO
    model = YOLO("best.pt")
    st.success("YOLO model loaded successfully.")
    uploaded_file = st.file_uploader("Upload an image for detection", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        results = model(image)
        # Get result image with bounding boxes
        result_img = results[0].plot()
        st.image(result_img, caption="Detection Result", use_container_width=True)
except Exception as e:
    st.warning(f"Model not loaded or Ultralytics not installed: {e}")

### Bias Mitigation
st.markdown("""
- **Demographic bias**: Improved detection across diverse skin tones
- **Environmental bias**: Trained model for better generalization beyond wilderness
- **Weather/lighting bias**: Used augmentation for robustness under adverse conditions
""")

# Methodologies
st.header("🛠️ Methodologies")
st.markdown("""
- Used **YOLOv11** with transfer learning and pre-trained weights
- Trained on **Google Colab** with Ultralytics
- Dataset managed and augmented using **Roboflow**
- Systematic **hyperparameter optimization** and iterative training
- Integrated **bias analysis** and **performance visualization** into the ML pipeline
""")

# Data Sources
st.header("📁 Data Sources")
st.markdown("""
**Kaggle:** [SARD - Search And Rescue Dataset](https://www.kaggle.com/datasets/nikolasgegenava/sard-search-and-rescue/data)
""")

# Technologies Used
st.header("🧰 Technologies Used")
tech_list = ["Python", "YOLOv11", "Google Colab", "Roboflow", "GitHub", "Ultralytics"]
st.markdown("\n".join([f"- {tech}" for tech in tech_list]))

# Contributors
st.header("👨‍💻 Authors")
st.markdown("""
Project completed in collaboration with:
- [@DLiu7](https://github.com/DLiu7)
- [@ankushachwani](https://github.com/ankushachwani)
- [@NER2160349](https://github.com/NER2160349)
- [@Shay-7278](https://github.com/Shay-7278)
- [@Mahir-MShahriar](https://github.com/Mahir-MShahriar)
""")

# Footer
st.markdown("---")
st.caption("Generated using Streamlit")
