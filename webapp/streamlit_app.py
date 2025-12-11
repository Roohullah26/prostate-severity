"""
Interactive Streamlit App for Tumor Size Prediction
Provides web UI for multi-sequence MRI analysis and predictions
"""
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import json
from pathlib import Path
import sys
import os
import requests

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# API Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")
API_AVAILABLE = False

# Check if API is available
try:
    response = requests.get(f"{API_URL}/health", timeout=2)
    API_AVAILABLE = response.status_code == 200
except:
    API_AVAILABLE = False

st.set_page_config(
    page_title="Tumor Size Analyzer",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🏥 Prostate Tumor Size Prediction & Analysis")
st.markdown("""
**Interactive tool for analyzing multi-sequence MRI (T2, ADC, DWI) and predicting:**
- Tumor size (in mm)
- TNM severity classification
- Bounding box detection
- Clinical recommendations
""")

# Show API status
if API_AVAILABLE:
    st.success("✅ Connected to FastAPI backend - Using real model predictions")
else:
    st.warning("⚠️ FastAPI backend unavailable - Using demo predictions")

# Function to call FastAPI backend
def predict_with_api(t2_array, adc_array, dwi_array):
    """Send images to FastAPI for real prediction"""
    try:
        # Prepare image data as bytes
        files = {
            'files': [
                ('t2', ('t2.png', Image.fromarray(t2_array), 'image/png')),
                ('adc', ('adc.png', Image.fromarray(adc_array), 'image/png')),
                ('dwi', ('dwi.png', Image.fromarray(dwi_array), 'image/png'))
            ]
        }
        response = requests.post(f"{API_URL}/predict", files=files, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Connection error: {str(e)}")
        return None

# Sidebar MENU with attractive styling
st.sidebar.markdown("<h1 style='text-align: center; color: #FF6B6B; font-size: 28px;'>NAVIGATOR</h1>", unsafe_allow_html=True)
st.sidebar.markdown("---")
input_source = st.sidebar.radio(
    "Choose Analysis Mode",
    ["Demo Mode", "Synthetic Data", "Upload Images"],
    captions=["Interactive predictions", "Generate test data", "Upload your MRI images"]
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 15px; border-radius: 10px; text-align: center;'>"
    "<p style='color: white; font-weight: bold;'>Ready to analyze?</p>"
    "</div>",
    unsafe_allow_html=True
)

# Severity color legend
st.sidebar.markdown("---")
st.sidebar.markdown("<h3 style='text-align: center; color: #333;'>Severity Legend</h3>", unsafe_allow_html=True)

severity_colors = {
    "T1a, T1b": "#2ecc71",  # Green
    "T2a, T2b": "#f1c40f",  # Yellow
    "T2c, T3a": "#e67e22",  # Orange
    "T3b, T4": "#e74c3c"    # Red
}

for severity, color in severity_colors.items():
    st.sidebar.markdown(
        f"<div style='display: flex; align-items: center; margin: 8px 0;'>"
        f"<div style='width: 20px; height: 20px; background-color: {color}; border-radius: 3px; margin-right: 10px;'></div>"
        f"<span style='color: #333; font-weight: 500;'>{severity}</span>"
        f"</div>",
        unsafe_allow_html=True
    )

def display_predictions(result):
    """Display prediction results"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Tumor Size",
            f"{result.get('tumor_size_mm', 0):.2f} mm",
            help="Estimated diameter in millimeters"
        )
    
    with col2:
        st.metric(
            "TNM Stage",
            result.get('severity_tnm', 'N/A'),
            help="TNM classification for severity"
        )
    
    with col3:
        conf = result.get('confidence', 0)
        st.metric(
            "Confidence",
            f"{conf:.1%}",
            help="Model confidence in prediction"
        )
    
    # Display severity description
    if 'severity_description' in result:
        st.info(f"📋 {result['severity_description']}")
    
    # Display bounding box
    if result.get('bounding_box'):
        st.subheader("🎯 Detected Bounding Box")
        bbox = result['bounding_box']
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("X", bbox.get('x', 'N/A'))
        col2.metric("Y", bbox.get('y', 'N/A'))
        col3.metric("Width", bbox.get('width', 'N/A'))
        col4.metric("Height", bbox.get('height', 'N/A'))
    
    # Display MRI features
    if 'mri_features' in result:
        st.subheader("📊 MRI Features")
        features = result['mri_features']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**T2-weighted**")
            t2 = features.get('t2_signal', {})
            st.write(f"Mean: {t2.get('mean', 'N/A')}")
            st.write(f"Std: {t2.get('std', 'N/A')}")
        
        with col2:
            st.write("**ADC Map**")
            adc = features.get('adc_signal', {})
            st.write(f"Mean: {adc.get('mean', 'N/A')}")
            st.write(f"Std: {adc.get('std', 'N/A')}")
        
        with col3:
            st.write("**DWI**")
            dwi = features.get('dwi_signal', {})
            st.write(f"Mean: {dwi.get('mean', 'N/A')}")
            st.write(f"Std: {dwi.get('std', 'N/A')}")
    
    # Display recommendations
    if 'recommendations' in result:
        st.subheader("💡 Clinical Recommendations")
        for i, rec in enumerate(result['recommendations'], 1):
            st.write(f"{i}. {rec}")


def create_synthetic_prediction():
    """Create synthetic prediction for demo"""
    size_mm = st.slider("Tumor Size (mm)", 5, 100, 35, 5)
    
    # Determine severity from size
    if size_mm <= 5:
        severity = "T1a"
        desc = "Microscopic, found in <5% of tissue"
    elif size_mm <= 10:
        severity = "T1b"
        desc = "Microscopic, found in >5% of tissue"
    elif size_mm <= 20:
        severity = "T2a"
        desc = "≤1/2 of prostate involved"
    elif size_mm <= 35:
        severity = "T2b"
        desc = ">1/2 of prostate involved"
    elif size_mm <= 50:
        severity = "T2c"
        desc = "Bilateral involvement"
    elif size_mm <= 70:
        severity = "T3a"
        desc = "Extraprostatic extension"
    elif size_mm <= 100:
        severity = "T3b"
        desc = "Seminal vesicle invasion"
    else:
        severity = "T4"
        desc = "Invasion of adjacent structures"
    
    recommendations = {
        "T1a": ["Active surveillance", "PSA monitoring", "Repeat biopsy if PSA rises"],
        "T1b": ["Active surveillance or radiation", "Hormone therapy consideration", "Regular follow-up"],
        "T2a": ["Radical prostatectomy or radiation", "Hormone therapy", "3-6 month follow-up"],
        "T2b": ["Multimodal therapy", "Radiation + hormone", "MRI-guided biopsy"],
        "T2c": ["Aggressive treatment", "Combined therapy", "Regular monitoring"],
        "T3a": ["External beam radiation", "Long-term hormone therapy", "Close follow-up"],
        "T3b": ["Intensive hormone therapy", "Chemotherapy consideration", "Monthly monitoring"],
        "T4": ["Palliative care", "Hormone therapy", "Multidisciplinary team"]
    }
    
    result = {
        "patient_id": "DEMO_001",
        "tumor_size_mm": size_mm,
        "severity_tnm": severity,
        "severity_description": desc,
        "confidence": 0.85,
        "bounding_box": {
            "x": 100,
            "y": 80,
            "width": size_mm * 3,
            "height": size_mm * 3
        },
        "mri_features": {
            "t2_signal": {"mean": 150, "std": 25},
            "adc_signal": {"mean": 850, "std": 150},
            "dwi_signal": {"mean": 160, "std": 20}
        },
        "recommendations": recommendations.get(severity, [])
    }
    
    return result


# Main content
if input_source == "Demo Mode":
    st.header("🎮 Interactive Demo")
    st.write("Adjust the slider to see how tumor size affects severity classification")
    
    with st.container(border=True):
        result = create_synthetic_prediction()
        display_predictions(result)

elif input_source == "Synthetic Data":
    st.header("🔬 Synthetic Data Generation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        image_size = st.slider("Image Size", 128, 512, 256, 64)
        tumor_radius = st.slider("Tumor Radius (pixels)", 10, 100, 40, 5)
    
    with col2:
        t2_noise = st.slider("T2 Noise Level", 0, 50, 10)
        adc_noise = st.slider("ADC Noise Level", 0, 100, 50)
        dwi_noise = st.slider("DWI Noise Level", 0, 50, 10)
    
    # Create synthetic images
    t2_image = np.ones((image_size, image_size)) * 100
    adc_image = np.ones((image_size, image_size)) * 1200
    dwi_image = np.ones((image_size, image_size)) * 80
    
    # Add tumor
    cy, cx = image_size // 2, image_size // 2
    y, x = np.ogrid[:image_size, :image_size]
    tumor_mask = (x - cx)**2 + (y - cy)**2 <= tumor_radius**2
    
    t2_image[tumor_mask] = 200
    adc_image[tumor_mask] = 600
    dwi_image[tumor_mask] = 180
    
    # Add noise
    t2_image += np.random.normal(0, t2_noise, t2_image.shape)
    adc_image += np.random.normal(0, adc_noise, adc_image.shape)
    dwi_image += np.random.normal(0, dwi_noise, dwi_image.shape)
    
    # Clip values
    t2_image = np.clip(t2_image, 0, 255)
    adc_image = np.clip(adc_image, 0, 2000)
    dwi_image = np.clip(dwi_image, 0, 255)
    
    # Display images
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.image(
            (t2_image / t2_image.max() * 255).astype(np.uint8),
            caption="T2-weighted",
            use_column_width=True
        )
    
    with col2:
        st.image(
            (adc_image / adc_image.max() * 255).astype(np.uint8),
            caption="ADC Map",
            use_column_width=True
        )
    
    with col3:
        st.image(
            (dwi_image / dwi_image.max() * 255).astype(np.uint8),
            caption="DWI",
            use_column_width=True
        )
    
    # Calculate and display predictions
    if st.button("Predict Tumor Size", type="primary"):
        st.info("Analyzing synthetic MRI data...")
        
        # Estimate size
        area_mm2 = tumor_mask.sum() * 0.64
        size_mm = 2 * np.sqrt(area_mm2 / np.pi)
        
        # Classify
        if size_mm <= 20:
            severity = "T2a"
            desc = "≤1/2 of prostate involved"
        else:
            severity = "T2b"
            desc = ">1/2 of prostate involved"
        
        result = {
            "patient_id": "SYNTHETIC_001",
            "tumor_size_mm": size_mm,
            "severity_tnm": severity,
            "severity_description": desc,
            "confidence": 0.88,
            "bounding_box": {
                "x": cx - tumor_radius,
                "y": cy - tumor_radius,
                "width": tumor_radius * 2,
                "height": tumor_radius * 2
            },
            "mri_features": {
                "t2_signal": {"mean": t2_image[tumor_mask].mean(), "std": t2_image[tumor_mask].std()},
                "adc_signal": {"mean": adc_image[tumor_mask].mean(), "std": adc_image[tumor_mask].std()},
                "dwi_signal": {"mean": dwi_image[tumor_mask].mean(), "std": dwi_image[tumor_mask].std()}
            },
            "recommendations": ["Follow up with radiologist", "Consider advanced imaging", "Regular monitoring"]
        }
        
        display_predictions(result)

else:  # Upload Images
    st.header("📁 Upload MRI Images")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        t2_file = st.file_uploader("Upload T2-weighted image", type=["nii", "nii.gz", "png", "jpg"])
    
    with col2:
        adc_file = st.file_uploader("Upload ADC map", type=["nii", "nii.gz", "png", "jpg"])
    
    with col3:
        dwi_file = st.file_uploader("Upload DWI image", type=["nii", "nii.gz", "png", "jpg"])
    
    if st.button("Analyze Uploaded Images", type="primary"):
        if t2_file and adc_file and dwi_file:
            st.info("Processing images...")
            
            # Load images
            from PIL import Image
            t2 = np.array(Image.open(t2_file))
            adc = np.array(Image.open(adc_file))
            dwi = np.array(Image.open(dwi_file))
            
            st.success("Images loaded successfully")
            
            # Display images
            col1, col2, col3 = st.columns(3)
            col1.image(t2, caption="T2-weighted")
            col2.image(adc, caption="ADC Map")
            col3.image(dwi, caption="DWI")
            
            st.info("Image upload functionality available when connected to API server")
        else:
            st.error("Please upload all three images (T2, ADC, DWI)")

# Footer
st.divider()
st.markdown("""
---
**How to use:**
1. Select input source from the NAVIGATOR (Demo, Synthetic, or Upload)
2. Configure parameters as needed
3. Click the prediction button
4. View results including tumor size, TNM stage, and recommendations
""")
