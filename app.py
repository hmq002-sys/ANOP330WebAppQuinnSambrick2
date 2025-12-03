    import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Bucknell Reunion Attendance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #003366;
        text-align: center;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }
    .sub-header {
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #FF6B35;
        margin: 20px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background-color: #FF6B35;
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        with open('reunion_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('reunion_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('reunion_features.pkl', 'rb') as f:
            features = pickle.load(f)
        return model, scaler, features
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        st.info("Please run the training script first.")
        return None, None, None

def main():
    st.markdown('<h1 class="main-header">🎓 Bucknell Reunion Attendance Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Predict whether an alum will accept a reunion invitation</p>', unsafe_allow_html=True)
    
    model, scaler, features = load_model()
    
    if model is None:
        return
    
    with st.sidebar:
        st.header("⚙️ Settings")
        threshold = st.slider(
            "Prediction Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.05
        )
        st.divider()
        st.header("📋 Features Used")
        for feature in features:
            st.write(f"• {feature}")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Enter Alum Information")
        
        input_data = {}
        
        input_data['Reunion Years Out'] = st.number_input(
            "Reunion Years Out",
            min_value=0,
            max_value=50,
            value=5
        )
        
        input_data['Bucknell Class Year'] = st.number_input(
            "Bucknell Class Year",
            min_value=1900,
            max_value=2024,
            value=2000
        )
        
        input_data['Peer-to-Peer'] = st.selectbox(
            "Peer-to-Peer Contact",
            options=[0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )
        
        input_data['Parent - Current'] = st.selectbox(
            "Current Parent",
            options=[0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )
        
        if st.button("🔮 Predict Attendance", type="primary", use_container_width=True):
            input_df = pd.DataFrame([input_data])
            input_df = input_df[features]
            
            input_scaled = scaler.transform(input_df)
            
            probability = model.predict_proba(input_scaled)[0][1]
            prediction = 1 if probability >= threshold else 0
            
            with col2:
                st.subheader("📊 Prediction Result")
                
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                
                if prediction == 1:
                    st.success(f"## ✅ Likely to Accept")
                else:
                    st.warning(f"## ❌ Unlikely to Accept")
                
                st.metric("Probability", f"{probability:.1%}")
                st.progress(float(probability))
                st.caption(f"Threshold: {threshold:.0%}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.subheader("📋 Input Summary")
                for feature, value in input_data.items():
                    display_val = "Yes" if value == 1 else "No" if value == 0 else value
                    st.write(f"**{feature}:** {display_val}")

if __name__ == "__main__":
    main()
