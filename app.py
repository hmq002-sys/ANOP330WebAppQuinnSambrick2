import streamlit as st
import pandas as pd
import pickle

st.title("🎓 Reunion Attendance Predictor")

try:
    model = pickle.load(open("model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    features = pickle.load(open("features.pkl", "rb"))
    
    st.success("✅ Model loaded successfully!")
    
    reunion_years = st.number_input("Reunion Years Out", 0, 50, 5)
    class_year = st.number_input("Bucknell Class Year", 1900, 2024, 2000)
    peer = st.selectbox("Peer-to-Peer Contact", ["No", "Yes"])
    parent = st.selectbox("Current Parent", ["No", "Yes"])
    
    if st.button("Predict"):
        peer_num = 1 if peer == "Yes" else 0
        parent_num = 1 if parent == "Yes" else 0
        
        input_data = [[reunion_years, class_year, peer_num, parent_num]]
        input_df = pd.DataFrame(input_data, columns=features)
        
        scaled = scaler.transform(input_df)
        prob = model.predict_proba(scaled)[0][1]
        
        st.write(f"**Probability:** {prob:.1%}")
        if prob > 0.5:
            st.success("✅ Likely to attend")
        else:
            st.warning("❌ Unlikely to attend")
            
except Exception as e:
    st.error(f"Error: {str(e)}")
