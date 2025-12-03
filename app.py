import streamlit as st
import pandas as pd
import pickle

st.title("Simple Test")

try:
    # Try loading without sklearn
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    st.write("Model type:", type(model))
    
    # Test prediction
    import numpy as np
    test_input = [[5, 2000, 1, 0]]
    prob = model.predict_proba(test_input)[0][1]
    st.write(f"Test prediction: {prob:.1%}")
    
except Exception as e:
    st.write("Error:", str(e))
