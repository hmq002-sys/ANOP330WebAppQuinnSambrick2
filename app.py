import streamlit as st
import pandas as pd
import pickle

st.title("Reunion Predictor")

# Check if files exist
import os
st.write("Files available:", os.listdir())

# Try loading model
try:
    if os.path.exists("model.pkl"):
        model = pickle.load(open("model.pkl", "rb"))
        st.success("Model loaded!")
        st.write("Model type:", type(model))
    else:
        st.error("model.pkl not found")
except Exception as e:
    st.error(f"Load error: {e}")
