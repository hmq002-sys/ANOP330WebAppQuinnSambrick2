import streamlit as st
import pickle

st.title("🎓 Bucknell Reunion Predictor")
st.write("Enter alumni information below")

# Load without cache_resource decorator
try:
    features = pickle.load(open("features.pkl", "rb"))
    st.success("Model loaded")
except:
    features = ['Reunion Years Out', 'Bucknell Class Year', 'Peer-to-Peer', 'Parent - Current']
    st.info("Using default features")

# Simple inputs
years = st.slider("Reunion Years Out", 0, 50, 5)
class_year = st.slider("Class Year", 1900, 2024, 2000)
peer = st.checkbox("Peer Contact?")
parent = st.checkbox("Current Parent?")

if st.button("Predict"):
    # Simple calculation (no actual model needed for demo)
    prob = 0.3
    if peer:
        prob += 0.2
    if parent:
        prob += 0.1
    if years < 10:
        prob += 0.1
    if class_year > 2010:
        prob += 0.1
    
    prob = min(prob, 0.95)  # Cap at 95%
    
    st.write(f"**Probability:** {prob:.1%}")
    if prob > 0.5:
        st.success("✅ Likely to accept")
    else:
        st.warning("❌ Unlikely to accept")

st.caption("Demo Application")
