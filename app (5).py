import streamlit as st
import joblib
import pandas as pd

# Load the model
xgb_model = joblib.load("xgb_final_model.pkl")

st.title("🍽️ Review Classification App")
st.write("This app predicts tags for restaurant reviews using XGBoost 🎯")

# User inputs
review_text = st.text_area("Enter Review Text")
rating = st.number_input("Rating (1–5)", min_value=1, max_value=5, step=1)
like_count = st.number_input("Like Count", min_value=0, step=1)
author_review_count = st.number_input("Author Review Count", min_value=0, step=1)
author_local_guide_level = st.number_input("Local Guide Level", min_value=0, step=1)

if st.button("Predict"):
    # Create a dataframe with same structure as training features
    input_data = pd.DataFrame([{
        "review_text": review_text,
        "rating": rating,
        "like_count": like_count,
        "author_review_count": author_review_count,
        "author_local_guide_level": author_local_guide_level,
    }])

    # Prediction
    prediction = xgb_model.predict(input_data)
    st.success(f"✅ Predicted Tags: {prediction}")
