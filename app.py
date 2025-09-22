import streamlit as st
import pandas as pd
import joblib

@st.cache_resource
def load_pipeline():
    pipeline = joblib.load("xgb_pipeline.pkl")  # your saved pipeline
    return pipeline

pipeline = load_pipeline()
TAGS = ["Food Quality", "General", "Pricing", "Staff"]

st.set_page_config(page_title="Business Reviews Auto-Tagging", layout="wide")

st.title("📊 Business Reviews Auto-Tagging App")
st.write("Upload customer reviews in **any file format** and get insights with automatic tagging.")
st.markdown("""
#### 🏷️ Available Tags
- **Food Quality** 🍲  
- **Pricing** 💰  
- **Staff** 👩‍🍳  
- **General** 📌  
The model will automatically assign one or more of these tags to each customer review.
""")


option = st.sidebar.radio("Choose Mode", ["Single Review", "Bulk Reviews Dashboard"])

if option == "Single Review":
    st.subheader("🔹 Enter a Customer Review")
    review_text = st.text_area("Enter review text here...")

    if st.button("Predict Tags"):
        if review_text.strip() == "":
            st.warning("Please enter a review first!")
        else:
            prediction = pipeline.predict([review_text])
            predicted_tags = [tag for tag, val in zip(TAGS, prediction[0]) if val == 1]
            st.success(f"✅ Predicted Tags: {', '.join(predicted_tags) if predicted_tags else 'No tags predicted'}")

elif option == "Bulk Reviews Dashboard":
    st.subheader("📂 Upload File")
    st.write("Upload **CSV, Excel, JSON, or TXT** file containing reviews.")

    uploaded_file = st.file_uploader("Upload a file", type=["csv", "xlsx", "xls", "json", "txt"])

    if uploaded_file is not None:
        # Detect file type
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith(".json"):
            df = pd.read_json(uploaded_file)
        elif uploaded_file.name.endswith(".txt"):
            df = pd.read_csv(uploaded_file, sep="\n", header=None, names=["review_text"])
        else:
            st.error("Unsupported file format!")
            st.stop()

        st.write("✅ File uploaded successfully!")

        review_column = st.selectbox("Select the column that contains reviews", df.columns)

        if st.button("Analyze Reviews"):
            reviews = df[review_column].astype(str)

            predictions = pipeline.predict(reviews)
            predicted_tags = []
            for row in predictions:
                row_tags = [tag for tag, val in zip(TAGS, row) if val == 1]
                predicted_tags.append(", ".join(row_tags))
            df["Predicted Tags"] = predicted_tags

            st.subheader("📊 Review Insights Dashboard")
            col1, col2, col3, col4 = st.columns(4)

            tag_counts = pd.Series(sum([tags.split(", ") for tags in predicted_tags], [])).value_counts()

            col1.metric("Total Reviews", len(df))
            col2.metric("Food Quality Mentions", tag_counts.get("Food Quality", 0))
            col3.metric("Pricing Mentions", tag_counts.get("Pricing", 0))
            col4.metric("Staff Mentions", tag_counts.get("Staff", 0))

            st.bar_chart(tag_counts)

            st.subheader("🔎 Prediction Results")
            st.dataframe(df[[review_column, "Predicted Tags"]].head(20))

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇ Download Results CSV",
                data=csv,
                file_name="predicted_reviews.csv",
                mime="text/csv"
            )
