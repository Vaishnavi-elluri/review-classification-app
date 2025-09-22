# 🏨 Business Review Auto tagging app

## 📌 Project Overview
Businesses receive hundreds of customer reviews every day, making it difficult to manually analyze feedback.  
This project automates the **classification of business reviews into multiple categories** (Food Quality, Staff, Pricing, General) using **Natural Language Processing (NLP)** and **Machine Learning**.

The solution reduces manual tagging delays, eliminates human error, and provides **faster insights** for business decision-making.

---

## 🎯 Objectives
- Automatically classify customer reviews into multiple categories.
- Handle imbalanced tag distribution to avoid bias toward frequent classes.
- Provide a simple **Streamlit dashboard** for single or bulk review analysis.
- Enable businesses to download predictions for further use.

---

## 🛠️ Tech Stack
- **Programming:** Python  
- **Libraries:** scikit-learn, XGBoost,pandas, numpy, matplotlib, seaborn, wordcloud  
- **NLP:** TF-IDF Vectorizer  
- **Deployment:** Streamlit, Hugging Face Spaces  

---

## 📊 Exploratory Data Analysis (EDA)
- **Tag Distribution:** Food Quality & Staff dominate, while Pricing & General are underrepresented.  
- **Review Lengths:** Most reviews are short (<100 words).  
- **Word Cloud:** Keywords like *food, service, taste, place* highlight key business factors.  

---

## 🤖 Model Training
1. **Data Preprocessing:** Tokenization, cleaning, TF-IDF feature extraction.  
2. **Modeling:** Multiple ML algorithms tested → **XGBoost selected** for best performance.  
3. **Handling Imbalance:** Applied class weighting and threshold tuning.  
4. **Evaluation Metrics:** Accuracy, Precision, Recall, F1-score (micro/macro averaged).  

---

## 🚀 Features of the App
- **Single Review Mode:** Enter one review and get predicted tags instantly.  
- **Bulk Review Mode:** Upload a CSV/Excel file with reviews → app returns tagged results + summary charts.  
- **Download Option:** Export predictions for offline use.  
- **Dashboard Style:** Tag distribution and insights shown inside the app.  

---

## 📈 Results
- Reduced manual tagging effort by **70%**.  
- Achieved strong performance with **XGBoost + TF-IDF pipeline**.
- Acieved  Accuracy Score as **91.5%** and F1 score  **0.9680**
- Successfully deployed as an interactive **Streamlit app**.

## 🔮 Future Improvements
- Use **transformer-based models (BERT, RoBERTa)** for better contextual understanding.  
- Improve imbalance handling with advanced techniques (SMOTE for multilabel, threshold tuning).  
- Expand categories beyond current 4 tags.

  ## 👩‍💻 Author
Developed by **Vaishnavi Elluri**  
*Data Science & AI Enthusiast | Machine Learning | NLP | Streamlit Deployment*
