 # 📄 AI Resume Screening System

An AI-powered web application that classifies resumes into different job categories using Machine Learning and Natural Language Processing (NLP). The system helps automate recruitment by predicting the most suitable job category based on resume content.

---

## 🚀 PROJECT OVERVIEW

This project uses a **Machine Learning pipeline** to analyze resume text and predict categories such as:
- Category:
-INFORMATION-TECHNOLOGY    120
-BUSINESS-DEVELOPMENT      120
-FINANCE                   118
-ADVOCATE                  118
-ACCOUNTANT                118
-ENGINEERING               118
-CHEF                      118
-AVIATION                  117
-FITNESS                   117
-SALES                     116
-BANKING                   115
-HEALTHCARE                115
-CONSULTANT                115
-CONSTRUCTION              112
-PUBLIC-RELATIONS          111
-HR                        110
-DESIGNER                  107
-ARTS                      103
-TEACHER                   102
-APPAREL                    97
-DIGITAL-MEDIA              96
-AGRICULTURE                63
-AUTOMOBILE                 36
-BPO                        22
Name: count, dtype: int64


The model is deployed using **Flask** with a simple web interface.

---

## 🧠 TECHNOLOGIES USED

- Python 🐍
- Flask 🌐
- Scikit-learn 🤖
- Pandas & NumPy 📊
- NLP (TF-IDF Vectorization)
- HTML & CSS 🎨

---

## 📊 DATASET

- Contains resumes labeled into multiple job categories
- Features:
  - `ID`
  - `Category` (Target)
  - `Feature` (Resume text)

---

## ⚙️ WORKFLOW

### 1. Data Preprocessing
- Removed null values
- Cleaned text (lowercasing, removing punctuation & numbers)

### 2. Feature Extraction
- Used **TF-IDF Vectorizer** to convert text into numerical features

### 3. Model Training
- Logistic Regression model trained on resume dataset

### 4. Evaluation
- Model evaluated using accuracy and classification report

### 5. Deployment
- Flask web application created
- HTML frontend for user input
- Model predicts category in real-time

---

## 📸 UI PREVIEW

👉 Users paste resume text into a web interface and receive predicted job category instantly.

---
## 📊 CONCLUSION

The AI Resume Screening System successfully demonstrates how Machine Learning and Natural Language Processing can be applied to automate the recruitment process. The model is able to analyze resume text and accurately classify it into relevant job categories such as Information Technology, Finance, Engineering, Healthcare, and others.

By using TF-IDF for feature extraction and Logistic Regression for classification, the system achieves reliable performance in predicting resume categories based on textual content. The web-based Flask interface further enhances usability by allowing users to interact with the model in real time.

Overall, the project proves that AI can significantly reduce manual effort in recruitment, improve efficiency, and support faster decision-making in Human Resource processes.


## 💡 RECOMMENDATIONS

This project serves as a strong foundation for an AI-powered recruitment system and can be extended into a full-scale HR tech solution with further improvements in model complexity, interface design, and deployment strategy.