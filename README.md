# 📄  AI RESUME SCREENING SYSTEM

# 📌 PROJECT OVERVIEW

The AI Resume Screening System is a machine learning-based web application designed to automate the process of shortlisting job applicants based on their resumes. The system analyzes uploaded resumes or text input and predicts the most suitable job category using Natural Language Processing (NLP) and machine learning techniques.

This project reduces manual recruitment effort, improves efficiency, and ensures fair and consistent candidate evaluation.

# ❗PROBLEM STATEMENT

Recruiters often face challenges when manually screening hundreds or thousands of resumes, which leads to:

⏳ Time-consuming hiring processes
⚖️ Human bias in candidate selection
📄 Difficulty in managing large volumes of applications
❌ Risk of overlooking qualified candidates

There is a need for an automated system that can efficiently analyze resumes and classify them into relevant job categories.

# 🎯OBJECTIVES 

The main objectives of this project are:

1. To develop an AI model that classifies resumes into job categories
2. To preprocess and analyze resume text data using NLP techniques
3. To automate resume screening and reduce recruiter workload
4. To improve accuracy in candidate-job matching
5. To deploy the model as a web application for real-time predictions


# 📂 DATA DESCRIPTION

The dataset used in this project consists of resumes collected and categorized into different job roles.

Key Features:
📌 Resume text content
📌 Job category labels
📌 Preprocessed text data (cleaned and tokenized)

Data Processing Steps:
Removal of special characters and stopwords
Tokenization of text
Vectorization using TF-IDF
Label encoding of job categories


# 📊 EXPLORATORY DATA ANALYSIS (EDA)

During EDA, the following analyses were performed:

📈 Distribution of job categories
☁️ Word frequency analysis using word clouds
📊 Identification of most common skills and keywords
📉 Analysis of dataset imbalance across categories

## Key Insights:
Some job categories had significantly more resumes than others
Technical roles contained more domain-specific keywords
Cleaned text significantly improved model performance


# 🧠 CONCLUSIONS
The machine learning model successfully classifies resumes into relevant job categories
NLP preprocessing significantly improves prediction accuracy
TF-IDF vectorization is effective for resume text representation
Automation reduces manual screening effort and improves efficiency


# 💡 RECOMMENDATIONS
1.  Expand dataset to include more diverse resumes.
2.  Use advanced deep learning models (e.g., BERT) for better accuracy
3.  Improve class balancing to handle uneven datasets.
4.  Add features like skill extraction and job recommendation.
5.  Enhance UI/UX for better user experience.
6.  FUTURE IMPROVEMENTS.
7.  User authentication system for recruiters.
8.  Dashboard for analytics and insights.
9.  PDF resume upload support.
10. Cloud deployment for global access.