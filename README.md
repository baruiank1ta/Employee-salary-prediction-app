💼**Employee Salary Prediction App**

A Machine Learning web application that predicts whether an individual's income is:<=50K or >50K, based on demographic and employment-related features.
This project demonstrates end-to-end ML workflow including data preprocessing, model comparison, evaluation, serialization, and deployment.

🚀 Project Overview

This project uses the Adult Income dataset to build and compare multiple classification models. The best-performing model is deployed as an interactive web application using Streamlit.

Users can input:

Age
Education Level
Occupation
Hours per week

and instantly receive a prediction with confidence score.

🧠 Machine Learning Workflow
1️⃣ Data Preprocessing

*Handled missing values

*Encoded categorical variables

*Label encoding for occupation

*Feature selection

*Train-test split

2️⃣ Models Compared

*Logistic Regression

*Random Forest

*Support Vector Machine (SVM)

*K-Nearest Neighbors (KNN)

*Gradient Boosting

*Neural Network (Keras + SciKeras)

🏆 Best Model

Gradient Boosting Classifier

Accuracy: 79.35%

Evaluated using Precision, Recall, F1-score

Balanced model performance across classes

📊 Model Evaluation

Due to class imbalance in the dataset:

Class 0 (<=50K): ~70%

Class 1 (>50K): ~30%

Evaluation included:

*Confusion Matrix

*Precision & Recall

*F1-score


🛠 Tech Stack

*Python

*Pandas

*NumPy

*Scikit-learn

*Gradient Boosting

*SciKeras

*TensorFlow (for Neural Network experiments)

*Streamlit

*Pickle

✨ Features

*Clean interactive UI

*Real-time prediction

*Confidence percentage display

*End-to-end ML pipeline

*Live deployed application

📌 Key Learnings

*Model comparison & evaluation

*Handling class imbalance

*Model serialization

*Building ML-powered web apps

*Deployment using Streamlit Cloud

*Structuring production-ready ML projects


🔗 Live Demo: [https://your-app-name.streamlit.app](https://salarypredict-yabsk32qcx8aheohzrvpjo.streamlit.app)
