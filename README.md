# Streamlit-Project

This project is a **data science application** built with **Streamlit**.  
The goal is to identify whether a bank customer should be classified as **"Risk"** or **"No Risk"** for a lending system, based on customer characteristics such as:

- Gender  
- Age  
- Credit History  
- Employment Duration  
- Loan Amount, Loan Purpose, and other features  

Several machine learning models were tested, and **CatBoostClassifier** was chosen for its strong performance.  
The model’s results are evaluated with **confusion matrix** and **classification metrics** (precision, recall, f1-score).

## Process
- Data preprocessing (duplicate removal, feature encoding)
- Exploratory visualizations:
  - Risk vs. Sex
  - Risk vs. Credit History
  - Risk vs. Employment Duration
- CatBoost classification model with configurable threshold
- Confusion matrix and classification report (precision/recall/f1)
- Automated tests with `pytest`
- Dockerized for easy deployment
