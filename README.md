# Diabetes-Prediction-Model

## Overview  
This project aims to predict the likelihood of diabetes in individuals using the K-Nearest Neighbors (KNN) algorithm. The model is trained on publicly available data and achieves an accuracy of 73%. It serves as a practical example of applying machine learning techniques to solve real-world health problems.  

## Features  
- **Data Preprocessing:** Cleaning and preparing the dataset for analysis.  
- **Exploratory Data Analysis (EDA):** Analyzing data patterns to identify trends and correlations.  
- **Model Building:** Implementing the KNN algorithm for classification.  
- **Hyperparameter Tuning:** Using GridSearchCV to optimize the model's performance.  

## Tools and Libraries  
- **Programming Language:** Python  
- **Libraries:**  
  - Pandas: For data manipulation  
  - NumPy: For numerical computations  
  - Matplotlib/Seaborn: For data visualization  
  - scikit-learn: For model building and evaluation  

## Dataset  
The dataset used in this project is sourced from National Institute of Diabetes and Digestive and Kidney Diseases. It contains information about medical diagnostic measurements and diabetes status of patients.  

## Usage
Run the diabetes_prediction.py script to preprocess the data, train the model, and evaluate its performance.
The predictions can be tested with new inputs or additional test data provided in a CSV format.

## Results
The model achieves a classification accuracy of 73% on the test dataset.
The performance is further improved by hyperparameter tuning with GridSearchCV.

## Future Improvements
Experimenting with other machine learning algorithms such as Support Vector Machines or Random Forest.
Increasing the dataset size to improve the model's generalization.
Deploying the model using Flask or Streamlit for real-world usage.
