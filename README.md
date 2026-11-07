#  Smart Crop Prediction System

### Data-Driven Crop Selection for Farmers Using Classical Machine Learning

##  Project Overview

The **Smart Crop Prediction System** is a Machine Learning-based application that recommends suitable crops based on agricultural conditions such as soil type, season, irrigation source and weather parameters.

The system predicts the most suitable crop from 10 crop categories and displays the **Top 3 crop recommendations with confidence scores** through an interactive Streamlit application.

## Project Objectives

- Recommend suitable crops using agricultural data.
- Analyze soil and weather conditions.
- Compare different Machine Learning algorithms.
- Develop an interactive crop prediction application.

## Dataset Overview

| Description | Details |
|---|---|
| Total Records | 15,180 |
| Original Columns | 31 |
| Crop Categories | 10 |
| Selected Input Features | 8 |
| Problem Type | Multi-class Classification |

The dataset is simulated agricultural data modelled on Telangana soil-health data.

## Input Features

1. Season
2. Soil Type
3. Irrigation Source
4. Previous Crop
5. Soil pH
6. Rainfall
7. Average Temperature
8. Humidity

##  Project Workflow

The project follows the **CRISP-ML(Q) methodology**.

1. Business and Data Understanding
2. Data Cleaning and Preprocessing
3. Feature Engineering and Selection
4. Handling Class Imbalance
5. Model Building
6. Model Evaluation and Optimization
7. Model Deployment

##  Data Preprocessing

- Removed unnecessary columns.
- Handled missing values using Median and Mode Imputation.
- Corrected inconsistent values and spellings.
- Applied One-Hot Encoding and Label Encoding.
- Applied StandardScaler and PowerTransformer (Yeo-Johnson).
- Used feature selection techniques.
- Handled class imbalance using Class Weights, RandomOverSampler and ADASYN.

##  Machine Learning Algorithms

The following six algorithms were implemented and compared:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes
- Support Vector Classifier (SVC)
- Random Forest
- XGBoost

##  Model Evaluation

Models were evaluated using:

- Accuracy
- Precision
- Recall
- Macro F1-score
- Stratified 5-Fold Cross-Validation
- Bias-Variance Analysis

**Optuna** was used for hyperparameter optimization.

##  Final Model Performance

| Metric | Result |
|---|---:|
| Selected Model | Tuned Random Forest |
| Test Accuracy | 86.6% |
| Macro F1-score | 0.868 |

##  Application Features

- Interactive Streamlit interface.
- User input for agricultural parameters.
- Prediction across 10 crop categories.
- Top 3 crop recommendations.
- Confidence scores for predictions.

##  Technologies Used

| Category | Tools |
|---|---|
| Programming | Python |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn, XGBoost |
| Imbalanced Data | Imbalanced-learn |
| Hyperparameter Tuning | Optuna |
| Deployment | Streamlit |
| Model Saving | Pickle |

##  Project Structure

Smart-Crop-Prediction-System/
│
├── app.py
├── model.pkl
├── requirements.txt
├── README.md
│
├── dataset/
│   └── agricultural_data.csv
│
└── notebooks/
    └── crop_prediction.ipynb

*Note: Update the folder and file names above to match your actual GitHub repository.*

##  Future Enhancements

- Integrate real-time weather information.
- Include soil nutrients such as Nitrogen, Phosphorus and Potassium.
- Validate the model using real agricultural field data.
- Expand the dataset to more crops and geographical regions.
- Explore fertilizer and irrigation recommendations.

##  Disclaimer

This application provides crop recommendations based on its trained dataset. Actual crop suitability may vary depending on local soil, climate and farming conditions. Predictions should be treated as decision-support suggestions.

##  Author

**Sandeep Yadav**

GitHub: [sandeeppanga07](https://github.com/sandeeppanga07)


 If you find this project useful, consider giving the repository a star!
