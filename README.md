# airline-dataset-etl-prediction

# Airline Dataset ETL and Prediction

# Overview
This project analyzes an airline dataset, performs ETL (Extract, Transform, Load) operations, and applies machine learning models to predict flight statuses (On Time, Delayed, or Cancelled). The dataset is divided into two periods: the first half of the year is used for training, and the second half for testing.

# Features
- **Data Preprocessing**: Cleans and structures the dataset for analysis.
- **Exploratory Data Analysis (EDA)**: Generates statistical summaries and visualizations.
- **Feature Engineering**: Encodes categorical data and selects relevant features.
- **Machine Learning Models**: Compares various classification models for prediction.
- **Performance Evaluation**: Uses accuracy, precision, recall, and F1-score metrics.

# Installation
Ensure you have Python installed along with the required libraries:
```bash
pip install pandas matplotlib seaborn scikit-learn
```

# Dataset
- The dataset is an Excel file (`Airline Dataset Updated - v2.xlsx`).
- It contains information such as:
  - Passenger demographics (age, gender, nationality)
  - Flight details (airport name, departure date, flight status)
  - Airline-related information (pilot name, arrival airport, etc.)

# Data Preprocessing
1. Convert `Departure Date` to datetime format.
2. Encode `Flight Status` as:
   - `0` for On Time flights
   - `1` for Delayed and Cancelled flights
3. Extract first and second half-year data for training/testing.
4. Apply One-Hot Encoding to categorical features.

# Exploratory Data Analysis
- Age distribution of passengers
- Gender and nationality analysis
- Country-based passenger count
- Airport performance analysis (delays, on-time, cancellations)

# Model Training & Prediction
The following machine learning models are used:
- **Naive Bayes**
- **Logistic Regression**
- **K-Nearest Neighbors (KNN)**
- **Artificial Neural Networks (ANN)**
- **Decision Tree**

# Performance Metrics
Each model is evaluated using:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**
- **Confusion Matrix**

# Results
The best performing model is selected based on its ability to generalize to the second-half dataset. Performance is visualized using bar charts.

# Execution
Run the script using:
```bash
python airline_dataset_etl_prediction.py
```

## Future Improvements
- Incorporating more advanced ML models (e.g., Random Forest, XGBoost)
- Optimizing feature selection
- Experimenting with different sampling techniques for imbalanced classes

## Author
Sinem Demirci (Project Owner)
