# Sensor Fault Detection using Machine Learning

## Overview
This project predicts sensor faults and Remaining Useful Life (RUL) of aircraft engines using the NASA Turbofan Engine Degradation Dataset (CMAPSS). The system analyzes multiple sensor readings and identifies whether an engine is likely to fail within a defined cycle threshold.

The project includes:
- Data preprocessing
- RUL generation
- Fault classification
- Machine learning model training
- Hyperparameter tuning
- Streamlit web application deployment

---

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit

---

## Dataset
Dataset Used:
NASA CMAPSS FD001 Turbofan Engine Degradation Dataset

The dataset contains:
- Engine unit numbers
- Operational settings
- 21 sensor readings
- Time cycles

---

## Project Workflow

### 1. Data Preprocessing
- Loaded raw NASA sensor dataset
- Removed unnecessary columns
- Generated Remaining Useful Life (RUL)
- Created fault labels
- Normalized sensor values using MinMaxScaler

### 2. Model Development
Models explored:
- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)

Final Model:
- Random Forest Classifier

### 3. Hyperparameter Tuning
Used GridSearchCV for:
- n_estimators
- max_depth

### 4. Model Saving
Saved trained model and scaler using Joblib.

---

## Project Structure

sensor-fault-detection/
│
├── data/
│   └── train_FD001.txt
│
├── models/
│   ├── random_forest_model.pkl
│   └── scaler.pkl
│
├── src/
│   └── train_model.py
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/sensor-fault-detection.git
cd sensor-fault-detection
