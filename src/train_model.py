import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
cols = ['unit', 'time', 'op1', 'op2', 'op3'] + [f'sensor{i}' for i in range(1, 22)]
train = pd.read_csv("data/train_FD001.txt", sep=" ", header=None)
train = train.dropna(axis=1)
train.columns = cols

# Generate RUL
rul = train.groupby('unit')['time'].max().reset_index()
rul.columns = ['unit', 'max_time']

train = train.merge(rul, on='unit')
train['RUL'] = train['max_time'] - train['time']

# Create labels
train['label'] = np.where(train['RUL'] <= 30, 1, 0)

# Normalize sensor data
scaler = MinMaxScaler()
sensor_cols = [col for col in train.columns if 'sensor' in col]
train[sensor_cols] = scaler.fit_transform(train[sensor_cols])

# Save cleaned data
train.to_csv("cleaned_nasa_sensor_data.csv", index=False)


# Load dataset
df = pd.read_csv("cleaned_nasa_sensor_data.csv")

X = df[[col for col in df.columns if 'sensor' in col]]
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Random Forest Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))



from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [10, 20, None]
}

grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=3)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
print("Best Parameters:", grid.best_params_)


joblib.dump(model, "models/random_forest_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
print("Model saved successfully!")





