# Import libraries
from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the dataset (replace with the full path to your dataset)
dataset_path = r'D:\manufacturing_api\Machine Downtime.csv'  # Use raw string (r'') for Windows paths
df = pd.read_csv(dataset_path)

# Check for missing values and drop rows with missing values
print("Missing values in each column:")
print(df.isnull().sum())
df = df.dropna()

# Check if the dataset has a target column (e.g., 'Machine_Failure' or 'Downtime_Flag')
if 'Downtime' in df.columns:
    # Use the 'Downtime' column as the target
    y = df['Downtime']
else:
    # Create a synthetic target column (e.g., assume downtime if Spindle_Bearing_Temperature > 70°C)
    df['Downtime_Flag'] = df['Spindle_Bearing_Temperature(?C)'] > 70
    df['Downtime_Flag'] = df['Downtime_Flag'].astype(int)  # Convert True/False to 1/0
    y = df['Downtime_Flag']

# Check the distribution of the target column
print("Distribution of target column:")
print(y.value_counts())

# Features (X) and target (y)
X = df[['Hydraulic_Pressure(bar)', 'Coolant_Pressure(bar)', 'Air_System_Pressure(bar)',
        'Coolant_Temperature', 'Hydraulic_Oil_Temperature(?C)',
        'Spindle_Bearing_Temperature(?C)', 'Spindle_Vibration(?m)']]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Upload Endpoint: Accept a CSV file
@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    global df
    df = pd.read_csv(file)
    return jsonify({"message": "File uploaded successfully!", "rows": len(df)})

# Train Endpoint: Train the model
@app.route('/train', methods=['POST'])
def train():
    global model
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    return jsonify({"message": "Model trained successfully!", "accuracy": accuracy})

# Predict Endpoint: Make predictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = [data['Hydraulic_Pressure(bar)'], data['Coolant_Pressure(bar)'],
                data['Air_System_Pressure(bar)'], data['Coolant_Temperature'],
                data['Hydraulic_Oil_Temperature(?C)'],
                data['Spindle_Bearing_Temperature(?C)'], data['Spindle_Vibration(?m)']]
    prediction = model.predict([features])
    confidence = np.max(model.predict_proba([features]))
    return jsonify({"Downtime": "Yes" if prediction[0] == 1 else "No", "Confidence": float(confidence)})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)