# Manufacturing Predictive API

This project is a RESTful API designed to predict machine downtime in manufacturing operations. It uses a machine learning model to analyze operational parameters and predict whether a machine is likely to experience downtime. The API is built using Python and Flask, with scikit-learn for the predictive model.

---

## Setup Instructions

### Prerequisites
- Python 3.x
- pip (Python package installer)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/manufacturing_api.git
   ```
2. Navigate to the project folder:
   ```bash
   cd manufacturing_api
   ```
3. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application
Start the Flask application:
```bash
python app.py
```
The API will be available at [http://127.0.0.1:5000](http://127.0.0.1:5000).

---

## Usage Instructions

### 1. Upload Dataset
**URL:** `/upload`

**Method:** `POST`

**Description:** Upload a CSV file containing manufacturing data.

#### Steps:
- Use a tool like Postman or curl to send a POST request to [http://127.0.0.1:5000/upload](http://127.0.0.1:5000/upload).
- In the request body, include the dataset file as form-data with the key `file`.

#### Example Request:
```bash
curl -X POST -F "file=@data/Machine Downtime.csv" http://127.0.0.1:5000/upload
```

#### Example Response:
```json
{
  "message": "File uploaded successfully!",
  "rows": 2500
}
```

### 2. Train Model
**URL:** `/train`

**Method:** `POST`

**Description:** Train the machine learning model on the uploaded dataset.

#### Steps:
- Send a POST request to [http://127.0.0.1:5000/train](http://127.0.0.1:5000/train).

#### Example Request:
```bash
curl -X POST http://127.0.0.1:5000/train
```

#### Example Response:
```json
{
  "message": "Model trained successfully!",
  "accuracy": 0.95
}
```

### 3. Predict Downtime
**URL:** `/predict`

**Method:** `POST`

**Description:** Predict machine downtime based on input parameters.

#### Steps:
- Send a POST request to [http://127.0.0.1:5000/predict](http://127.0.0.1:5000/predict).
- Include the input parameters in JSON format in the request body.

#### Example Request:
```bash
curl -X POST -H "Content-Type: application/json" -d '{
  "Hydraulic_Pressure(bar)": 100,
  "Coolant_Pressure(bar)": 50,
  "Air_System_Pressure(bar)": 70,
  "Coolant_Temperature": 60,
  "Hydraulic_Oil_Temperature(?C)": 75,
  "Spindle_Bearing_Temperature(?C)": 85,
  "Spindle_Vibration(?m)": 0.5
}' http://127.0.0.1:5000/predict
```

#### Example Response:
```json
{
  "Downtime": "Yes",
  "Confidence": 0.85
}
```

---

## Folder Structure
```plaintext
manufacturing_api/
│
├── app.py                        # Main Flask application code
├── requirements.txt              # List of Python libraries required
├── README.md                     # Project documentation
│
├── data/                         # Folder for datasets
│   └── Machine Downtime.csv      # Dataset file
```

---

## Technologies Used
- **Python:** Programming language.
- **Flask:** Web framework for building the API.
- **Pandas:** Data manipulation and analysis.
- **Scikit-learn:** Machine learning library for training the model.
- **NumPy:** Numerical computations.

---

## License
This project is licensed under the MIT License. See the LICENSE file for details.
