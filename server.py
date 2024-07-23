from flask import Flask, request, jsonify, render_template, send_from_directory
import pickle
import numpy as np
import json
import os

app = Flask(__name__, template_folder='templates')

# Print the current working directory for debugging
print("Current Working Directory:", os.getcwd())

# Define the absolute paths for position prediction
POSITION_MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'artifacts', 'NBA_Position_classification.pickle'))
POSITION_SCALER_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'artifacts', 'pos_scaler.pickle'))
POSITION_COLUMNS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'artifacts', 'pos_columns.json'))
POSITION_ENCODER_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'artifacts', 'pos_label_encoder.pickle'))

# Define the absolute paths for salary prediction
SALARY_MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'artifacts', 'NBA_player_salary_model.pickle'))
SALARY_SCALER_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'artifacts', 'scaler.pickle'))
SALARY_COLUMNS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'artifacts', 'columns.json'))

# Print paths to verify
print("POSITION_MODEL_PATH:", POSITION_MODEL_PATH)
print("POSITION_SCALER_PATH:", POSITION_SCALER_PATH)
print("POSITION_COLUMNS_PATH:", POSITION_COLUMNS_PATH)
print("POSITION_ENCODER_PATH:", POSITION_ENCODER_PATH)

print("SALARY_MODEL_PATH:", SALARY_MODEL_PATH)
print("SALARY_SCALER_PATH:", SALARY_SCALER_PATH)
print("SALARY_COLUMNS_PATH:", SALARY_COLUMNS_PATH)

# Load the models, scalers, columns, and encoder
with open(POSITION_MODEL_PATH, 'rb') as f:
    position_model = pickle.load(f)

with open(POSITION_SCALER_PATH, 'rb') as f:
    position_scaler = pickle.load(f)

with open(POSITION_COLUMNS_PATH, 'r') as f:
    position_data_columns = json.load(f)['data_columns']
    
with open(POSITION_ENCODER_PATH, 'rb') as f:
    position_le = pickle.load(f)

with open(SALARY_MODEL_PATH, 'rb') as f:
    salary_model = pickle.load(f)

with open(SALARY_SCALER_PATH, 'rb') as f:
    salary_scaler = pickle.load(f)

with open(SALARY_COLUMNS_PATH, 'r') as f:
    salary_data_columns = json.load(f)['data_columns']

print("Position Data Columns:", position_data_columns)  # Debug: Print data columns
print("Salary Data Columns:", salary_data_columns)  # Debug: Print data columns

@app.route('/')
def home():
    return send_from_directory(os.getcwd(),'index.html')

@app.route('/predict_position', methods=['POST'])
def predict_position():
    data = request.form
    print("Received data for position prediction:", data)  # Debug: Log the received form data

    input_data = np.zeros(len(position_data_columns))

    for key in position_data_columns:
        input_data[position_data_columns.index(key)] = float(data.get(key, 0))
    
    print("Feature array before scaling (position):", input_data)  # Debug: Log the feature array before scaling

    # Scale the input features using the fitted scaler
    input_data_scaled = position_scaler.transform([input_data])[0]

    print("Feature array after scaling (position):", input_data_scaled)  # Debug: Log the feature array after scaling

    prediction = position_model.predict([input_data_scaled])[0]
    predicted_position = position_le.inverse_transform([prediction])[0]
    print("Prediction (position):", predicted_position)  # Debug: Log the prediction
    return jsonify({'position': predicted_position})

@app.route('/predict_salary', methods=['POST'])
def predict_salary():
    data = request.form
    print("Received data for salary prediction:", data)  # Debug: Log the received form data

    position = data['position']
    print("Position:", position)  # Debug: Print position
    
    Yrs = float(data['Yrs'])
    Age = float(data['Age'])
    GP = float(data['GP'])
    MIN = float(data['MIN'])
    FT_PCT = float(data['FT_PCT'])
    REB = float(data['REB'])
    AST = float(data['AST'])
    TOV = float(data['TOV'])
    STL = float(data['STL'])
    BLK = float(data['BLK'])
    PTS = float(data['PTS'])
    Efficiency = float(data['Efficiency'])
    PIE_scaled = float(data['PIE_scaled'])
    
    # Position one-hot encoding
    C = int(position == 'C')
    PF = int(position == 'PF')
    PG = int(position == 'PG')
    SF = int(position == 'SF')
    SG = int(position == 'SG')

    x = np.zeros(len(salary_data_columns))
    
    x[0] = Yrs
    x[1] = Age
    x[2] = GP
    x[3] = MIN
    x[4] = FT_PCT
    x[5] = REB
    x[6] = AST
    x[7] = TOV
    x[8] = STL
    x[9] = BLK
    x[10] = PTS
    x[11] = Efficiency
    x[12] = PIE_scaled
    x[13] = C
    x[14] = PF
    x[15] = PG
    x[16] = SF
    x[17] = SG

    print("Feature array before scaling (salary):", x)  # Debug: Log the feature array before scaling

    # Scale the input features using the fitted scaler
    x_scaled = salary_scaler.transform([x])[0]

    print("Feature array after scaling (salary):", x_scaled)  # Debug: Log the feature array after scaling

    prediction = salary_model.predict([x_scaled])[0]
    prediction = int(prediction)  # Convert to standard Python integer
    print("Prediction (salary):", prediction)  # Debug: Log the prediction
    return jsonify({'salary': prediction})

if __name__ == '__main__':
    app.run(debug=True)
