from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

app = Flask(__name__)

# Load the trained Random Forest model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the encoders used during training (assuming they were saved)
with open('gp_name_encoder.pkl', 'rb') as encoder_file:
    gp_name_encoder = pickle.load(encoder_file)

with open('constructor_encoder.pkl', 'rb') as encoder_file:
    constructor_encoder = pickle.load(encoder_file)

with open('driver_encoder.pkl', 'rb') as encoder_file:
    driver_encoder = pickle.load(encoder_file)

def encode_gp_name(gp_name):
    return gp_name_encoder.transform([gp_name])[0]

def encode_constructor(constructor):
    return constructor_encoder.transform([constructor])[0]

def encode_driver(driver):
    return driver_encoder.transform([driver])[0]

def process_dob(dob):
    # Convert date of birth string to datetime object
    dob_date = datetime.strptime(dob, '%Y-%m-%d')
    # Extract the year and calculate the age
    age = datetime.now().year - dob_date.year
    return age

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Extract and process form data
        gp_name = request.form['GP_name']
        quali_pos = int(request.form['quali_pos'])
        constructor = request.form['constructor']
        driver = request.form['driver']
        driver_confidence = float(request.form['driver_confidence'])
        constructor_relaiblity = float(request.form['constructor_relaiblity'])
        
        # Encode categorical variables
        gp_name_encoded = encode_gp_name(gp_name)
        constructor_encoded = encode_constructor(constructor)
        driver_encoded = encode_driver(driver)
        
        # Combine all features into a single array
        features = np.array([[gp_name_encoded, quali_pos, constructor_encoded, driver_encoded,
                              driver_confidence, constructor_relaiblity]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        
        return render_template('result.html', prediction=prediction, probability=probability,
                               GP_name=gp_name, driver=driver, constructor=constructor,
                               quali_pos=quali_pos, driver_confidence=driver_confidence,
                               constructor_relaiblity=constructor_relaiblity)
    
    return render_template('predict.html')


@app.route('/metrics')
def metrics():
    # Assuming the metrics are calculated and saved during model training
    rf_precision = 0.9261270750359326
    rf_recall = 0.9191723663268379
    rf_f1 = 0.9221878505011034
    
    return render_template('metrics.html', precision=rf_precision, recall=rf_recall, f1_score=rf_f1)

if __name__ == "__main__":
    app.run(debug=True)
