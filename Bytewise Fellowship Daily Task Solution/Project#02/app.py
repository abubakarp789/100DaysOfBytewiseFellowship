from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, f1_score, recall_score

# Initialize the Flask application
app = Flask(__name__)

# Load the cleaned data
cleaned_data = pd.read_csv('cleaned_data.csv')

# Data preprocessing
le = LabelEncoder()
cleaned_data['GP_name'] = le.fit_transform(cleaned_data['GP_name'])
cleaned_data['constructor'] = le.fit_transform(cleaned_data['constructor'])
cleaned_data['driver'] = le.fit_transform(cleaned_data['driver'])

X = cleaned_data.drop(['position', 'active_driver', 'active_constructor', 'dob'], axis=1)
y = cleaned_data['position'].apply(lambda x: 1 if x < 4 else 3 if x > 10 else 2)

# Initialize the model
rf = RandomForestClassifier(n_estimators=1600, min_samples_split=20, min_samples_leaf=1, max_features='sqrt', max_depth=90, bootstrap=True)

# Define the main route
@app.route('/')
def index():
    return render_template('index.html')

# Route for model evaluation
@app.route('/evaluate', methods=['POST'])
def evaluate():
    kf = StratifiedKFold(n_splits=10, random_state=None, shuffle=False)
    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    
    # Evaluate the model
    cnf_mat_rf = confusion_matrix(y_test, y_pred_rf)
    rf_precision = precision_score(y_test, y_pred_rf, average='macro')
    rf_f1 = f1_score(y_test, y_pred_rf, average='macro')
    rf_recall = recall_score(y_test, y_pred_rf, average='macro')

    # Render the results on a new page
    return render_template('evaluation.html', precision=rf_precision, f1=rf_f1, recall=rf_recall)

# Run the application
if __name__ == '__main__':
    app.run(debug=True)
