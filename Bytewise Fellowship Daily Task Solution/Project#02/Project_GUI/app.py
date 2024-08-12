import ergast_py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, precision_score, f1_score, recall_score

# Initialize Ergast API client
e = ergast_py.Ergast()

# Function to fetch data from Ergast for a given year
def fetch_data(year):
    try:
        # Query Ergast API for the given year
        results = e.season(year).get_results()
        races = results.get('MRData', {}).get('RaceTable', {}).get('Races', [])
        
        # Check if races data is not empty
        if not races:
            raise ValueError(f"No races found for year {year}")
        
        # Collect race results
        data = []
        for race in races:
            for result in race.get('Results', []):
                result.update({
                    'raceName': race.get('raceName', ''),
                    'round': race.get('round', ''),
                    'date': race.get('date', ''),
                    'circuitId': race.get('Circuit', {}).get('circuitId', ''),
                    'location': race.get('Circuit', {}).get('Location', {}).get('locality', ''),
                    'country': race.get('Circuit', {}).get('Location', {}).get('country', '')
                })
                data.append(result)
        return data
    except Exception as e:
        print(f"Error fetching data for {year}: {e}")
        return []

# Fetch data from 2010 to 2023
def fetch_all_data():
    all_results = []
    for year in range(2010, 2024):
        data = fetch_data(year)
        if data:
            all_results.extend(data)
    return all_results

# Convert fetched data to DataFrame
def create_dataframe(results):
    df = pd.DataFrame(results)
    if not df.empty:
        # Drop unnecessary columns and rename columns
        df = df.drop(columns=['url', 'time', 'statusId'], errors='ignore')
        df.rename(columns={
            'raceName': 'GP_name',
            'position': 'position',
            'grid': 'quali_pos',
            'driverId': 'driver_id',
            'constructorId': 'constructor_id',
            'date': 'date',
            'circuitId': 'circuit_id',
            'location': 'location',
            'country': 'country'
        }, inplace=True)
        
        # Handle missing values
        df.fillna('', inplace=True)
        
        # Convert date columns to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        return df
    else:
        print("No data available to create DataFrame.")
        return pd.DataFrame()

# Fetch and prepare data
results = fetch_all_data()
data = create_dataframe(results)

# Check if DataFrame is not empty and contains expected columns
if not data.empty and {'driver_nationality', 'constructor_nationality', 'statusId'}.issubset(data.columns):
    # Adding driver home and constructor home indicators
    data['driver_home'] = (data['driver_nationality'] == data['country']).astype(int)
    data['constructor_home'] = (data['constructor_nationality'] == data['country']).astype(int)

    # Adding DNF indicators
    dnf_status_ids = [3, 4, 20, 29, 31, 41, 68, 73, 81, 97, 82, 104, 107, 130, 137]
    data['driver_dnf'] = data['statusId'].apply(lambda x: 1 if x in dnf_status_ids else 0)
    data['constructor_dnf'] = data['statusId'].apply(lambda x: 1 if x not in (dnf_status_ids + [1]) else 0)

    # Calculate DNF ratios for drivers
    dnf_by_driver = data.groupby('driver')['driver_dnf'].sum()
    driver_race_entered = data.groupby('driver')['driver_dnf'].count()
    driver_dnf_ratio = (dnf_by_driver * 100 / driver_race_entered).sort_values(ascending=False)
    driver_confidence = 1 - driver_dnf_ratio / 100
    driver_confidence_dict = dict(zip(driver_confidence.index, driver_confidence))

    # Plotting DNF ratios for drivers
    plt.figure(figsize=(30, 10))
    plt.bar(driver_dnf_ratio.index, driver_dnf_ratio, align='center', width=0.5)
    plt.xticks(rotation=90)
    plt.xlabel('Drivers')
    plt.ylabel('Driver DNF ratio')
    plt.title('DNFs ratio due to driver error')
    plt.show()

    # Calculate DNF ratios for constructors
    dnf_by_constructor = data.groupby('constructor')['constructor_dnf'].sum()
    constructor_race_entered = data.groupby('constructor')['constructor_dnf'].count()
    constructor_dnf_ratio = (dnf_by_constructor * 100 / constructor_race_entered).sort_values(ascending=False)
    constructor_reliability = 1 - constructor_dnf_ratio / 100
    constructor_reliability_dict = dict(zip(constructor_reliability.index, constructor_reliability))

    # Plotting DNF ratios for constructors
    plt.figure(figsize=(30, 10))
    plt.bar(constructor_dnf_ratio.index, constructor_dnf_ratio, align='center', width=0.8)
    plt.xticks(rotation=90)
    plt.xlabel('Constructors')
    plt.ylabel('Constructor DNF ratio')
    plt.title('DNFs ratio due to constructor error')
    plt.show()

    # Calculate driver home point finish ratio
    driver_home_points_finish = data.loc[(data['position'] < 11) & (data['driver_home'] == 1)].groupby('driver').count()['position']
    total_home_races = data[data['driver_home'] == 1].groupby('driver')['driver_home'].sum()
    driver_home_point_finish_ratio = (driver_home_points_finish * 100 / total_home_races).sort_values(ascending=False).fillna(0)

    # Plotting driver home point finish ratio
    plt.figure(figsize=(30, 10))
    plt.bar(driver_home_point_finish_ratio.index, driver_home_point_finish_ratio, align='center', width=0.8)
    plt.xticks(rotation=90)
    plt.xlabel('Driver')
    plt.ylabel('Percentage')
    plt.title('Drivers point finish percentage at home race')
    plt.show()

    # Calculate constructor home point finish ratio
    constructor_home_points_finish = data.loc[(data['position'] < 11) & (data['constructor_home'] == 1)].groupby('constructor').count()['position']
    total_home_races_constructor = data[data['constructor_home'] == 1].groupby('constructor')['constructor_home'].sum()
    constructor_home_point_finish_ratio = (constructor_home_points_finish * 100 / total_home_races_constructor).sort_values(ascending=False).fillna(0)

    # Plotting constructor home point finish ratio
    plt.figure(figsize=(30, 10))
    plt.bar(constructor_home_point_finish_ratio.index, constructor_home_point_finish_ratio, align='center', width=0.8)
    plt.xticks(rotation=90)
    plt.xlabel('Constructors')
    plt.ylabel('Percentage')
    plt.title('Constructor point finish percentage at home race')
    plt.show()

    # Prepare cleaned data
    cleaned_data = data[['GP_name', 'quali_pos', 'constructor', 'driver', 'position',
                         'driver_confidence', 'constructor_reliability', 'active_driver', 'active_constructor', 'dob']]
    cleaned_data = cleaned_data[(cleaned_data['active_driver'] == 1) & (cleaned_data['active_constructor'] == 1)]

    # Encode categorical variables
    encoder = LabelEncoder()
    cleaned_data['GP_name'] = encoder.fit_transform(cleaned_data['GP_name'])
    cleaned_data['constructor'] = encoder.fit_transform(cleaned_data['constructor'])
    cleaned_data['driver'] = encoder.fit_transform(cleaned_data['driver'])
    cleaned_data['dob'] = encoder.fit_transform(cleaned_data['dob'])

    # Split into features and target variable
    X = cleaned_data.drop('position', axis=1)
    y = cleaned_data['position']

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Evaluate different models
    models = {
        'Logistic Regression': LogisticRegression(),
        'Random Forest': RandomForestClassifier(),
        'Support Vector Classifier': SVC(),
        'Decision Tree': DecisionTreeClassifier(),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Naive Bayes': GaussianNB()
    }

    results = {}
    for name, model in models.items():
        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        cv_results = cross_val_score(model, X_scaled, y, cv=kfold, scoring='accuracy')
        results[name] = cv_results.mean()
    
    # Print model performance
    for model, score in results.items():
        print(f"{model}: {score:.2f}")

    # Evaluate best model (you can choose the model based on results)
    best_model = RandomForestClassifier()
    best_model.fit(X_scaled, y)
    y_pred = best_model.predict(X_scaled)

    # Print classification metrics
    print(f"Accuracy: {best_model.score(X_scaled, y):.2f}")
    print(f"Precision: {precision_score(y, y_pred, average='weighted'):.2f}")
    print(f"Recall: {recall_score(y, y_pred, average='weighted'):.2f}")
    print(f"F1 Score: {f1_score(y, y_pred, average='weighted'):.2f}")

else:
    print("Data is missing or does not contain expected columns.")
