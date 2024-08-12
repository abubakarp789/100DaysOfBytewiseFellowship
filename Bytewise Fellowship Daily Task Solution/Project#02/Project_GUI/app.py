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

# Load the datasets
results = pd.read_csv('results.csv')
races = pd.read_csv('races.csv')
quali = pd.read_csv('qualifying.csv')
drivers = pd.read_csv('drivers.csv')
constructors = pd.read_csv('constructors.csv')
circuit = pd.read_csv('circuits.csv')

# Drop unnecessary columns
circuit = circuit.drop(columns=['url'])
constructors = constructors.drop(columns=['url'])
drivers = drivers.drop(columns=['url'])
races = races.drop(columns=['url'])

# Merging the datasets
data1 = pd.merge(races, results, how='inner', on=['raceId'])
data2 = pd.merge(data1, quali, how='inner', on=['raceId', 'driverId', 'constructorId'])
data3 = pd.merge(data2, drivers, how='inner', on=['driverId'])
data4 = pd.merge(data3, constructors, how='inner', on=['constructorId'])
data5 = pd.merge(data4, circuit, how='inner', on=['circuitId'])

# Dropping more unnecessary columns
data = data5.drop([
    'round', 'circuitId', 'time_x', 'resultId', 'driverId', 'constructorId', 'number_x', 
    'positionText', 'position_x', 'positionOrder', 'laps', 'time_y', 'rank', 'fastestLapTime', 
    'fastestLapSpeed', 'qualifyId', 'driverRef', 'number', 'code', 'circuitRef', 'location', 
    'lat', 'lng', 'alt', 'number_y', 'points', 'constructorRef', 'name_x', 'raceId', 
    'fastestLap', 'q2', 'q3', 'milliseconds', 'q1'
], axis=1)

# Filter data for years >= 2010
data = data[data['year'] >= 2010]

# Rename columns
data.rename(columns={
    'name': 'GP_name', 'position_y': 'position', 'grid': 'quali_pos', 
    'name_y': 'constructor', 'nationality_x': 'driver_nationality', 
    'nationality_y': 'constructor_nationality'
}, inplace=True)

data['driver'] = data['forename'] + ' ' + data['surname']
data['date'] = pd.to_datetime(data['date'])
data['dob'] = pd.to_datetime(data['dob'])
data.drop(['forename', 'surname'], axis=1, inplace=True)

# Adding driver age parameter
data['age_at_gp_in_days'] = abs(data['dob'] - data['date'])
data['age_at_gp_in_days'] = data['age_at_gp_in_days'].apply(lambda x: int(str(x).split(' ')[0]))

# Truncate some string data to three characters
data['driver_nationality'] = data['driver_nationality'].apply(lambda x: str(x)[:3])
data['constructor_nationality'] = data['constructor_nationality'].apply(lambda x: str(x)[:3])
data['country'] = data['country'].apply(lambda x: 'Bri' if x == 'UK' else ('Ame' if x == 'USA' else ('Fre' if x == 'Fra' else str(x)[:3])))

# Adding home race indicators
data['driver_home'] = (data['driver_nationality'] == data['country']).astype(int)
data['constructor_home'] = (data['constructor_nationality'] == data['country']).astype(int)

# Adding DNF indicators
dnf_status_ids = [3, 4, 20, 29, 31, 41, 68, 73, 81, 97, 82, 104, 107, 130, 137]
data['driver_dnf'] = data['statusId'].apply(lambda x: 1 if x in dnf_status_ids else 0)
data['constructor_dnf'] = data['statusId'].apply(lambda x: 1 if x not in (dnf_status_ids + [1]) else 0)

# Calculate DNF ratios
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

# Add confidence and reliability to data
data['driver_confidence'] = data['driver'].apply(lambda x: driver_confidence_dict.get(x, 0))
data['constructor_reliability'] = data['constructor'].apply(lambda x: constructor_reliability_dict.get(x, 0))

# Filter active drivers and constructors
active_constructors = ['Renault', 'Williams', 'McLaren', 'Ferrari', 'Mercedes',
                       'AlphaTauri', 'Racing Point', 'Alfa Romeo', 'Red Bull', 'Haas F1 Team']
active_drivers = ['Daniel Ricciardo', 'Kevin Magnussen', 'Carlos Sainz',
                  'Valtteri Bottas', 'Lance Stroll', 'George Russell',
                  'Lando Norris', 'Sebastian Vettel', 'Kimi Räikkönen',
                  'Charles Leclerc', 'Lewis Hamilton', 'Daniil Kvyat',
                  'Max Verstappen', 'Pierre Gasly', 'Alexander Albon',
                  'Sergio Pérez', 'Esteban Ocon', 'Antonio Giovinazzi',
                  'Romain Grosjean', 'Nicholas Latifi']

data['active_driver'] = data['driver'].apply(lambda x: int(x in active_drivers))
data['active_constructor'] = data['constructor'].apply(lambda x: int(x in active_constructors))

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

for model_name, model in models.items():
    skf = StratifiedKFold(n_splits=10)
    cv_scores = cross_val_score(model, X_scaled, y, cv=skf, scoring='accuracy')
    print(f"{model_name} average accuracy: {cv_scores.mean()}")

# Use Logistic Regression as a sample model
logreg = LogisticRegression()
logreg.fit(X_scaled, y)
y_pred = logreg.predict(X_scaled)

# Print classification metrics
conf_matrix = confusion_matrix(y, y_pred)
precision = precision_score(y, y_pred, average='weighted')
recall = recall_score(y, y_pred, average='weighted')
f1 = f1_score(y, y_pred, average='weighted')

print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
