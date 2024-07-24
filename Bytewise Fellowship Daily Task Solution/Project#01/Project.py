import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('Bytewise Fellowship Daily Task Solution\Project#01\Historical_Data.csv')

# Handle missing values
data = data.dropna()

# Remove duplicates
data = data.drop_duplicates()

# Feature Engineering: Extract year and month from the 'Date' column
data['Date'] = pd.to_datetime(data['Date'], format='%Y%m%d')
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month

# Define features and target variable
X = data[['Year', 'Month']]
y = data['Sold_Units']

# Create polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Build the Ridge regression model
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
accuracy_percentage = r2 * 100

print('Mean Squared Error:', mse)
print('R-squared:', r2)
print('Model Accuracy:', accuracy_percentage, '%')

# Visualize actual vs predicted sales
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Sold Units')
plt.ylabel('Predicted Sold Units')
plt.title('Actual vs Predicted Sold Units')
plt.show()
