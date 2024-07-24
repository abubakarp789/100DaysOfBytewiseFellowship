import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# (a). Load the Boston Housing dataset
boston_dataset = pd.read_csv('Day5/BostonHousing.csv')

# Remove rows with null values
boston_dataset = boston_dataset.dropna()

# Select the features and target variable
features = ['lstat', 'rm']
target = 'medv'

X_data = boston_dataset[features]
y_data = boston_dataset[target]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=5)


#(b). Train the linear regression model
linear_regression_model = LinearRegression()
linear_regression_model.fit(X_train, y_train)

#(c). Print the model's coefficients and intercept
print("Coefficients:", linear_regression_model.coef_)
print("Intercept:", linear_regression_model.intercept_)

#(d). Predict housing prices on the test set
y_predict = linear_regression_model.predict(X_test)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_predict)
print("Mean Squared Error:", mse)

#(e) Visualize the regression line and data points
sns.scatterplot(x=y_test, y=y_predict, color='blue', label='Actual Data points')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', label='Regression Line')
plt.xlabel('ISLAT')
plt.ylabel('MEDV')
plt.legend()
plt.show()


