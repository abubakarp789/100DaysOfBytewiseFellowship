import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import altair as alt
import matplotlib.pyplot as plt

# Title of the app
st.title('Predicting Product Sales Based on Historical Data')

# Upload CSV file
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the CSV file
    data = pd.read_csv(uploaded_file)
    
    # Handle missing values
    data = data.dropna()
    
    # Remove duplicates
    data = data.drop_duplicates()
    
    # Feature Engineering: Extract year and month from the 'Date' column
    data['Date'] = pd.to_datetime(data['Date'], format='%Y%m%d')
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    
    # Display the data
    st.write("Data Preview:")
    st.write(data.head())
    
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
    
    st.write('Mean Squared Error:', mse)
    st.write('R-squared:', r2)
    st.write('Model Accuracy:', accuracy_percentage, '%')
    
    # Visualize actual vs predicted sales
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred)
    ax.set_xlabel('Actual Sold Units')
    ax.set_ylabel('Predicted Sold Units')
    ax.set_title('Actual vs Predicted Sold Units')
    st.pyplot(fig)
    
    # Select columns for visualization
    st.write("Select columns for visualization")
    x_col = st.selectbox("Select X-axis column", data.columns)
    y_col = st.selectbox("Select Y-axis column", data.columns)
    
    # Create a line chart using Altair
    chart = alt.Chart(data).mark_line().encode(
        x=x_col,
        y=y_col
    ).properties(
        title="Product Sales Over Time"
    )
    
    # Display the chart
    st.altair_chart(chart, use_container_width=True)
    
    # Predict future sales (simple example using moving average)
    st.write("Predicting future sales using moving average")
    window_size = st.slider("Select window size for moving average", 1, 30, 7)
    data['Moving_Avg'] = data[y_col].rolling(window=window_size).mean()
    
    # Display the moving average chart
    chart_ma = alt.Chart(data).mark_line(color='red').encode(
        x=x_col,
        y='Moving_Avg'
    ).properties(
        title="Moving Average of Product Sales"
    )
    
    st.altair_chart(chart_ma, use_container_width=True)
    
    # Display the data with moving average
    st.write("Data with Moving Average:")
    st.write(data)
