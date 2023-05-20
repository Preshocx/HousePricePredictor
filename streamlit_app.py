import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the dataset
data = pd.read_csv('house_data.csv')

# Create the feature matrix X and the target variable y
X = data[['bedrooms', 'area']]
y = data['price']

# Train the linear regression model
model = LinearRegression()
model.fit(X, y)

def main():
    st.title("House Price Predictor")
    st.write("Welcome to the House Price Predictor!")
    st.write("Enter the number of bedrooms and the area of the house to get a price estimate.")

    # User input for bedrooms and area
    bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=1)
    area = st.number_input("Area (in square feet)", min_value=100, max_value=10000, value=1000)

    if st.button("Predict"):
        # Predict the price based on user input
        predicted_price = model.predict([[bedrooms, area]])
        st.success(f"The predicted price of the house is ${predicted_price[0]:,.2f}")

if __name__ == '__main__':
    main()
