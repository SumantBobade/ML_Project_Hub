import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X=[[151],[132],[163],[190],[145],[178]]
y=[40,35,48,70,45,65]

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

LR = LinearRegression()
LR.fit(train_X, train_y)

# Streamlit app

st.title("Prediction of weight based on height")
st.write("Enter height in cm to predict weight in kg")

height_input = st.number_input("Enter height in cm : ", max_value=200, min_value=100, value=160)

# Predict weight
if st.button("Predict weight"):
    predict_weight = LR.predict([[height_input]])
    st.success(f"Predicted weight is : {predict_weight[0]:.2f} Kg")