import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, f_regression

#Import data
df = pd.read_csv("./house_data.csv")
df = df.drop('date', axis=1)
X= df.drop('price', axis=1)
y = df['price']

#Select top 5 feature
selector = SelectKBest(score_func=f_regression, k=5)
fit = selector.fit(X, y)

selected_features = X.columns[selector.get_support()]
print("Top 5 features:", selected_features.tolist())

X_selected = X[selected_features]

#train-test split
train_X, test_X, train_y, test_y = train_test_split(X_selected, y, random_state=42, test_size=0.2)

LR = LinearRegression()
LR.fit(train_X, train_y)

# Streamlit
st.title("House price prediction system")
st.write("Enter the inputs this system will predict price: ")

input_data = []
for feature in selected_features:
    val = st.number_input(f"Enter value of {feature}", value=float(X[feature].mean()))
    input_data.append(val)
    
if st.button("Predict Price"):
    input_array = np.array(input_data).reshape(1,-1)
    predicted_value = LR.predict(input_array)
    st.success(f"The predicte price is ${predicted_value[0]:.2f}")