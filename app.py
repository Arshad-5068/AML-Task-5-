import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


st.set_page_config(
    page_title="Sales Prediction App",
    
    layout="wide"
)

st.title("Sales Prediction using Machine Learning")
st.write(
    "This web app predicts **product sales** based on advertising expenditure "
    "using a **Linear Regression** model."
)


@st.cache_data
def load_data():
    data = pd.read_csv("Advertising.csv")
    return data

data = load_data()


X = data[["TV", "Radio", "Newspaper"]]
y = data["Sales"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


st.subheader("Predict Sales")

tv = st.number_input("TV Advertising Budget", min_value=0.0, step=1.0)
radio = st.number_input("Radio Advertising Budget", min_value=0.0, step=1.0)
newspaper = st.number_input("Newspaper Advertising Budget", min_value=0.0, step=1.0)

if st.button("Predict Sales"):
    input_data = pd.DataFrame({
        "TV": [tv],
        "Radio": [radio],
        "Newspaper": [newspaper]
    })
    prediction = model.predict(input_data)
    st.success(f"Predicted Sales: {prediction[0]:.2f}")

st.subheader("Model Performance")

col3, col4, col5, col6 = st.columns(4)

col3.metric("MAE", round(mean_absolute_error(y_test, y_pred), 2))
col4.metric("MSE", round(mean_squared_error(y_test, y_pred), 2))
col5.metric("RMSE", round(np.sqrt(mean_squared_error(y_test, y_pred)), 2))
col6.metric("R² Score", round(r2_score(y_test, y_pred), 2))

st.subheader("Actual vs Predicted Sales")

fig, ax = plt.subplots(figsize=(4, 3))  
ax.scatter(y_test, y_pred)
ax.set_xlabel("Actual Sales")
ax.set_ylabel("Predicted Sales")
ax.set_title("Actual vs Predicted Sales")

st.pyplot(fig, use_container_width=False)

st.subheader("Data Visualization")

col1, col2 = st.columns(2)

with col1:
    st.write("### Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

with col2:
    st.write("### TV Advertising vs Sales")
    fig, ax = plt.subplots()
    sns.scatterplot(x=data["TV"], y=data["Sales"], ax=ax)
    st.pyplot(fig)

st.subheader("Dataset Preview")
st.dataframe(data.head())

st.subheader("Dataset Statistics")
st.write(data.describe())

st.markdown("---")
st.markdown("**Sales Prediction App | Built with Python & Streamlit**")
