# Sales Prediction App

A simple Streamlit web application that predicts product sales based on advertising budgets (TV, Radio, Newspaper) using a Linear Regression model built with scikit-learn.

---


## Overview

This app demonstrates a basic machine learning workflow:

1. Load the Advertising dataset (`Advertising.csv`).
2. Train a Linear Regression model on advertising budgets (TV, Radio, Newspaper) to predict sales.
3. Provide a UI to input budgets and predict sales.
4. Display model metrics and visualizations (correlation heatmap, Actual vs Predicted scatter plot, etc.).

The UI is built with Streamlit for quick interactive prototyping.

---

## Dataset

Filename expected: `Advertising.csv`

Expected columns (case-sensitive):
- `TV` — TV advertising budget (numeric)
- `Radio` — Radio advertising budget (numeric)
- `Newspaper` — Newspaper advertising budget (numeric)
- `Sales` — Sales (numeric), target column

Place `Advertising.csv` in the same directory as `app.py` or update the path in `app.py` accordingly.

---

## Features & Target

- Features (X): `TV`, `Radio`, `Newspaper`
- Target (y): `Sales`

---

## How it works

- Data is loaded using pandas and cached with `@st.cache_data` to avoid reloading on every interaction.
- The dataset is split into training and testing sets with `train_test_split`.
- A `LinearRegression` model is trained on the training set.
- Predictions are generated on the test set for evaluation and on user inputs for on-demand predictions.
- Performance metrics shown:
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - R² Score

---



## Usage

- Enter numeric values for `TV`, `Radio`, and `Newspaper` advertising budgets using the numeric inputs in the sidebar/main area.
- Click the "Predict Sales" button to get a predicted sales value.
- Review model metrics shown (MAE, MSE, RMSE, R²) and visualize Actual vs Predicted results.
- Explore dataset preview and summary statistics.

---

## Model Performance & Evaluation

After training and testing the Linear Regression model, the app computes:
- MAE — mean absolute difference between actual and predicted sales.
- MSE — average squared difference.
- RMSE — square root of MSE (same units as target).
- R² Score — proportion of variance explained by the model.


---

## Graph's

 ![Input Form](https://github.com/Arshad-5068/AML-Task-5-/blob/main/graph%201.png?raw=true)

 ![Input Form](https://github.com/Arshad-5068/AML-Task-5-/blob/main/graph%202.png?raw=true)

 ![Input Form](https://github.com/Arshad-5068/AML-Task-5-/blob/main/graph%203.png?raw=true)

## Dashboard 

 ![Input Form](https://github.com/Arshad-5068/AML-Task-5-/blob/main/output.png?raw=true)
---

## Project Structure

- `app.py` — Main Streamlit application
- `Advertising.csv` — Dataset (required)
- `requirements.txt` — Python dependencies (recommended)

---
