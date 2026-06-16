# 🏠 House Price Prediction

## Overview

This project predicts house prices using Machine Learning. A Random Forest Regression model is trained on housing data to estimate the price of a house based on features such as area, number of bedrooms, bathrooms, parking spaces, furnishing status, and other property details.

The dataset is automatically downloaded from Kaggle, preprocessed, and used to train the model. Users can then enter house details through the command line and receive a predicted house price.

---

## Features

* Automatic dataset download using KaggleHub
* Data cleaning and preprocessing
* Encoding of categorical features
* Random Forest Regression model
* Model performance evaluation
* Interactive price prediction

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* KaggleHub

---

## Dataset Features

The model uses the following features:

* Area
* Bedrooms
* Bathrooms
* Stories
* Parking
* Main Road Access
* Guest Room
* Basement
* Hot Water Heating
* Air Conditioning
* Preferred Area
* Furnishing Status

**Target Variable:** Price

---

## How It Works

1. Download the housing dataset.
2. Remove missing values.
3. Convert categorical data into numerical values.
4. Split the data into training and testing sets.
5. Train a Random Forest Regressor.
6. Evaluate model performance using RMSE and R² Score.
7. Predict house prices based on user input.

---

## Installation

Install the required libraries:

```bash
pip install pandas numpy scikit-learn kagglehub
```

---

## Run the Project

```bash
python house_price.py
```

Enter the requested house details when prompted, and the model will display the predicted price.

---

## Example Output

```text
MSE: 123456789
RMSE: 11111.11
R2: 0.89

Predicted Price: 7500000
```

---

## Future Improvements

* Save and load trained models
* Build a web interface using Streamlit or Flask
* Hyperparameter tuning for better accuracy
* Deploy the model online

