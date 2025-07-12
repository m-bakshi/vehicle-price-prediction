# Vehicle Price Prediction


This project predicts the price of a vehicle based on various features such as make, model, mileage, engine, transmission type and more. It uses XGBoost Regressor for modeling and includes a Streamlit web application for easy user interaction.


## Project files

vehicle_price_prediction.zip	                  #Raw dataset

vehicle_price_model.pkl	                        #Trained XGBoost model

onehotencoder.pkl	                              #Trained OneHotEncoder

app.py	                                        #Streamlit app script

README.md	                                      #Project documentation



## Libraries

pandas

numpy

scikit-learn

xgboost

joblib

matplotlib

streamlit


## Workflow

## 1. Data Preparation:

- Loaded data from a zip archive.

- Dropped rows with missing critical values and duplicates.

- Filled missing categorical values with "Unknown"

- Created an age feature from vehicle year and removes unnecessary columns

- Log-transformed the price for better regression stability

## 2. Feature Engineering & Encoding:

- Split data into training and test sets

- One-hot encoded categorical features using OneHotEncoder

- Combined numerical and encoded categorical features

## 3. Model Training & Hyperparameter Tuning:

- Used XGBRegressor

- Applied RandomizedSearchCV to find the best hyperparameters

- Fitted model on training data

## 4. Model Evaluation:

- Predicted on test set and inverse-transformed predictions

- Evaluated with Mean Absolute Error, Root Mean Squared Error and R² score

- Visualized top 10 feature importances from the trained model

- Saved the best XGBoost model and the one-hot encoder using joblib

## 5. Interactive Streamlit App

- Allows users to input vehicle details via a web interface

- Encodes inputs and predicts vehicle price dynamically


## Results

Mean Absolute Error: 3971.344932432433
Root Mean Squared Error: 9229.875959037921
R^2 Score: 0.81152374352559

## Visualisation

<img width="640" height="480" alt="top_10_feat_importances" src="https://github.com/user-attachments/assets/5fb0ef45-0acc-4447-9a81-cb5b0e12e7a5" />
