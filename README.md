# Real Estate Investment Advisor

A Machine Learning-powered web application that helps users analyze real estate properties and make data-driven investment decisions.

## Project Overview

The Real Estate Investment Advisor uses trained XGBoost machine learning models to analyze property details and provide:

- Property investment potential prediction
- Estimated property value
- Investment recommendation

The application provides an interactive interface built with Streamlit, allowing users to enter property details, select a locality, choose available amenities, and receive predictions.

## Features

- Predicts property investment potential
- Estimates property value
- Provides an investment recommendation
- Interactive Streamlit web application
- Locality-based property analysis
- Amenity selection
- Uses trained XGBoost machine learning models

## Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Joblib

## Machine Learning Models

### Classification Model

The XGBoost classification model predicts the investment potential of a property, such as:

- Profitable Investment Potential
- Lower Investment Potential

### Regression Model

The XGBoost regression model estimates the property value based on the details entered by the user.

## Input Features

The application analyzes multiple property-related features, including:

- BHK
- Property size
- Price
- Price per square foot
- Year built
- Age of property
- State and city
- Locality
- Property type
- Furnishing status
- Parking space
- Security
- Facing direction
- Owner type
- Availability status
- Public transport accessibility

### Amenities

Users can select available amenities such as:

- Clubhouse
- Garden
- Gym
- Playground
- Pool

## Project Structure

```text
Real_Estate_Project/
│
├── streamlit_app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── xgboost_classifier.pkl
├── xgboost_classification_features.pkl
│
├── xgboost_regressor.pkl
├── xgboost_regression_features.pkl
│
├── locality_frequency_map.pkl
└── amenity_classes.pkl
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/srivalli5454/Real-Estate-Investment-Advisor.git
```

### 2. Navigate to the project folder

```bash
cd Real-Estate-Investment-Advisor
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

**macOS/Linux:**

```bash
source venv/bin/activate
```

**Windows:**

```bash
venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the application

```bash
streamlit run streamlit_app.py
```

The application will open in your browser, usually at:

```text
http://localhost:8501
```

## Application Output

The application provides:

- Estimated Property Value
- Investment Potential Prediction
- Investment Recommendation

## Future Improvements

- Use actual locality names instead of encoded locality labels
- Add more real estate features
- Improve prediction accuracy with larger datasets
- Add property comparison functionality
- Add market trend visualizations
- Deploy the application online

## Author

**Srivalli Korada**

B.Tech CSIT Student  
Aspiring Data Scientist

## GitHub Repository

https://github.com/srivalli5454/Real-Estate-Investment-Advisor