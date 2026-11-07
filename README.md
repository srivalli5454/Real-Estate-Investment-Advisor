# Real Estate Investment Advisor

A Machine Learning-based web application that helps users analyze real estate properties by predicting:

- Estimated Property Value
- Investment Potential
- Investment Recommendation

The application uses trained Machine Learning models and provides an interactive interface built with Streamlit.

## Live Application

You can access the deployed application here:

**https://real-estate-investment-advisor-h963vhfzt2dznvmshprgnd.streamlit.app**

## Features

- Predicts estimated property value
- Predicts investment potential
- Provides investment recommendations
- Accepts property details such as BHK, price, size, location, and property features
- Interactive and user-friendly Streamlit interface

## Technologies Used

- Python
- Streamlit
- Scikit-learn
- XGBoost
- Pandas
- NumPy
- Machine Learning

## Project Structure

```text
Real_Estate_Project/
│
├── app.py/
│   ├── classification_features.pkl
│   ├── classification_models.pkl
│   ├── regression_features.pkl
│   └── regression_models.pkl
│
├── amenity_classes.pkl
├── locality_frequency_map.pkl
├── xgboost_classification_features.pkl
├── xgboost_classifier.pkl
├── xgboost_regression_features.pkl
├── xgboost_regressor.pkl
├── streamlit_app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation and Setup

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
- Deploy additional versions with enhanced features

## Author

**Srivalli Korada**

B.Tech CSIT Student  
Aspiring Data Scientist

## GitHub Repository

https://github.com/srivalli5454/Real-Estate-Investment-Advisor

## Live Deployment

https://real-estate-investment-advisor-h963vhfzt2dznvmshprgnd.streamlit.app