import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Real Estate Investment Advisor",
    page_icon="🏠",
    layout="wide"
)

# --------------------------------------------------
# LOAD MODELS AND FILES
# --------------------------------------------------

@st.cache_resource
def load_models():
    classifier = joblib.load("xgboost_classifier.pkl")
    regressor = joblib.load("xgboost_regressor.pkl")

    classifier_features = joblib.load(
        "xgboost_classification_features.pkl"
    )

    regression_features = joblib.load(
        "xgboost_regression_features.pkl"
    )

    locality_map = joblib.load(
        "locality_frequency_map.pkl"
    )

    amenity_classes = joblib.load(
        "amenity_classes.pkl"
    )

    return (
        classifier,
        regressor,
        classifier_features,
        regression_features,
        locality_map,
        amenity_classes
    )


(
    classifier,
    regressor,
    classifier_features,
    regression_features,
    locality_map,
    amenity_classes
) = load_models()


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("Real Estate Investment Advisor")

st.write(
    "Enter property details to predict property profitability "
    "and estimated property value."
)

st.divider()


# --------------------------------------------------
# PROPERTY DETAILS
# --------------------------------------------------

st.header("Property Details")

col1, col2, col3 = st.columns(3)

with col1:
    bhk = st.number_input(
        "BHK",
        min_value=1,
        max_value=10,
        value=2
    )

    size = st.number_input(
        "Size (Square Feet)",
        min_value=100.0,
        value=1000.0
    )


with col2:
    price = st.number_input(
        "Current Price (Lakhs)",
        min_value=1.0,
        value=50.0
    )

    year_built = st.number_input(
        "Year Built",
        min_value=1900,
        max_value=2026,
        value=2020
    )


with col3:

    state = st.selectbox(
        "State",
        [
            "Andhra Pradesh",
            "Assam",
            "Bihar",
            "Chhattisgarh",
            "Delhi",
            "Gujarat",
            "Haryana",
            "Jharkhand",
            "Karnataka",
            "Kerala",
            "Madhya Pradesh",
            "Maharashtra",
            "Odisha",
            "Punjab",
            "Rajasthan",
            "Tamil Nadu",
            "Telangana",
            "Uttar Pradesh",
            "Uttarakhand",
            "West Bengal"
        ]
    )

    city = st.text_input(
        "City",
        "Visakhapatnam"
    )


# --------------------------------------------------
# PROPERTY TYPE
# --------------------------------------------------

st.header("Property Information")

col1, col2, col3 = st.columns(3)

with col1:

    property_type = st.selectbox(
        "Property Type",
        [
            "Apartment",
            "Independent House",
            "Villa"
        ]
    )

    furnishing_status = st.selectbox(
        "Furnishing Status",
        [
            "Furnished",
            "Semi-furnished",
            "Unfurnished"
        ]
    )


with col2:

    parking = st.selectbox(
        "Parking Space",
        [
            "No",
            "Yes"
        ]
    )

    security = st.selectbox(
        "Security",
        [
            "No",
            "Yes"
        ]
    )

    facing = st.selectbox(
        "Facing Direction",
        [
            "East",
            "North",
            "South",
            "West"
        ]
    )


with col3:

    owner_type = st.selectbox(
        "Owner Type",
        [
            "Broker",
            "Builder",
            "Owner"
        ]
    )

    availability = st.selectbox(
        "Availability Status",
        [
            "Ready to Move",
            "Under Construction"
        ]
    )

    transport = st.selectbox(
        "Public Transport Accessibility",
        [
            "Low",
            "Medium",
            "High"
        ]
    )


# --------------------------------------------------
# LOCALITY
# --------------------------------------------------

st.header("Location Details")

locality = st.selectbox(
    "Select Locality",
    sorted(locality_map.keys())
)


# --------------------------------------------------
# AMENITIES
# --------------------------------------------------

st.header("Amenities")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    clubhouse = st.checkbox("Clubhouse")

with col2:
    garden = st.checkbox("Garden")

with col3:
    gym = st.checkbox("Gym")

with col4:
    playground = st.checkbox("Playground")

with col5:
    pool = st.checkbox("Pool")


# --------------------------------------------------
# PREDICTION BUTTON
# --------------------------------------------------

st.divider()

if st.button("Analyze Property", use_container_width=True):

    # ----------------------------------------------
    # CALCULATED FEATURES
    # ----------------------------------------------

    price_per_sqft = (
        price * 100000
    ) / size

    age_of_property = 2026 - year_built


    # ----------------------------------------------
    # CREATE BASE INPUT
    # ----------------------------------------------

    input_data = {}

    input_data["BHK"] = bhk
    input_data["Size_in_SqFt"] = size
    input_data["Price_in_Lakhs"] = price
    input_data["Price_per_SqFt"] = price_per_sqft
    input_data["Year_Built"] = year_built
    input_data["Age_of_Property"] = age_of_property


    # ----------------------------------------------
    # LOCALITY FREQUENCY
    # ----------------------------------------------

    input_data["Locality_Frequency"] = locality_map.get(
        locality,
        0
    )


    # ----------------------------------------------
    # ONE HOT ENCODING
    # ----------------------------------------------

    all_features = list(
        set(
            classifier_features +
            regression_features
        )
    )

    for feature in all_features:
        if feature not in input_data:
            input_data[feature] = 0


    # State
    state_feature = "State_" + state
    if state_feature in input_data:
        input_data[state_feature] = 1


    # Property Type
    property_feature = "Property_Type_" + property_type
    if property_feature in input_data:
        input_data[property_feature] = 1


    # Furnishing
    furnishing_feature = (
        "Furnished_Status_" +
        furnishing_status
    )

    if furnishing_feature in input_data:
        input_data[furnishing_feature] = 1


    # Parking
    parking_feature = (
        "Parking_Space_" +
        parking
    )

    if parking_feature in input_data:
        input_data[parking_feature] = 1


    # Security
    security_feature = (
        "Security_" +
        security
    )

    if security_feature in input_data:
        input_data[security_feature] = 1


    # Facing
    facing_feature = (
        "Facing_" +
        facing
    )

    if facing_feature in input_data:
        input_data[facing_feature] = 1


    # Owner
    owner_feature = (
        "Owner_Type_" +
        owner_type
    )

    if owner_feature in input_data:
        input_data[owner_feature] = 1


    # Availability
    availability_feature = (
        "Availability_Status_" +
        availability
    )

    if availability_feature in input_data:
        input_data[availability_feature] = 1


    # Transport
    transport_feature = (
        "Public_Transport_Accessibility_" +
        transport
    )

    if transport_feature in input_data:
        input_data[transport_feature] = 1


    # Amenities
    input_data["Amenity_Clubhouse"] = int(clubhouse)
    input_data["Amenity_Garden"] = int(garden)
    input_data["Amenity_Gym"] = int(gym)
    input_data["Amenity_Playground"] = int(playground)
    input_data["Amenity_Pool"] = int(pool)


    # ----------------------------------------------
    # CREATE DATAFRAMES
    # ----------------------------------------------

    classification_input = pd.DataFrame(
        [input_data]
    )

    regression_input = pd.DataFrame(
        [input_data]
    )


    # Match exact feature order

    classification_input = classification_input.reindex(
        columns=classifier_features,
        fill_value=0
    )

    regression_input = regression_input.reindex(
        columns=regression_features,
        fill_value=0
    )


    # ----------------------------------------------
    # PREDICTIONS
    # ----------------------------------------------

    classification_prediction = classifier.predict(
        classification_input
    )[0]

    regression_prediction = regressor.predict(
        regression_input
    )[0]


    # ----------------------------------------------
    # RESULTS
    # ----------------------------------------------

    st.header("Prediction Results")

    result_col1, result_col2 = st.columns(2)


    with result_col1:

        st.metric(
            "Estimated Property Value",
            f"₹ {regression_prediction:,.2f} Lakhs"
        )


    with result_col2:

        if classification_prediction == 1:

            st.success(
                "Profitable Investment"
            )

        else:

            st.warning(
                "Lower Investment Potential"
            )


    # ----------------------------------------------
    # INVESTMENT RECOMMENDATION
    # ----------------------------------------------

    st.subheader("Investment Recommendation")

    if classification_prediction == 1:

        st.write(
            "Based on the trained machine learning model, "
            "this property shows positive investment potential."
        )

    else:

        st.write(
            "Based on the trained machine learning model, "
            "this property may have lower investment potential."
        )