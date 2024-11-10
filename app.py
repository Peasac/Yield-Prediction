import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
import base64

# Cache model loading and preparation
@st.cache_resource
def load_and_prepare_model():
    yield_data = pd.read_csv("yield_df.csv")
    yield_data_cleaned = yield_data.drop(columns=['Unnamed: 0'], errors='ignore')

    # Encoding categorical columns
    area_enc = LabelEncoder()
    item_enc = LabelEncoder()
    yield_data_cleaned['Area'] = area_enc.fit_transform(yield_data_cleaned['Area'])
    yield_data_cleaned['Item'] = item_enc.fit_transform(yield_data_cleaned['Item'])

    X = yield_data_cleaned.drop(columns=['hg/ha_yield'])
    y = yield_data_cleaned['hg/ha_yield']

    # Train-test split and scaling
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    # Train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model, scaler, area_enc, item_enc, yield_data

model, scaler, area_enc, item_enc, yield_data = load_and_prepare_model()

# Cache unique values for dropdowns
@st.cache_data
def get_unique_values(data):
    areas = data['Area'].unique()
    crops = data['Item'].unique()
    return areas, crops

areas, crops = get_unique_values(yield_data)

# Background image setup
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

img_file = 'nightcrop.jpg'
base64_img = get_base64_of_bin_file(img_file)

# Apply custom CSS for background, text, buttons, output messages, and footer
st.markdown(
    f"""
    <style>
    .stApp {{
        background: url("data:image/jpg;base64,{base64_img}") no-repeat center center fixed;
        background-size: cover;
        color: white;
        padding-bottom: 120px;  /* Adds space above footer */
    }}
    /* Ensure all text is white in both light and dark modes */
    .stText, .stTitle, .stHeader, .stSubheader {{
        color: white !important;
    }}
    /* Style for input field labels */
    label {{
        color: #000000 !important;  /* Dark color for readability */
        font-weight: bold;
        background-color: rgba(255, 255, 255, 0.7);  /* Semi-transparent background */
        padding: 5px;
        border-radius: 5px;
    }}
    /* Style input fields for visibility */
    .stNumberInput, .stSelectbox {{
        background-color: rgba(255, 255, 255, 0.8);  /* Semi-transparent input background */
        color: black !important;  /* Black text in inputs */
        border-radius: 5px;
    }}
    /* Style for buttons */
    .stButton button {{
        color: black !important;  /* Button text in black */
        background-color: #F0F0F0 !important;  /* Light button background */
        border-radius: 5px;
        font-weight: bold;
    }}
    /* Style the output text for visibility in light mode */
    .stSuccess, .stError {{
        color: black !important;  /* Black color for text */
        background-color: rgba(255, 255, 255, 0.8) !important; /* Light background */
        border-radius: 5px;
        font-weight: bold;
    }}
    /* Footer styling */
    .footer {{
        width: 100%;
        background-color: #1F2937;
        color: white;
        text-align: center;
        padding: 10px;
        position: relative;
        margin-top: 20px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit UI
st.title("Agricultural Yield Prediction App")

# Collect user inputs
st.header("Enter details for yield prediction:")
area = st.selectbox("Select Area", areas)
crop = st.selectbox("Select Crop Type", crops)
year = st.number_input("Enter the Year", min_value=1900, max_value=2100, value=2023)
avg_rainfall = st.number_input("Average Rainfall (mm/year)", min_value=0.0)
pesticides = st.number_input("Amount of Pesticides Used (tonnes)", min_value=0.0)
avg_temp = st.number_input("Average Temperature (°C)", min_value=-50.0, max_value=60.0)

# Prediction function
def predict_yield(area, crop, year, avg_rainfall, pesticides, avg_temp):
    try:
        area_encoded = area_enc.transform([area])[0]
        crop_encoded = item_enc.transform([crop])[0]
    except ValueError as e:
        return f"Error: {e}. Please select valid Area and Crop."

    input_data = pd.DataFrame([[area_encoded, crop_encoded, year, avg_rainfall, pesticides, avg_temp]],
                              columns=['Area', 'Item', 'Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp'])
    
    input_data_scaled = scaler.transform(input_data)
    predicted_yield = model.predict(input_data_scaled)
    
    return predicted_yield[0]

if st.button("Predict Yield"):
    predicted_yield = predict_yield(area, crop, year, avg_rainfall, pesticides, avg_temp)
    if isinstance(predicted_yield, str):
        st.error(predicted_yield)
    else:
        st.success(f"Predicted Yield (hg/ha): {predicted_yield:.2f}")

st.markdown("""<div class="footer">Copyright © SmartSow2024</div>""", unsafe_allow_html=True)



