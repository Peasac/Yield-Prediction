import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
yield_data = pd.read_csv("Ayield_df.csv")

# Drop unnecessary columns
yield_data_cleaned = yield_data.drop(columns=['Unnamed: 0'])

# Encode categorical columns with separate encoders for Area and Item
area_enc = LabelEncoder()
item_enc = LabelEncoder()
yield_data_cleaned['Area'] = area_enc.fit_transform(yield_data_cleaned['Area'])
yield_data_cleaned['Item'] = item_enc.fit_transform(yield_data_cleaned['Item'])

# Define features (X) and target (y)
X = yield_data_cleaned.drop(columns=['hg/ha_yield'])
y = yield_data_cleaned['hg/ha_yield']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the feature data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Function to predict yield
def predict_yield(area, crop, year, avg_rainfall, pesticides, avg_temp):
    try:
        # Encode input area and crop using the same LabelEncoders
        area_encoded = area_enc.transform([area])[0]
        crop_encoded = item_enc.transform([crop])[0]
    except ValueError as e:
        return f"Error: {e}. Please enter a valid Area and Crop from the dataset."

    # Create a DataFrame for the input
    input_data = pd.DataFrame([[area_encoded, crop_encoded, year, avg_rainfall, pesticides, avg_temp]],
                              columns=['Area', 'Item', 'Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp'])
    
    # Standardize the input data
    input_data_scaled = scaler.transform(input_data)
    
    # Predict yield
    predicted_yield = model.predict(input_data_scaled)
    
    return predicted_yield[0]

# Collecting user inputs
area = input("Enter the Area: ")
crop = input("Enter the Crop Type: ")
year = int(input("Enter the Year: "))
avg_rainfall = float(input("Enter the Average Rainfall (mm/year): "))
pesticides = float(input("Enter the Amount of Pesticides Used (tonnes): "))
avg_temp = float(input("Enter the Average Temperature (°C): "))

# Predict yield based on user inputs
predicted_yield = predict_yield(area, crop, year, avg_rainfall, pesticides, avg_temp)
print(f"Predicted Yield (hg/ha): {predicted_yield}")
