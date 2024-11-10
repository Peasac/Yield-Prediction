import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def preprocess_data(yield_data):
    # Label encoding
    area_enc = LabelEncoder()
    item_enc = LabelEncoder()
    yield_data['Area'] = area_enc.fit_transform(yield_data['Area'])
    yield_data['Item'] = item_enc.fit_transform(yield_data['Item'])
    
    # Splitting features and target
    X = yield_data.drop(columns=['hg/ha_yield'])
    y = yield_data['hg/ha_yield']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, scaler, area_enc, item_enc
