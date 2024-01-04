import pickle
with open(r"C:\Users\HP\OneDrive\Desktop\window\rf_regressor.pkl","rb") as f:
    rf_regressor=pickle.load(f)
import streamlit as st
# Load the trained model

import pandas as pd
# Function to perform predictions using the loaded model
from sklearn.impute import SimpleImputer

# Function to perform predictions using the loaded model
# ... (Previous code remains unchanged)
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor# Replace 'your_model.pkl' with your model file

# Function to perform predictions using the loaded model
def predict_house_price(area, bedrooms, bathrooms, stories, mainroad, guestroom,
                        basement, hotwaterheating, airconditioning, parking, prefarea,
                        furnishingstatus):
    # Prepare input data for prediction
    input_data = {
        'area': area,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'stories': stories,
        'mainroad': mainroad,
        'guestroom': guestroom,
        'basement': basement,
        'hotwaterheating': hotwaterheating,
        'airconditioning': airconditioning,
        'parking': parking,
        'prefarea': prefarea,
        'furnishingstatus': furnishingstatus
    }

    # Create a DataFrame from the input data
    input_df = pd.DataFrame([input_data])

    # Realign columns to ensure they're in the same order as in the model
    expected_columns = ['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom',
                        'basement', 'hotwaterheating', 'airconditioning', 'parking', 'prefarea',
                        'furnishingstatus']  # Add other columns as required

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df.reindex(columns=expected_columns, fill_value=0)

    # Ensure the number of features matches the model's expectations
    if input_df.shape[1] != 12:  # Update this number according to the exact number of features
        raise ValueError(f"Input X has {input_df.shape[1]} features, but RandomForestRegressor is expecting 12 features.")

    # Convert the DataFrame to a NumPy array for prediction
    input_array = input_df.values

    # Perform predictions using the loaded model
    predicted_price = rf_regressor.predict(input_array)  # Use the input array for prediction
    

    return predicted_price

# Streamlit UI
def main():
    st.title('House Price Prediction')

    # Input features for prediction
    area = st.number_input('Area', min_value=0)
    bedrooms = st.number_input('Number of Bedrooms', min_value=0)
    bathrooms = st.number_input('Number of Bathrooms', min_value=0)
    stories = st.number_input('Number of Stories', min_value=0)
    mainroad = st.selectbox('Main Road (Yes/No)', ['Yes', 'No'])
    guestroom = st.selectbox('Guest Room (Yes/No)', ['Yes', 'No'])
    basement = st.selectbox('Basement (Yes/No)', ['Yes', 'No'])
    hotwaterheating = st.selectbox('Hot Water Heating (Yes/No)', ['Yes', 'No'])
    airconditioning = st.selectbox('Air Conditioning (Yes/No)', ['Yes', 'No'])
    parking = st.number_input('Parking Spaces', min_value=0)
    prefarea = st.selectbox('Preferred Area (Yes/No)', ['Yes', 'No'])
    furnishingstatus = st.selectbox('Furnishing Status', ['Furnished', 'Semi-Furnished', 'Unfurnished'])

    if st.button('Predict Price'):
        # Convert categorical variables to numerical values
        mainroad_encoded = 1 if mainroad == 'Yes' else 0
        guestroom_encoded = 1 if guestroom == 'Yes' else 0
        basement_encoded = 1 if basement == 'Yes' else 0
        hotwaterheating_encoded = 1 if hotwaterheating == 'Yes' else 0
        airconditioning_encoded = 1 if airconditioning == 'Yes' else 0
        prefarea_encoded = 1 if prefarea == 'Yes' else 0

        # Map furnishingstatus to encoded values
        furnishingstatus_encoded = {'Furnished': 0, 'Semi-Furnished': 1, 'Unfurnished': 2}[furnishingstatus]

        # Perform prediction
        predicted_price = predict_house_price(area, bedrooms, bathrooms, stories, mainroad_encoded, guestroom_encoded,
                                             basement_encoded, hotwaterheating_encoded, airconditioning_encoded,
                                             parking, prefarea_encoded, furnishingstatus_encoded)

        st.success(f'Predicted Price: ${predicted_price[0]:,.2f}')

if __name__ == '__main__':
    main()
