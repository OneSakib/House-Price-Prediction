import locale
import streamlit as st
import pickle
import pandas as pd

model = pickle.load(open('RidgeModel.pkl', 'rb'))
df = pd.read_csv('cleaned_data.csv')
locations = df['location'].unique()
bhks = sorted(df['bhk'].unique())
bathrooms = sorted(df['bath'].unique())
st.title('House Price Prediction')
location = st.selectbox('Location', locations)
bhk = st.selectbox('BHK', bhks)
no_of_bathrooms = st.selectbox('No of Bathrooms', bathrooms)
total_sqft = st.number_input('Total Sqft')
predict_btn = st.button('Predict')
locale.setlocale(locale.LC_ALL, 'en_IN')
if predict_btn:
    input_data = pd.DataFrame([[total_sqft, bhk, no_of_bathrooms, location]], columns=[
                              'total_sqft', 'bhk', 'bath', 'location'])
    predict_price = model.predict(input_data)
    inr_format = locale.currency(
        predict_price[0]*100000, grouping=True, symbol="₹")
    st.write('Predicted Price:', inr_format)
