import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime as dt
import datetime
import time
import requests

st.title('DHL Freight Brokerage Spot Rate Prediction')

#### pickup and delivery date and time ---------------------------------------------------
st.markdown('#### Schedule Your Shipment')

pu_date, pu_time, dl_date, dl_time = st.columns(4)

# define pickup date and time
pu_date.date_input('Pickup Date', key='pu_date')
pu_time.time_input('Pickup Time', datetime.time(9, 00), key='pu_time')

# define delivery date and time
dl_date.date_input('Delivery Date', key='dl_date')
dl_time.time_input('Delivery Time', datetime.time(9, 00), key='dl_time')

pu_appt = dt.combine(st.session_state.pu_date, st.session_state.pu_time)
dl_appt = dt.combine(st.session_state.dl_date, st.session_state.dl_time)

# raise error for inappropriate input of date and time
if dl_appt < pu_appt:
    st.error('Delivery date cannot be earlier than the pick up date!')

#### geographic infomation ---------------------------------------------------
st.markdown('### From')
# uplode us zipcode
# @st.cache
def load_state_abbr(country='us'):
    state_abbr = pd.read_csv(f'data/{country}_state_abbr.csv')
    return state_abbr
us_state_abbr = load_state_abbr('us')
ca_state_abbr = load_state_abbr('ca')

# location for origin
ori_country, ori_state, ori_city, ori_zip = st.columns(4)
ori_country.selectbox('Origin Country', ['United States', 'Canada'], key='origin_country')

with ori_state:
    if st.session_state.origin_country == 'United States':
        st.selectbox('State', us_state_abbr, key='select_box_1')
    elif st.session_state.origin_country == 'Canada':
        st.selectbox('State', ca_state_abbr, key='select_box_2')

ori_city.text_input('City', placeholder='Input city name', key='origin_city')
ori_zip.text_input('Zipcode', placeholder='(Optional)', key='origin_zip')

# location for destination ---------------------------------------------------
st.markdown('#### To')
des_country, des_state, des_city, des_zip = st.columns(4)
des_country.selectbox('Destinated Country', ['United States', 'Canada'], key='dest_country')

with des_state:
    if st.session_state.dest_country == 'United States':
        st.selectbox('State', us_state_abbr)
    elif st.session_state.dest_country == 'Canada':
        st.selectbox('State', ca_state_abbr)

des_city.text_input('City', placeholder='Input city name', key='dest_city')
des_zip.text_input('Zipcode', placeholder='(Optional)', key='dest_zip')

#### item Attributes ---------------------------------------------------
st.markdown('#### Item Description')

# weight, case, volume
weight, case, volume, mode = st.columns(4)
weight.number_input('Weight (lbs)', min_value=0.00, key='weight')
case.number_input('Case', min_value=0, key='case')
volume.number_input('Volume (cu ft)', min_value=0.00, key='volume')

# shipment mode
mode.selectbox('Shipment Mode', ['LTL', 'TL', 'Intermodal'])

# submit button
if st.button('Submit Order'):
    
    with st.spinner('Retrieving Result...'):
        time.sleep(2)
    
        # write API to post request and retrieve results.
        res = 1666.666

    st.success('Prediction Completed!')
    st.markdown(f'### Estimated price: {res}')

