import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime as dt, timedelta
import datetime
import time
import requests
from PIL import Image

image = Image.open('data/dhl_image.jpeg')
st.image(image)
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

# store PU_APPT and DL_APPT
pu_appt = dt.combine(st.session_state.pu_date, st.session_state.pu_time)
dl_appt = dt.combine(st.session_state.dl_date, st.session_state.dl_time)

# raise error for inappropriate input of date and time
if dl_appt < pu_appt:
    st.error('Delivery date cannot be earlier than the pick up date!')

#### geographic infomation ---------------------------------------------------
st.markdown('### From')

# uplode us zipcode ---------------------------------------------------
@st.cache
def load_state_abbr(country='us'):
    state_abbr = pd.read_csv(f'data/{country}_state_abbr.csv')
    return state_abbr
us_state_abbr = load_state_abbr('us')
ca_state_abbr = load_state_abbr('ca')

# location for origin
ori_country, ori_state, ori_city, ori_zip = st.columns(4)
ori_country.selectbox('Origin Country', ['United States', 'Canada'], key='origin_country')

# select box for US and Canada origin states respectively
with ori_state:
    if st.session_state.origin_country == 'United States':
        st.selectbox('State', us_state_abbr, key='origin_state')
    elif st.session_state.origin_country == 'Canada':
        st.selectbox('State', ca_state_abbr, key='origin_state')

# text input box for origin city and zipcode
ori_city.text_input('City', placeholder='Enter city', key='origin_city')
ori_zip.text_input('Zipcode', placeholder='(Optional)', key='origin_zip')

# location for destination ---------------------------------------------------
st.markdown('#### To')
des_country, des_state, des_city, des_zip = st.columns(4)
des_country.selectbox('Destinated Country', ['United States', 'Canada'], key='dest_country')

# select box for US and Canada dest states respectively
with des_state:
    if st.session_state.dest_country == 'United States':
        st.selectbox('State', us_state_abbr, key='dest_state')
    elif st.session_state.dest_country == 'Canada':
        st.selectbox('State', ca_state_abbr, key='dest_state')

# text input box for dest city and zipcode
des_city.text_input('City', placeholder='Enter city', key='dest_city')
des_zip.text_input('Zipcode', placeholder='(Optional)', key='dest_zip')

#### class to calculate distance ---------------------------------------------------
class GeographicDimensions:

    def __init__(self, unit:str):
        self.unit = unit
        self.ori_unit = 'ORIGIN_' + self.unit
        self.des_unit = 'DEST_' + self.unit
        self.data = pd.read_csv(f'data/unique_distance_by_{self.unit.lower()}.csv', 
                                dtype={'ORIGIN_ZIP':'str', 'DEST_ZIP':'str'})
    
    def distance_calculator(self):
        if self.unit == 'ZIP':
            self.origin_loc = st.session_state.origin_zip
            self.dest_loc = st.session_state.dest_zip
        
            # if no zipcode provided, jump to using city-paired distance
            if self.origin_loc == '' and self.dest_loc == '':
                self.unit = 'CITY' 
                self.ori_unit = 'ORIGIN_' + self.unit
                self.des_unit = 'DEST_' + self.unit
                self.data = pd.read_csv(f'data/unique_distance_by_{self.unit.lower()}.csv')
            
            # if the zipcodes provided are identical, return default distance as 5
            elif self.origin_loc == self.dest_loc and self.unit == 'ZIP':
                self.distance = 5
                return self.distance

            # if the first 3 digits of zipcodes provided are identical, return default distance as 15
            elif self.origin_loc[:3] == self.dest_loc[:3] and self.unit == 'ZIP':
                self.distance = 15
                return self.distance

        if self.unit == 'CITY':
            self.origin_loc = st.session_state.origin_city.lower()
            self.dest_loc = st.session_state.dest_city.lower()

            # if cities provided are identical, return default distance as 35
            if self.origin_loc == self.dest_loc:
                self.distance = 35
                return self.distance

        if self.unit == 'STATE':
            self.origin_loc = st.session_state.origin_state
            self.dest_loc = st.session_state.dest_state

            # if states provided are identical, return default distance as 200
            if self.origin_loc == self.dest_loc:
                self.distance = 200
                return self.distance

        self.distance = self.retrieve_value('DISTANCE')
        return self.distance

    def retrieve_value(self, col):
        '''Retrieve specific values in the data defined above'''
        data = self.data

        self.res = (data.loc[
                (data[self.ori_unit] == self.origin_loc) &
                (data[self.des_unit] == self.dest_loc),
                col
            ].values[0]
        )
        return self.res

    def geographic_att(self):
        self.origin_lat = self.retrieve_value('ORIGIN_LAT')
        self.origin_lng = self.retrieve_value('ORIGIN_LNG')
        self.dest_lat = self.retrieve_value('DEST_LAT')
        self.dest_lng = self.retrieve_value('DEST_LNG')

        return [self.origin_lat, self.origin_lng, self.dest_lat, self.dest_lng]

    def return_raw_data(self):
        return self.data

#### item attributes ---------------------------------------------------
st.markdown('#### Item Description')

# weight, case, volume
weight, case, volume, mode = st.columns(4)
weight.number_input('Weight (lbs)', min_value=0.00, key='weight')
case.number_input('Case', min_value=0, key='case')
volume.number_input('Volume (cu ft)', min_value=0.00, key='volume')

# shipment mode
mode.selectbox('Shipment Mode', ['LTL', 'TL', 'Intermodal'], key='actual_mode')

# submit button
if st.button('Submit Order'):
    
    with st.spinner('Retrieving Result...'):
        
        # calculate distance
        try:
            res = GeographicDimensions('ZIP')
            dis = res.distance_calculator()
            origin_lat, origin_lng, dest_lat, dest_lng = res.geographic_att()
        except:
            try:
                res = GeographicDimensions('CITY')
                dis = res.distance_calculator()
                origin_lat, origin_lng, dest_lat, dest_lng = res.geographic_att()
            except:
                try:
                    res = GeographicDimensions('STATE')
                    dis = res.distance_calculator()
                    origin_lat, origin_lng, dest_lat, dest_lng = res.geographic_att()
                except:
                    dis = 150
                    # load distance csv file to extract mean lat, lng
                    distance_by_zip = pd.read_csv('data/unique_distance_by_zip.csv')
                    origin_lat, origin_lng, dest_lat, dest_lng = distance_by_zip[['ORIGIN_LAT', 'ORIGIN_LNG', 'DEST_LAT', 'DEST_LNG']].mean().values
        
        # calculate estimated delivery time
        est_delivery_day = round((dl_appt - pu_appt) / timedelta(days=1), 1)

        # calculate estimated delivery speed
        est_delivery_speed = round(dis / ((dl_appt - pu_appt).total_seconds() / 3600), 4)

        # assemble all inputs
        post_input = ['EXEL', '2', st.session_state.actual_mode, '53DV', 
                      st.session_state.origin_city, st.session_state.origin_state,
                      st.session_state.dest_city, st.session_state.dest_state, dis, 
                      st.session_state.case, st.session_state.weight, st.session_state.volume,
                      est_delivery_day, est_delivery_speed, origin_lat, origin_lng, dest_lat, dest_lng,
                      pu_appt]

        # convert to str values and concat together
        post_input = ', '.join([str(i) for i in post_input])

        # generate input for POST
        final_input = {'data':post_input}

        # create request
        url = r'https://3ihagmn4a2.execute-api.us-east-1.amazonaws.com/prod/predict-dhl-price'
        r = requests.post(url=url, json=final_input)

        # retrieve status
        status = r.status_code
        # retrieve result
        predicted_price = round(float(r.text), 2)

    st.success('Prediction Completed!')
    st.markdown(f'### Estimated Distance: {dis} miles')
    st.markdown(f'### Estimated Delivery Time: {est_delivery_day} days')
    st.markdown(f'### Status: {status}')
    st.markdown(f'### Predicted Price: ${predicted_price}')
    st.write(f'{final_input}')

