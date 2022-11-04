import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime as dt, timedelta
import datetime
import time
import requests
from PIL import Image
from feature_engine.encoding import RareLabelEncoder, CountFrequencyEncoder, OrdinalEncoder
import joblib
import lightgbm

image = Image.open('data/dhl_image.jpeg')
st.image(image)
st.title('DHL Freight Brokerage Spot Rate Prediction')

#### pickup and delivery date and time ---------------------------------------------------
st.markdown('### Schedule Your Shipment')

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

#### Feature Engineering ---------------------------------------------------

### convert datetime features ---------------------------------------------------
time_features = pd.DataFrame({'PU_APPT':[pu_appt], 'DL_APPT':[dl_appt]})

### function to extract time attritubes ---------------------------------------------------
def time_attritubes(cols:str=['PU_APPT', 'DL_APPT'], df:pd.DataFrame=time_features):
    '''Return specific datetime attritubes.'''
    df['PU_APPT_YEAR'] = df['PU_APPT'].dt.year
    df['DL_APPT_YEAR'] = df['DL_APPT'].dt.year

    df['PU_APPT_MONTH'] = df['PU_APPT'].dt.month
    df['DL_APPT_MONTH'] = df['DL_APPT'].dt.month

    df['PU_APPT_QUARTER'] = df['PU_APPT'].dt.quarter
    df['DL_APPT_QUARTER'] = df['DL_APPT'].dt.quarter

    df['PU_APPT_WEEK'] = df['PU_APPT'].dt.isocalendar()['week']
    df['DL_APPT_WEEK'] = df['DL_APPT'].dt.isocalendar()['week']

    df['PU_APPT_DAY_OF_WEEK'] = df['PU_APPT'].dt.day_of_week
    df['DL_APPT_DAY_OF_WEEK'] = df['DL_APPT'].dt.day_of_week

    df['PU_APPT_IS_WEEKEND'] = (df['PU_APPT_DAY_OF_WEEK'] > 4).astype(np.int32)
    df['DL_APPT_IS_WEEKEND'] = (df['DL_APPT_DAY_OF_WEEK'] > 4).astype(np.int32)

    # define seasonality
    # quiet season: January - March
    quiet_season = [1, 2, 3]
    # produce season: April - July
    produce_season = [4, 5, 6, 7]
    # peak season: August - October
    peak_season = [8, 9, 10]
    # holiday season: November - December
    holiday_season = [11, 12]

    # PU_APPT_MONTH or DL_APPT_MONTH is in quiet season
    df["IS_QUIET_SEASON"] = (
        ((df["PU_APPT_MONTH"].isin(quiet_season)) | 
            (df["DL_APPT_MONTH"].isin(quiet_season)))
        .astype(np.int32)
        )

    # PU_APPT_MONTH or DL_APPT_MONTH is in produce season
    df["IS_PRODUCE_SEASON"] = (
        ((df["PU_APPT_MONTH"].isin(produce_season)) | 
            (df["DL_APPT_MONTH"].isin(produce_season)))
        .astype(np.int32)
        )

    # PU_APPT_MONTH or DL_APPT_MONTH is in peak season
    df["IS_PEAK_SEASON"] = (
        ((df["PU_APPT_MONTH"].isin(peak_season)) | 
            (df["DL_APPT_MONTH"].isin(peak_season)))
        .astype(np.int32)
        )

    # PU_APPT_MONTH or DL_APPT_MONTH is in holiday season
    df["IS_HOLIDAY_SEASON"] = (
        ((df["PU_APPT_MONTH"].isin(holiday_season)) | 
            (df["DL_APPT_MONTH"].isin(holiday_season)))
        .astype(np.int32)
        )

    return df

### function to calculate peak freight periods ---------------------------------------------------
def peak_freight_periods(df:pd.DataFrame=time_features):
    #  peak freight periods features
    years = np.arange(2018,2023,1)
    # christmas
    christmas_eve = pd.to_datetime(['2018-12-24', '2019-12-24','2020-12-24', '2021-12-24', '2022-12-24'])
    # New Year's Eve
    new_year_eve= pd.to_datetime(['2018-12-31', '2019-12-31','2020-12-31', '2021-12-31', '2022-12-31'])
    # Independence Day
    independence_day = pd.to_datetime(['2018-07-04', '2019-07-04','2020-07-04', '2021-07-04', '2022-07-04'])
    # Valentine's Day
    valentines_day = pd.to_datetime(['2018-02-14', '2019-02-14','2020-02-14', '2021-02-14', '2022-02-14'])
    # Halloween
    halloween = pd.to_datetime(['2018-10-31', '2019-10-31','2020-10-31', '2021-10-31', '2022-10-31']) 
    # Double 11, Single's Day
    double_11 = pd.to_datetime(['2018-11-11', '2019-11-11','2020-11-11', '2021-11-11', '2022-11-11']) 
    # Chinese Golden Week
    chinese_golden_week = pd.to_datetime(['2018-10-01', '2019-10-01','2020-10-01', '2021-10-01', '2022-10-01']) 
    # Easter
    easter = pd.to_datetime(['2018-04-01', '2019-04-21','2020-04-12', '2021-04-04', '2022-04-17'])
    # Black Friday
    black_friday = pd.to_datetime(['2018-11-23', '2019-11-29','2020-11-27','2021-11-26','2022-11-25'])
    # Cyber Monday
    cyber_monday = pd.to_datetime(['2018-11-26', '2019-12-02','2020-11-30','2021-11-29','2022-11-28'])
    # Super Bowl
    super_bowl = pd.to_datetime(['2018-02-04', '2019-02-03','2020-02-02','2021-02-07','2022-02-13'])

    events_list = [christmas_eve, new_year_eve, independence_day, valentines_day, 
                halloween, double_11, chinese_golden_week, easter, black_friday, cyber_monday,super_bowl]
    events_names = ['christmas_eve', 'new_year_eve', 'independence_day', 'valentines_day', 
                'halloween', 'double_11', 'chinese_golden_week', 'easter', 'black_friday', 'cyber_monday','super_bowl']
    
    for event, event_name in zip(events_list,events_names):
        df[event_name] = df['PU_APPT_YEAR'].map({year:d for year, d in zip(years,event)})
        # 7 days before event 
        df[f"_7_days_before_{event_name}"] = df[event_name] - np.timedelta64(7,'D')
        # 14 days before event 
        df[f"_14_days_before_{event_name}"] = df[event_name] - np.timedelta64(14,'D')

    for event_name in events_names:
        # create feature for 7 days to an event
        df[f"IS_7_DAYS_TO_{event_name.upper()}"] = (
            # `PU_APPT` within 7 days to event
            (((df['PU_APPT'].dt.floor('D') >= df[f'_7_days_before_{event_name}']) &
            (df['PU_APPT'].dt.floor('D') <= df[event_name])) |
            # or `DL_APPT` within 7 days to event
            ((df['DL_APPT'].dt.floor('D') >= df[f'_7_days_before_{event_name}']) &
            (df['DL_APPT'].dt.floor('D') <= df[event_name])))
            .astype(np.int32)
        )
        
        # create feature for 14 days to an event
        df[f"IS_14_DAYS_TO_{event_name.upper()}"] = (
            # `PU_APPT` within 14 days to event
            (((df['PU_APPT'].dt.floor('D') >= df[f'_14_days_before_{event_name}']) &
            (df['PU_APPT'].dt.floor('D') <= df[event_name])) |
            # or `DL_APPT` within 14 days to event
            ((df['DL_APPT'].dt.floor('D') >= df[f'_14_days_before_{event_name}']) &
            (df['DL_APPT'].dt.floor('D') <= df[event_name])))
            .astype(np.int32)
        )

    return df

### function to extract meteorological seasons ---------------------------------------------------
def meteorological_seasons(df:pd.DataFrame=time_features):
    # define spring
    spring = [3,4,5]
    # define summer
    summer = [6,7,8]
    # define fall
    fall = [9,10,11]
    # define winter
    winter = [12,1,2]

    # `PU_APPT_MONTH` or `DL_APPT_MONTH` is in spring
    df["IS_SPRING"] = (
        ((df["PU_APPT_MONTH"].isin(spring)) | 
        (df["DL_APPT_MONTH"].isin(spring)))
        .astype(np.int32)
        )

    # `PU_APPT_MONTH` or `DL_APPT_MONTH` is in summer
    df["IS_SUMMER"] = (
        ((df["PU_APPT_MONTH"].isin(summer)) | 
        (df["DL_APPT_MONTH"].isin(summer)))
        .astype(np.int32)
        )

    # `PU_APPT_MONTH` or `DL_APPT_MONTH` is in fall
    df["IS_FALL"] = (
        ((df["PU_APPT_MONTH"].isin(fall)) | 
        (df["DL_APPT_MONTH"].isin(fall)))
        .astype(np.int32)
        )

    # `PU_APPT_MONTH` or `DL_APPT_MONTH` is in winter
    df["IS_WINTER"] = (
        ((df["PU_APPT_MONTH"].isin(winter)) | 
        (df["DL_APPT_MONTH"].isin(winter)))
        .astype(np.int32)
        )
    return df

### Number of shipments and distance travelled by week and month ---------------------------------------------------
wkl_shipments = pd.read_csv('data/weekly_shipments_by_state.csv')
mhl_shipments = pd.read_csv('data/monthly_shipments_by_state.csv')
wkl_distance = pd.read_csv('data/weekly_distance_by_state.csv')
mhl_distance = pd.read_csv('data/monthly_distance_by_state.csv')

def periodical_attritubes(type:str,
                          year:int=2021,
                          country:str=st.session_state.origin_country,
                          state:str=st.session_state.origin_state):

    if type == 'shipments':
        # extract weekly shipment according to week, country, and state
        condition = (
            (wkl_shipments['PU_APPT_YEAR'] == year) &
            (wkl_shipments['PU_APPT_WEEK'] == time_features['PU_APPT_WEEK'].values[0]) &
            (wkl_shipments['ORIGIN_COUNTRY'] == country) &
            (wkl_shipments['ORIGIN_STATE'] == state)
            )
        WEEKLY_SHIPMENTS = wkl_shipments.loc[condition, 'WEEKLY_SHIPMENTS'].values[0]
        
        if not WEEKLY_SHIPMENTS:
            temp = wkl_shipments.groupby(['PU_APPT_YEAR', 'PU_APPT_MONTH', 'PU_APPT_WEEK'])['WEEKLY_SHIPMENTS'].mean().reset_index()
            WEEKLY_SHIPMENTS = temp.loc[
                (temp['PU_APPT_YEAR'] == 2021) & 
                (temp['PU_APPT_WEEK'] == time_features['PU_APPT_WEEK'].values[0]),
                'WEEKLY_SHIPMENTS'
                ].values[0]

        # extract monthly shipment according to month, country, and state
        condition = (
            (mhl_shipments['PU_APPT_YEAR'] == year) & 
            (mhl_shipments['PU_APPT_MONTH'] == time_features['PU_APPT_MONTH'].values[0]) &
            (mhl_shipments['ORIGIN_COUNTRY'] == country) &
            (mhl_shipments['ORIGIN_STATE'] == state)
            )
        MONTHLY_SHIPMENTS = mhl_shipments.loc[condition, 'MONTHLY_SHIPMENTS'].values[0]

        if not MONTHLY_SHIPMENTS:
            temp = mhl_shipments.groupby(['PU_APPT_YEAR', 'PU_APPT_MONTH'])['MONTHLY_SHIPMENTS'].mean().reset_index()
            MONTHLY_SHIPMENTS = temp.loc[
                (temp['PU_APPT_YEAR'] == 2021) &
                (temp['PU_APPT_MONTH'] == time_features['PU_APPT_MONTH'].values[0]),
                'MONTHLY_SHIPMENTS'
            ].values[0]

        return WEEKLY_SHIPMENTS, MONTHLY_SHIPMENTS

    if type == 'distance':
        # extract weekly distance travelled according to week, country, and state
        condition = (
            (wkl_distance['PU_APPT_YEAR'] == year) &
            (wkl_distance['PU_APPT_WEEK'] == time_features['PU_APPT_WEEK'].values[0]) &
            (wkl_distance['ORIGIN_COUNTRY'] == country) &
            (wkl_distance['ORIGIN_STATE'] == state)
            )
        WEEKLY_DISTANCE_TRAVELLED = wkl_distance.loc[condition, 'WEEKLY_DISTANCE_TRAVELLED'].values[0]

        if not WEEKLY_DISTANCE_TRAVELLED:
            temp = wkl_distance.groupby(['PU_APPT_YEAR', 'PU_APPT_MONTH', 'PU_APPT_WEEK'])['WEEKLY_DISTANCE_TRAVELLED'].mean().reset_index()
            WEEKLY_SHIPMENTS = temp.loc[
                (temp['PU_APPT_YEAR'] == 2021) & 
                (temp['PU_APPT_WEEK'] == time_features['PU_APPT_WEEK'].values[0]),
                'WEEKLY_DISTANCE_TRAVELLED'
                ].values[0]

        # extract monthly distance travelled according to month, country, and state
        condition = (
            (mhl_distance['PU_APPT_YEAR'] == year) & 
            (mhl_distance['PU_APPT_MONTH'] == time_features['PU_APPT_MONTH'].values[0]) &
            (mhl_distance['ORIGIN_COUNTRY'] == country) &
            (mhl_distance['ORIGIN_STATE'] == state)
            )
        MONTHLY_DISTANCE_TRAVELLED = mhl_distance.loc[condition, 'MONTHLY_DISTANCE_TRAVELLED'].values[0]

        if not MONTHLY_DISTANCE_TRAVELLED:
            temp = mhl_distance.groupby(['PU_APPT_YEAR', 'PU_APPT_MONTH'])['MONTHLY_DISTANCE_TRAVELLED'].mean().reset_index()
            MONTHLY_SHIPMENTS = temp.loc[
                (temp['PU_APPT_YEAR'] == 2021) &
                (temp['PU_APPT_MONTH'] == time_features['PU_APPT_MONTH'].values[0]),
                'MONTHLY_DISTANCE_TRAVELLED'
            ].values[0]

        return WEEKLY_DISTANCE_TRAVELLED, MONTHLY_DISTANCE_TRAVELLED

### Other features ---------------------------------------------------
EQUIPMENT_VOLUME_CAPACITY_CUFT = 4095.84
TRUCK_WEIGHT_CAPACITY_LBS = 45000
#NUM_TRUCK_USED = (1 if st.session_state.weight < TRUCK_WEIGHT_CAPACITY_LBS or 
#                        st.session_state.volume < EQUIPMENT_VOLUME_CAPACITY_CUFT
#                    else 0)
# WEIGHT_PER_CUFT = st.session_state.weight / st.session_state.volume

# gather categorical features
cat_features = pd.DataFrame({'ACTUAL_MODE':[st.session_state.actual_mode],
                             'ACTUAL_EQUIP':['53DV'], 
                             'ORIGIN_COUNTRY':[st.session_state.origin_country],
                             'DEST_COUNTRY':[st.session_state.dest_country],
                             'ORIGIN_CITY':[st.session_state.origin_city],
                             'ORIGIN_STATE':[st.session_state.origin_state],
                             'DEST_CITY':[st.session_state.dest_city],
                             'DEST_STATE':[st.session_state.dest_state]})

# NMFC Class
#nmfc = pd.read_csv('data/National Motor Freight Classification.csv', usecols=[0, 2, 3])
# create bins to classify data 
#cut_bins = list(nmfc['MIN_POUND_PER_CUFT'])[::-1] + [150.0]
#labels = list(nmfc['NMFC_CLASS'])[::-1]
#cat_features['NMFC_CLASS'] = pd.cut(cat_features['WEIGHT_PER_CUFT'], bins=cut_bins, labels=labels)
#cat_features = cat_features.drop(columns='WEIGHT_PER_CUFT')

### Encoding for categorical features ---------------------------------------------------
# read train data for preprocessing
@st.cache
def read_train_data():
    train = pd.read_parquet('data/train_lgbm_internal_v1.parquet').reset_index(drop=True)
    target = pd.read_csv('data/target_lgbm_internal_v1.csv')['TOTAL_ACTUAL_COST']
    return train, target
train, target = read_train_data()

def rare_encode(train:pd.DataFrame, test:pd.DataFrame, 
                rare_feature:list, tol=.05, n_categories=10, replace_with='Rare'):
    '''Perform rare label encoding for data set'''
    encoder = RareLabelEncoder(tol=tol, n_categories=n_categories,
                               variables=rare_feature, replace_with='Rare')
    encoder.fit(train)
    train = encoder.transform(train)
    test = encoder.transform(test)
    return train, test

def frequency_encode(train:pd.DataFrame, test:pd.DataFrame, vars:list, method='frequency'):
    '''Perform frequency encoding for data set'''
    encoder = CountFrequencyEncoder(encoding_method=method, variables=vars)
    encoder.fit(train)
    train = encoder.transform(train)
    test = encoder.transform(test)
    return train, test

def ordinal_encoder(X_train:pd.DataFrame, X_test:pd.DataFrame, 
                    y_train:pd.Series,
                    vars:list, method='ordered'):
    '''Perform ordinal encoding for data set'''
    encoder = OrdinalEncoder(encoding_method=method, variables=vars)
    encoder.fit(X_train, y_train)
    train = encoder.transform(X_train)
    test = encoder.transform(X_test)
    return train, test   

#### submit button ---------------------------------------------------
if st.button('Submit Order'):
    
    with st.spinner('Retrieving Result...'):
        progress_bar = st.progress(0.0)
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
        est_delivery_hours = est_delivery_day * 24

        # calculate estimated delivery speed
        try:
            est_delivery_speed = round(dis / ((dl_appt - pu_appt).total_seconds() / 3600), 4)
        except:
            est_delivery_speed = 0
            st.error('Please enter your shipment dates.')
            st.stop()

        for i in range(10):
            time.sleep(.02)
            pct = 0
            progress_bar.progress(pct + i/100)

        # create time features
        time_features = time_attritubes()
        time_features = peak_freight_periods()
        time_features = meteorological_seasons()
        time_features = time_features.drop(
            columns=['PU_APPT', 'DL_APPT', 'christmas_eve', 'new_year_eve',
                    'independence_day', 'valentines_day', 'halloween', 'double_11',
                    'chinese_golden_week', 'easter', 'black_friday', 'cyber_monday',
                    'super_bowl', '_7_days_before_christmas_eve',
                    '_14_days_before_christmas_eve', '_7_days_before_new_year_eve',
                    '_14_days_before_new_year_eve', '_7_days_before_independence_day',
                    '_14_days_before_independence_day',
                    '_7_days_before_valentines_day', '_14_days_before_valentines_day',
                    '_7_days_before_halloween', '_14_days_before_halloween',
                    '_7_days_before_double_11', '_14_days_before_double_11',
                    '_7_days_before_chinese_golden_week',
                    '_14_days_before_chinese_golden_week', '_7_days_before_easter',
                    '_14_days_before_easter', '_7_days_before_black_friday',
                    '_14_days_before_black_friday', '_7_days_before_cyber_monday',
                    '_14_days_before_cyber_monday', '_7_days_before_super_bowl',
                    '_14_days_before_super_bowl'])
        # convert time features to object
        for col in time_features.columns:
            time_features[col] = time_features[col].astype('object')
            train[col] = train[col].astype('object')

        for i in range(10):
            time.sleep(.02)
            pct = 0.1
            progress_bar.progress(pct + i/100)

        # gather numeric features
        num_features = pd.DataFrame({'DISTANCE':[dis],
                                     'CASES':[st.session_state.case],
                                     'WEIGHT':[st.session_state.weight],
                                     'VOLUME':[st.session_state.volume],
                                     'EST_DELIVERY_DAYS':[est_delivery_day],
                                     'EST_DELIVERY_HOURS':[est_delivery_hours],
                                     'EST_DELIVERY_SPEED_MPH':[est_delivery_speed],
                                     'ORIGIN_LAT':[origin_lat],
                                     'ORIGIN_LNG':[origin_lng],
                                     'DEST_LAT':[dest_lat],
                                     'DEST_LNG':[dest_lng]})

        # create periodical features
        num_features['WEEKLY_SHIPMENTS'], num_features['MONTHLY_SHIPMENTS'] = periodical_attritubes('shipments')
        num_features['WEEKLY_DISTANCE_TRAVELLED'], num_features['MONTHLY_DISTANCE_TRAVELLED'] = periodical_attritubes('distance')

        # gather weight and volume capacity
        num_features['EQUIPMENT_VOLUME_CAPACITY_CUFT'] = EQUIPMENT_VOLUME_CAPACITY_CUFT
        num_features['TRUCK_WEIGHT_CAPACITY_LBS'] = TRUCK_WEIGHT_CAPACITY_LBS

        for i in range(10):
            time.sleep(.02)
            pct = 0.2
            progress_bar.progress(pct + i/100)

        # OD PAIR
        cat_features['OD_PAIR'] = st.session_state.origin_zip[:3] + '-' + st.session_state.dest_zip[:3]
        
        # convert all categorical faetures to object
        for col in cat_features.columns:
            cat_features[col] = cat_features[col].astype('object')
            train[col] = train[col].astype('object')

        input = pd.concat([cat_features, time_features, num_features], axis=1)

        for i in range(10):
            time.sleep(.02)
            pct = 0.3
            progress_bar.progress(pct + i/100)

        # encoding
        # rare label encoding
        rare_features = ['ORIGIN_CITY', 'DEST_CITY', 'ORIGIN_STATE', 
                         'DEST_STATE', 'ACTUAL_EQUIP', 'OD_PAIR']
        train, input = rare_encode(train, input, rare_features)

        for i in range(10):
            time.sleep(.02)
            pct = 0.4
            progress_bar.progress(pct + i/50)

        # frequency encoding
        vars = ['ORIGIN_CITY', 'DEST_CITY']
        train, input = frequency_encode(train, input, vars)

        for i in range(10):
            time.sleep(.02)
            pct = 0.6
            progress_bar.progress(pct + i/50)

        # ordinal encoding (took 40 seconds to run)
        vars = ['ACTUAL_MODE', 'ACTUAL_EQUIP', 'ORIGIN_STATE', 'DEST_STATE', 
                'ORIGIN_COUNTRY', 'DEST_COUNTRY', 'OD_PAIR'] + list(time_features)
        train, input = ordinal_encoder(train, input, target, vars=vars)

        # check final input
        # st.dataframe(input)

        for i in range(10):
            time.sleep(.02)
            pct = 0.8
            progress_bar.progress(pct + i/50)
        progress_bar.progress(1.0)

        # predict
        fold_predictions = []
        for i in range(5):
            model = joblib.load('lightgbm_regressor/' + f'lgbm_fold{i}_seed42.pkl')
            pred = model.predict(input)
            fold_predictions.append(pred)
        predicted_cost = np.mean(fold_predictions)

    st.success('Prediction Completed!')
    st.markdown(f'### Estimated Total Cost: ${predicted_cost:.2f}')

