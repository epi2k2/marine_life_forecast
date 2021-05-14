from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from malif.preprocessing import MultilabelEncoder
from geopy.geocoders import Nominatim
import wikipedia


# Function that tranforms cyclic data to sine and cosine
def cycling_transform(value, function, steps):
    if function == 'sin':
        return np.sin(2*np.pi*value/steps)
    elif function == 'cos':
        return np.cos(2*np.pi*value/steps)


# Function that returns the name of locality, waterBody and countryCode
# assigned by the knn imputer
def knn_transform(data, le, knn, column):
    data_aux = data[['decimalLatitude', 'decimalLongitude']].copy()
    data_aux[column+'_aux'] = np.nan
    data_aux[column +
             '_aux'] = np.around(knn.transform(data_aux)[:, 2]).astype(int)
    return le.inverse_transform(data_aux[column+'_aux'])


# Function that gets the predictions
def predictions(lat, lon):
    float_cols = ['decimalLatitude', 'decimalLongitude', 'depth']
    int_cols = ['hour', 'day', 'month']

    data = pd.DataFrame(data=np.array(
        [[lat, lon, '', '', '', hour, day, month, depth]]),
        columns=['decimalLatitude', 'decimalLongitude', 'waterBody', 'locality', 'countryCode',
                 'hour', 'day', 'month', 'depth']
    )

    # Obtaining nearest locality, water body and country code
    data['locality'] = knn_transform(
        data, knn_malif[0][0], knn_malif[0][1], 'locality')
    data['waterBody'] = knn_transform(
        data, knn_malif[1][0], knn_malif[1][1], 'waterBody')
    data['countryCode'] = knn_transform(
        data, knn_malif[2][0], knn_malif[2][1], 'countryCode')

    # Casting columns to float
    for col in float_cols:
        data[col] = data[col].astype('float')

    # Casting columns to int
    for col in int_cols:
        data[col] = data[col].astype('int')

    # Array that will contain probabilities per target for every
    # depth between 0 and maximun depth selected
    probas_array = [[] for _ in range(len(targets))]

    # Predictions for every depth between 0 and depth selected
    for d in range(0, depth + 1):
        data['depth'] = d
        probas = clf_lr.predict_proba(data)
        for i in range(len(probas)):
            probas_array[i].append(probas[i][0][1])

    # Dicionary compose by targets and its maximum probability
    max_probas = {}
    for t, p in zip(targets, probas_array):
        max_probas[str(t)] = np.max(p)

    # Sorting probabilities from max to min
    probas_sorted = {}

    for key in sorted(max_probas, key=max_probas.get, reverse=True):
        probas_sorted[key] = max_probas[key]

    return probas_sorted


# Function that shows the data and probabilities of animals
def show_info(probabilities):
    containers = {}
    cols1 = {}
    cols2 = {}

    # Create a container per target that contains 2 columns, one for the image
    # and the other for the info
    for key in probabilities.keys():
        if probabilities[key] > 0.1:
            containers[key] = st.empty()
            cols1[key], cols2[key] = containers[key].beta_columns([
                                                                  3, 6])
            text = wikipedia.summary(key)
            cols1[key].write('')
            cols1[key].write('')
            cols1[key].image(wikipedia.page(key).images[0])
            cols2[key].subheader(
                key + ' (' + str(int(probabilities[key]*100)) + '%) ')
            cols2[key].write(text)


# Function that configures streamlit
def config(st):
    st.set_page_config(
        layout="wide",
        page_title='Malif'
    )

    # Ocultar menu
    hide_streamlit_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                </style>
                """

    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    st.markdown(
        "<style>.element-container{opacity:1 !important}</style>",
        unsafe_allow_html=True
    )


# Initial config
config(st)

# Initial position
location = None
lat = 42.2376602
lon = -8.7247205

# Model targets
targets = ['Moray eels',
           'Firefish',
           'Damselfish',
           'Sea turtle',
           'Grouper']

# Config for Nominatim (open street map)
geolocator = Nominatim(user_agent="Malif")

# Transformers and estimators folder
models_folder = './models/'
# Loading KNNImputer for locality, water body and country code
filename_knn = 'knn.sav'  # Saved KNNImputer
knn_malif = pickle.load(open(models_folder + filename_knn, 'rb'))

# Loading model
filename_clf = 'model.sav'  # Saved model
clf_lr = pickle.load(open(models_folder + filename_clf, 'rb'))

# Front end elements
# Map container
map_container = st.empty()

# Side bar elements
st.sidebar.title('Marine Life Forecasting')

locality = st.sidebar.text_input(
    label='Locality or diving spot'
)

date = st.sidebar.date_input('Select date', min_value=datetime.today())

day = date.day
month = int(date.month)
hour = st.sidebar.slider('Hour', 0, 23, 10)

depth = st.sidebar.slider(
    'Select max depth (m)',
    0, 40, 20)

search = st.sidebar.button('Search')

# Shows the address in the title if the search returns data
if locality != '':
    location = geolocator.geocode(locality)
    if location != None:
        lat = location.latitude
        lon = location.longitude

        map_expander = map_container.beta_expander(
            label='Marine life in ' + location.address, expanded=True)

    else:
        map_expander = map_container.beta_expander(
            label='Marine life', expanded=True)

else:
    map_expander = map_container.beta_expander(
        label='Marine life', expanded=True)


data_map = pd.DataFrame([[lat, lon]],
                        columns=['lat', 'lon']
                        )

map = map_expander.map(data_map, use_container_width=False)

# Prediction based on input variables. Also shows data from marine life predicted
if search:
    location = geolocator.geocode(locality)

    if location != None:
        map_container.empty()

        map_expander = map_container.beta_expander(
            label='Marine life in ' + location.address, expanded=True)

        map_expander.map(data_map, use_container_width=False)

        lat = location.latitude
        lon = location.longitude
        probabilities = predictions(lat, lon)

        show_info(probabilities)
