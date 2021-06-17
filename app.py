import streamlit as st
import pandas as pd
import numpy as np
import functions
import plotly.express as px

st.set_page_config(layout='wide')

st.write('## CIPW Normative Mineralogy Web App')
st.write(
    'This web app uses the implementation from Verma et al (date) to calculate normative \
    mineralogy from bulk geochemistry data.')

# Sidebar upload file
st.sidebar.write('## Data Upload')
st.sidebar.write('Upload your bulk geochemisty sample data below. Samples must contain the 11 major \
                 oxides as a minimum')


file = st.sidebar.file_uploader(' ', type=['.csv', '.xlsx'])

data = functions.load_data(file)

if file is not None:
    st.write(data)

st.sidebar.write('## Fe Correction Method')
fe_option = st.sidebar.selectbox('Fe Correction Method', ['Constant', 'La Maitre', 'Specified'])

if fe_option == 'Constant':
    fe_slider = st.sidebar.slider(label='Correction Factor', min_value=0.0, max_value=1.0, step=0.01)

elif fe_option == 'Specified':
    if file is not None:
        specified_ops = data.columns.tolist()
        chosen_col = st.sidebar.selectbox('Choose Column', specified_ops)

elif fe_option == 'La Maitre':
    rock_select = st.sidebar.radio(label='Igneous Type', options=['Plutonic', 'Volcanic'])
    # remove capitilisation
    if rock_select == 'Plutonic':
        rock_select = 'plutonic'
    elif rock_select == 'Volcanic':
        rock_select = 'volcanic'
    else:
        rock_select = None

st.sidebar.write('### Calculate')
cal_button = st.sidebar.empty()

if file is not None:
    if fe_option == 'Constant':
        adj_factor = fe_slider


    elif fe_option == 'Specified':
        corrected = functions.fe_correction(df=data, method='Constant', constant=data[chosen_col])
        data['FeO'] = corrected['FeO']
        data['Fe2O3'] = corrected['Fe2O3']
        adj_factor = 0


    elif fe_option == 'La Maitre':
        corrected = functions.fe_correction(df=data, method='La Maitre', ig_type=rock_select)
        data['FeO'] = corrected['FeO']
        data['Fe2O3'] = corrected['Fe2O3']
        adj_factor = 0

    if cal_button.button('Calculate Mineralogy'):
        norms = functions.CIPW_normative(data, Fe_adjustment_factor=adj_factor, majors_only=False)

        st.write(norms)

        st.markdown(functions.download_df(norms), unsafe_allow_html=True)