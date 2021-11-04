import streamlit as st
import functions

st.set_page_config(layout='wide')


def cipw():
    st.write('## CIPW Normative Mineralogy Web App')
    st.write("""
        This web app uses the implementation from Verma et al (2003) to calculate normative 
        mineralogy from bulk geochemistry data.
        Major oxides should be given in wt%, minor and trace elements in ppm
            """
             )

    # Sidebar upload file
    st.sidebar.write('## Data Upload')
    st.sidebar.write('Upload your bulk geochemisty sample data below. Samples must contain the 11 major \
                     oxides as a minimum')


    file = st.sidebar.file_uploader(' ', type=['.csv', '.xlsx'])

    data = functions.load_data(file)

    if file is not None:
        sum_threshold=90
        data['Sum'] = functions.major_sum(data)
        st.write(
            data.style.apply(
                functions.highlight_lessthan, threshold=sum_threshold, column='Sum', axis=1
            )
        )
        n_samples = functions.summation_warning(data, sum_threshold)
        if n_samples > 0:
            st.write("""
            **Warning!** {} samples dont sum up to more than {}%. \n
            *The highlighted cells show the problem samples.
            This may cause issues with the normative calculation* \n
            """.format(n_samples, sum_threshold)
            )



    st.sidebar.write('## Fe Correction Method')
    fe_option = st.sidebar.selectbox('Fe Correction Method', ['Constant', 'Le Maitre', 'Specified'])

    if fe_option == 'Constant':
        fe_slider = st.sidebar.slider(label='Correction Factor', min_value=0.0, max_value=1.0, step=0.01)

    elif fe_option == 'Specified':
        if file is not None:
            specified_ops = data.columns.tolist()
            chosen_col = st.sidebar.selectbox('Choose Column', specified_ops)

    elif fe_option == 'Le Maitre':
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


        elif fe_option == 'Le Maitre':
            corrected = functions.fe_correction(df=data, method='Le Maitre', ig_type=rock_select)
            data['FeO'] = corrected['FeO']
            data['Fe2O3'] = corrected['Fe2O3']
            adj_factor = 0

        if cal_button.button('Calculate Mineralogy'):
            norms = functions.CIPW_normative(data, Fe_adjustment_factor=adj_factor, majors_only=False, subdivide=True)

            st.write(norms.style.apply(functions.highlight_greaterthan, threshold=101, column='Sum', axis=1))
            if len(norms[norms['Sum'] > 101]):
                st.write('*Highlighted cells show where the normative sum is > 101.*')
            st.markdown(functions.download_df(norms), unsafe_allow_html=True)



    # Contact


    # Reference
    st.write('### References')
    st.write('''
    Le Maitre, R.W. Some problems of the projection of chemical data into mineralogical classifications
        *Contr. Mineral. and Petrol. 56, 181–189 (1976). https://doi.org/10.1007/BF00399603*
    
    Verma, S.P., Torres-Alvarado, I.S. & Velasco-Tapia, F., 2003. A revised CIPW norm.
    *Schweizerische Mineralogische und Petrographische Mitteilungen, 83(2), pp.197–216.*
    ''')


cipw()
