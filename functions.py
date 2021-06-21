import pandas as pd
import numpy as np
import streamlit as st
import base64

oxides = ['SiO2', 'TiO2', 'Al2O3', 'Fe2O3', 'FeO', 'MnO', 'MgO', 'CaO',
              'Na2O', 'K2O', 'P2O5']

def major_sum(data):
    return data[oxides].sum(axis=1)

def summation_warning(data, threshold):
    return len(data[data[oxides].sum(axis=1) < threshold])


def load_data(file):
    if file is not None:
        if 'csv' in file.name:
            data = pd.read_csv(file)
        elif 'xlsx' in file.name:
            data = pd.read_excel(file)
        else:
            data = None
    else:
        data = None
    return data

@st.cache
def CIPW_normative(df, Fe_adjustment_factor, majors_only=True, subdivide=False):
    """
    Calculates mineralogy from bulk geochemistry
    -----------
    df : :class:`pandas.DataFrame`
        Dataframe with 11 major oxides in weight percentage, and trace/minor
        elements in ppm
    Returns
    --------
    :class:`pandas.DataFrame`
    """
    df = df.copy(deep=True)

    oxide_molecular_weights = {
        'SiO2': 60.0843,
        'TiO2': 79.8658,
        'Al2O3': 101.961276,
        'Fe2O3': 159.6882,
        'FeO': 71.8444,
        'MnO': 70.937449,
        'MgO': 40.3044,
        'CaO': 56.0774,
        'Na2O': 61.97894,
        'K2O': 94.1960,
        'P2O5': 141.944522,
        'CO2': 44.0095,
        'SO3': 80.0582,
        'F': 18.9984,
        'Cl': 35.453,
        'S': 32.06,
        'NiO': 74.6928,
        'CoO': 74.9326,
        'SrO': 103.6194,
        'BaO': 153.3294,
        'Rb2O': 186.935,
        'Cs2O': 281.8103,
        'Li2O': 29.8814,
        'ZrO2': 123.2188,
        'Cr2O3': 151.9902,
        'V2O3': 149.8812
    }

    mineral_codes = {
        'Q': 'Quartz',
        'Z': 'Zircon',
        'Ks': 'K2SiO3',
        'An': 'Anorthite',
        'Ns': 'Na2SiO3',
        'Ac': 'Acmite',
        'Di': 'Diopside',
        'Fe-Di': 'Clinoferrosilite',
        'Mg-Di': 'Clinoentatite',
        'Tn': 'Sphene',
        'Hy': 'Hypersthene',
        'Fe-Hy': 'Ferrosilite',
        'Mg-Hy': 'Enstatite',
        'Ab': 'Albite',
        'Or': 'Orthoclase',
        'Pl': 'Plagioclase',
        'Wo': 'Wollastonite',
        'Ol': 'Olivine',
        'Fe-Ol': 'Fayalite',
        'Mg-Ol': 'Forsterite',
        'Pf': 'Perovskite',
        'Ne': 'Nepheline',
        'Lc': 'Leucite',
        'Cs': 'Larnite',
        'Kp': 'Kalsilite',
        'Ap': 'Apatite',
        'Fr': 'Fluorite',
        'Pr': 'Pyrite',
        'Cm': 'Chromite',
        'Il': 'Ilmenite',
        'Cc': 'Calcite',
        'C': 'Corundum',
        'Ru': 'Rutile',
        'Mt': 'Magnetite',
        'Hm': 'Hematite'
    }

    mineral_molecular_weights = {
        'Q': 60.0843,
        'Z': 183.3031,
        'Ks': 154.2803,
        'An': 278.207276,
        'Ns': 122.0632,
        'Ac': 462.0083,
        'Di': 225.99234699428553,
        'Tn': 196.0625,
        'Hy': 109.82864699428553,
        'Ab': 524.446,
        'Or': 556.6631,
        'Wo': 116.1637,
        'Ol': 159.57299398857106,
        'Pf': 135.9782,
        'Ne': 284.1088,
        'Lc': 436.4945,
        'Cs': 172.2431,
        'Kp': 316.3259,
        'Ap': 328.8691887,
        'Fr': 94.0762,
        'Pr': 135.96640000000002,
        'Cm': 223.83659999999998,
        'Il': 151.7452,
        'Cc': 100.0892,
        'C': 101.9613,
        'Ru': 79.8988,
        'Mt': 231.53860000000003,
        'Hm': 159.6922,
        'Mg-Di': 216.5504,
        'Mg-Hy': 100.3887,
        'Mg-Ol': 140.6931
    }

    element_AW = {
        'F': 18.9984032,
        'Cl': 35.4527,
        'S': 32.066,
        'Ni': 58.6934,
        'Co': 58.93320,
        'Sr': 87.62,
        'Ba': 137.327,
        'Rb': 85.4678,
        'Cs': 132.90545,
        'Li': 6.941,
        'Zr': 91.224,
        'Cr': 51.9961,
        'V': 50.9415,
        'O': 15.9994
    }

    element_oxide = {
        'F': 'F',
        'Cl': 'Cl',
        'S': 'S',
        'Ni': 'NiO',
        'Co': 'CoO',
        'Sr': 'SrO',
        'Ba': 'BaO',
        'Rb': 'Rb2O',
        'Cs': 'Cs2O',
        'Li': 'Li2O',
        'Zr': 'ZrO2',
        'Cr': 'Cr2O3',
        'V': 'V2O3'
    }

    print(df[['Cr2O3', 'Cr']])

    oxides = ['SiO2', 'TiO2', 'Al2O3', 'Fe2O3', 'FeO', 'MnO', 'MgO', 'CaO',
              'Na2O', 'K2O', 'P2O5']

    trace_oxides = ['CO2', 'SO3', 'F', 'Cl', 'S', 'NiO', 'CoO',
                    'SrO', 'BaO', 'Rb2O', 'Cs2O', 'Li2O', 'ZrO2', 'Cr2O3', 'V2O3']

    # Check for major and trace elements. Add missing columns filled with 0 value.
    # force numeric type

    for oxide in oxides:
        if oxide in df.columns.tolist():
            df[oxide] = pd.to_numeric(df[oxide], errors='coerce')
            continue
        else:
            df[oxide] = 0

    for trace in trace_oxides:
        if trace in df.columns.tolist():
            df[trace] = pd.to_numeric(df[trace], errors='coerce')
            continue
        else:
            df[oxide] = 0

    for element in element_oxide.keys():
        if element in df.columns.tolist():
            df[element] = pd.to_numeric(df[element], errors='coerce')
            continue
        else:
            df[element] = 0

    # replace str values from df with 0
    def unique_strings(df, col):
        return df[col][df[col].map(type) == str].unique().tolist()

    # replace nans with 0

    df.fillna(0, inplace=True)

    for col in df.columns.tolist():
        unique_str = (unique_strings(df, col))
        df[col].replace(unique_str, 0, inplace=True)

    # conversion of trace elements from ppm to oxide % m/m


    def calculate_oxide_1(oxide_weight, element_AW, ppm):
        return (oxide_weight/element_AW) * ppm * (10**-4)


    def calculate_oxide_2(oxide_weight, element_AW, ppm):
        return (oxide_weight/(2 * element_AW)) * ppm * (10**-4)


    trace_elements_present = list(set(element_AW).intersection(df.columns.tolist()))

    print(df[['Cr2O3', 'Cr']])

    for element in trace_elements_present:
        if element in ['Ni', 'Co', 'Sr', 'Ba', 'Zr']:
            oxide_name = element_oxide[element]
            df[oxide_name] = calculate_oxide_1(
                                         oxide_molecular_weights[oxide_name],
                                         element_AW[element],
                                         df[element])
        elif element in ['S', 'Cl', 'F']:
            df[element] = df[element] * (10**-4)
        else:
            oxide_name = element_oxide[element]
            df[oxide_name] = calculate_oxide_2(
                                         oxide_molecular_weights[oxide_name],
                                         element_AW[element],
                                         df[element])

    print(df[['Cr2O3', 'Cr']])

    trace_oxides_present = list(set(trace_oxides).intersection(df.columns.tolist()))

    # where minor/trace oxide is missing, add column with 0 value
    for oxide in trace_oxides:
        if oxide not in trace_oxides_present:
            df[oxide] = 0

    print(df[['Cr2O3', 'Cr']])

    # Adjustment of Fe-oxidation ratio and 100% sum as well as computation of
    # some petrogenetically useful parameters
    if Fe_adjustment_factor > 0:
        df['total_Fe_as_FeO'] = (df['Fe2O3']/1.11134) + df['FeO']

        df['adjusted_Fe2O3'] = df['total_Fe_as_FeO'] * Fe_adjustment_factor * 1.11134
        df['adjusted_FeO'] = df['total_Fe_as_FeO'] * (1-Fe_adjustment_factor)

        df['Fe2O3'] = df['adjusted_Fe2O3']
        df['FeO'] = df['adjusted_FeO']

    df['intial_sum'] = df[oxides].sum(axis=1)
    df['adjustment_1'] = 100 / df['intial_sum']
    df[oxides] = df[oxides].mul(df['adjustment_1'], axis=0)



    if not majors_only:
        df['major_minor_sum'] = df[oxides].sum(axis=1) + df[trace_oxides_present].sum(axis=1)
        df['adjustment_2'] = 100 / df['major_minor_sum']
        df[oxides + trace_oxides_present] = df[oxides + trace_oxides_present].mul(df['adjustment_2'], axis=0)

    df[oxides + trace_oxides_present] = df[oxides + trace_oxides_present] .round(3)

    corrected_oxides = df[oxides + trace_oxides_present].copy()

    df['Feo/MgO'] = ((2*oxide_molecular_weights['FeO']/oxide_molecular_weights['MgO']) *
    df['Fe2O3'] + df['FeO'] / df['MgO'])

    df['SI'] = 100 * df['MgO'] / (df['MgO'] + df['FeO'] + df['Fe2O3'] + df['Na2O'] + df['K2O'])

    df['AR_True'] = (df['Al2O3'] + df['CaO'] + df['Na2O'] + df['K2O']) / (df['Al2O3'] + df['CaO'] - df['Na2O'] - df['K2O'])
    df['AR_False'] = (df['Al2O3'] + df['CaO'] + 2*df['Na2O']) / (df['Al2O3'] + df['CaO'] - 2*df['Na2O'])

    df['AR'] = np.where(((df['K2O']/df['Na2O']) >= 1) & ((df['K2O']/df['Na2O']) <= 2.5) & (df['SiO2'] > 0.5), df['AR_True'], df['AR_False'])

    df['Mg#'] = 100*df['MgO']/(df['MgO'] + df['FeO'])

    df

    if not majors_only:
        oxides = oxides + trace_oxides

    # Mole Calculations
    for oxide in oxides:
        df['n_' + oxide] = df[oxide]/oxide_molecular_weights[oxide]

    if not majors_only:
        for oxide in trace_oxides:
            df['n_' + oxide] = df[oxide]/oxide_molecular_weights[oxide]


    # Minor oxide combinations

    if majors_only:
        df['n_FeO_corr'] = df['n_FeO'] + df['n_MnO']
    else:
        df['n_FeO_corr'] = df['n_FeO'] + df['n_MnO'] + df['n_NiO'] + df['n_CoO']
        df['n_CaO_corr'] = df['n_CaO'] + df['n_SrO'] + df['n_BaO']
        df['n_K2O_corr'] = df['n_K2O'] + df['n_Rb2O'] + df['n_Cs2O']
        df['n_Na2O_corr'] = df['n_Na2O'] + df['n_Li2O']
        df['n_Cr2O3_corr'] = df['n_Cr2O3'] + df['n_V2O3']


    # Corrected oxide molecular weight computations
    df['x_MnO'] = df['n_MnO'] / df['n_FeO_corr']
    df['x_FeO'] = df['n_FeO'] / df['n_FeO_corr']

    if not majors_only:
        df['x_NiO'] = df['n_NiO'] / df['n_FeO_corr']
        df['x_CoO'] = df['n_CoO'] / df['n_FeO_corr']

        df['x_SrO'] = df['n_SrO'] / df['n_CaO_corr']
        df['x_BaO'] = df['n_BaO'] / df['n_CaO_corr']
        df['x_CaO'] = df['n_CaO'] / df['n_CaO_corr']


        df['x_Rb2O'] = df['n_Rb2O'] / df['n_K2O_corr']
        df['x_Cs2O'] = df['n_Cs2O'] / df['n_K2O_corr']
        df['x_K2O'] = df['n_K2O'] / df['n_K2O_corr']

        df['x_Li2O'] = df['n_Li2O'] / df['n_Na2O_corr']
        df['x_Na2O'] = df['n_Na2O'] / df['n_Na2O_corr']

        df['x_V2O3'] = df['n_V2O3'] / df['n_Cr2O3_corr']
        df['x_Cr2O3'] = df['n_Cr2O3'] / df['n_Cr2O3_corr']


    if majors_only:
        df['n_FeO'] = df['n_FeO_corr']
    else:
        df['n_FeO'] = df['n_FeO_corr']
        df['n_CaO'] = df['n_CaO_corr']
        df['n_K2O'] = df['n_K2O_corr']
        df['n_Na2O'] = df['n_Na2O_corr']
        df['n_Cr2O3'] = df['n_Cr2O3_corr']


    # Corrected normative mineral molecular weight computations

    def corr_m_wt(oxide):
        return(df['x_'+ oxide] * oxide_molecular_weights[oxide])


    if majors_only:
        df['MW_FeO_corr'] = corr_m_wt('MnO') + corr_m_wt('FeO')
    else:
        df['MW_FeO_corr'] = corr_m_wt('MnO') + corr_m_wt('NiO') + corr_m_wt('CoO') + corr_m_wt('FeO')
        df['MW_CaO_corr'] = corr_m_wt('BaO') + corr_m_wt('SrO') + corr_m_wt('CaO')
        df['MW_K2O_corr'] = corr_m_wt('Rb2O') + corr_m_wt('Cs2O') + corr_m_wt('K2O')
        df['MW_Na2O_corr'] = corr_m_wt('Li2O') + corr_m_wt('Na2O')
        df['MW_Cr2O3_corr'] = corr_m_wt('V2O3') + corr_m_wt('Cr2O3')

        # Corrected molecular weight of Ca, Na and Fe
        df['MW_Ca_corr'] = df['MW_CaO_corr'] - element_AW['O']
        df['MW_Na_corr'] = (df['MW_Na2O_corr'] - element_AW['O'])/2
        df['MW_Fe_corr'] = df['MW_FeO_corr'] - element_AW['O']




    if majors_only:
        mineral_molecular_weights['Fe-Hy'] = df['MW_FeO_corr'] + 60.0843
        mineral_molecular_weights['Fe-Ol'] = (2 * df['MW_FeO_corr']) + 60.0843
        mineral_molecular_weights['Fe-Di'] = df['MW_FeO_corr'] + 176.2460
        mineral_molecular_weights['Mt'] = df['MW_FeO_corr'] + 159.6882
        mineral_molecular_weights['Il'] = df['MW_FeO_corr'] + 79.8658

    else:
        mineral_molecular_weights['Fe-Hy'] = df['MW_FeO_corr'] + 60.0843
        mineral_molecular_weights['Fe-Ol'] = (2 * df['MW_FeO_corr']) + 60.0843
        mineral_molecular_weights['Mt'] = df['MW_FeO_corr'] + 159.6882
        mineral_molecular_weights['Il'] = df['MW_FeO_corr'] + 79.8658
        mineral_molecular_weights['An'] = df['MW_CaO_corr'] + 222.129876
        mineral_molecular_weights['Mg-Di'] = df['MW_CaO_corr'] + 160.4730
        mineral_molecular_weights['Wo'] = df['MW_CaO_corr'] + 60.0843
        mineral_molecular_weights['Cs'] = 2*df['MW_CaO_corr'] + 60.0843
        mineral_molecular_weights['Tn'] = df['MW_CaO_corr'] + 139.9501
        mineral_molecular_weights['Pf'] = df['MW_CaO_corr'] + 79.8558
        mineral_molecular_weights['CaF2-Ap'] = 3*df['MW_CaO_corr'] + (1/3)*df['MW_Ca_corr'] + 154.6101241
        mineral_molecular_weights['CaO-Ap'] = (10/3)*df['MW_CaO_corr'] + 141.944522
        mineral_molecular_weights['Cc'] = df['MW_CaO_corr'] + 44.0095
        mineral_molecular_weights['Ab'] = df['MW_Na2O_corr'] + 462.467076
        mineral_molecular_weights['Ne'] = df['MW_Na2O_corr'] + 222.129876
        mineral_molecular_weights['Th'] = df['MW_Na2O_corr'] + 80.0642
        mineral_molecular_weights['Nc'] = df['MW_Na2O_corr'] + 44.0095
        mineral_molecular_weights['Ac'] = df['MW_Na2O_corr'] + 400.0254
        mineral_molecular_weights['Ns'] = df['MW_Na2O_corr'] + 60.0843
        mineral_molecular_weights['Or'] = df['MW_K2O_corr'] + 462.467076
        mineral_molecular_weights['Lc'] = df['MW_K2O_corr'] + 342.298476
        mineral_molecular_weights['Kp'] = df['MW_K2O_corr'] + 222.129876
        mineral_molecular_weights['Ks'] = df['MW_K2O_corr'] + 60.0843
        mineral_molecular_weights['Fe-Di'] = df['MW_FeO_corr'] + df['MW_CaO_corr'] + 120.1686
        mineral_molecular_weights['Cm'] = df['MW_FeO_corr'] + df['MW_Cr2O3_corr']
        mineral_molecular_weights['Hl'] = df['MW_Na_corr'] + 35.4527
        mineral_molecular_weights['Fr'] = df['MW_Ca_corr'] + 37.9968064
        mineral_molecular_weights['Pr'] = df['MW_Fe_corr'] + 64.132

    df['Y'] = 0


    if not majors_only:
        # Normative zircon
        if 'ZrO2' in trace_oxides_present:
            df['Z'] = df['n_ZrO2']

            df['Y'] = df['Z']


    # Normative apatite

    df['n_P2O5']

    df['Ap'] = np.where(
        df['n_CaO'] >= (3+1/3) * df['n_P2O5'], df['n_P2O5'], df['n_CaO']/(3+1/3)
        ).T

    df['n_CaO_'] = np.where(
        df['n_CaO'] >= (3+1/3) * df['n_P2O5'], df['n_CaO'] - (3+1/3) * df['Ap'], 0
        ).T

    df['n_P2O5_'] = np.where(
        df['n_CaO'] < (3+1/3) * df['n_P2O5'], df['n_P2O5'] - df['Ap'], 0).T

    df['n_CaO'] = df['n_CaO_']
    df['n_P2O5'] = df['n_P2O5_']

    df['FREE_P2O5'] = df['n_P2O5']

    if not majors_only:
        # apatite options where F in present
        if 'F' in trace_oxides_present:
            df['ap_option'] = np.where(
                df['n_F'] >= (2/3) * df['Ap'], 2, 3).T

            df['n_F'] = np.where(
                df['ap_option'] == 2, df['n_F'] - (2/3 * df['Ap']), df['n_F']).T

            df['CaF2-Ap'] = np.where(
                df['ap_option'] == 3, df['n_F'] * 1.5, 0).T

            df['CaO-Ap'] = np.where(
                df['ap_option'] == 3, df['n_P2O5'] - (1.5 * df['n_F']), 0).T

            df['Ap'] = np.where(
                df['ap_option'] == 3, df['CaF2-Ap'] + df['CaO-Ap'], df['Ap']).T

            df['FREEO_12b'] = np.where(df['ap_option'] == 2, 1/3 * df['Ap'], 0).T
            df['FREEO_12c'] = np.where(df['ap_option'] == 3, df['n_F']/2, 0).T
        else:
            df['FREEO_12b'] = 0
            df['FREEO_12c'] = 0
            df['CaO-Ap'] = 0
            df['CaF2-Ap'] = 0

        # Normative Fluorite
        if 'F' in trace_oxides_present:
            df['Fr'] = np.where(df['n_CaO'] >= df['n_F']/2, df['n_F']/2, df['n_CaO']).T

            df['n_CaO'] = np.where(
                df['n_CaO'] >= df['n_F']/2, df['n_CaO'] - df['Fr'], 0).T

            df['n_F'] = np.where(
                df['n_CaO'] >= df['n_F']/2, df['n_F'], df['n_F'] - (2*df['Fr'])).T

            df['FREEO_13'] = df['Fr']
            df['FREE_F'] = df['n_F']
        else:
            df['FREEO_13'] = 0
            df['FREE_F'] = 0


        # Normative halite
        if 'Cl' in trace_oxides_present:
            df['Hl'] = np.where(
                df['n_Na2O'] >= 2*df['n_Cl'], df['n_Cl'], df['n_Na2O']/2).T

            df['n_Na2O'] = np.where(
                df['n_Na2O'] >= 2*df['n_Cl'], df['n_Na2O']-df['Hl']/2, 0).T

            df['n_Cl'] = np.where(
                df['n_Na2O'] >= 2*df['n_Cl'], df['n_Cl'], df['n_Cl'] - df['Hl']).T

            df['FREE_Cl'] = df['n_Cl']
            df['FREEO_14'] = df['Hl']/2
        else:
            df['FREE_Cl'] = 0
            df['FREEO_14'] = 0


        # Normative thenardite

        if 'SO3' in trace_oxides_present:
            df['Th'] = np.where(df['n_Na2O'] >= df['SO3'], df['SO3'], df['n_Na2O']).T

            df['n_Na2O'] = np.where(df['n_Na2O'] >= df['SO3'], df['n_Na2O'] - df['Th'], 0).T

            df['n_SO3'] = np.where(df['n_Na2O'] >= df['SO3'], df['n_SO3'], df['n_SO3'] - df['Th']).T

            df['FREE_SO3'] = df['n_SO3']

        else:
            df['FREE_SO3'] = 0

        # Normative Pyrite

        if 'S' in trace_oxides_present:
            df['Pr'] = np.where(df['n_FeO'] >= 2*df['n_S'], df['n_S']/2, df['n_FeO']).T

            df['n_FeO'] = np.where(df['n_FeO'] >= 2*df['n_S'], df['n_FeO'] - df['Pr'], df['n_FeO'] - df['Pr']*2).T

            df['FREE_S'] = np.where(df['n_FeO'] >= 2*df['n_S'], 0, df['n_FeO']).T

            df['n_FeO'] = df['n_FeO'] - df['FREE_S']

            df['FREEO_16'] = df['Pr']

        else:
            df['FREEO_16'] = 0
            df['FREE_S'] = 0


        # Normative sodium carbonate or calcite
        if 'CO2' in trace_oxides_present:
            df['Nc'] = np.where(df['n_Na2O'] >= df['n_CO2'], df['n_CO2'], df['n_Na2O']).T

            df['n_Na2O'] = np.where(df['n_Na2O'] >= df['n_CO2'], df['n_Na2O'] - df['Nc'], df['n_Na2O']).T

            df['n_CO2'] = np.where(df['n_Na2O'] >= df['n_CO2'], df['n_CO2'], df['n_CO2'] - df['Nc']).T

            df['Cc'] = np.where(df['n_CaO'] >= df['n_CO2'], df['n_CO2'], df['n_CaO']).T

            df['CaO'] = np.where(df['n_Na2O'] >= df['n_CO2'], df['CaO'] - df['Cc'], df['CaO']).T

            df['CO2'] = np.where(df['n_Na2O'] >= df['n_CO2'], df['n_CO2'], df['n_CO2'] - df['Cc']).T

            df['FREECO2'] = df['n_CO2']
        else:
            df['FREECO2'] = 0

        # Normative Chromite
        if 'Cr2O3' in trace_oxides_present:
            df['Cm'] = np.where(df['n_FeO'] >= df['n_Cr2O3'], df['n_Cr2O3'], df['n_FeO']).T

            df['n_FeO'] = np.where(df['n_FeO'] >= df['n_Cr2O3'], df['n_FeO'] - df['Cm'], 0).T
            df['n_Cr2O3'] = np.where(df['n_FeO'] >= df['n_Cr2O3'], df['n_Cr2O3'] - df['Cm'], df['n_Cr2O3']).T

            df['FREE_CR2O3'] = df['Cm']
        else:
            df['FREE_CR2O3'] = 0

    # Normative Ilmenite
    df['Il'] = np.where(df['n_FeO'] >= df['n_TiO2'], df['n_TiO2'], df['n_FeO']).T

    df['n_FeO_'] = np.where(df['n_FeO'] >= df['n_TiO2'], df['n_FeO'] - df['Il'], 0).T

    df['n_TiO2_'] = np.where(df['n_FeO'] >= df['n_TiO2'], 0, df['n_TiO2'] - df['Il']).T

    df['n_FeO'] = df['n_FeO_']

    df['n_TiO2'] = df['n_TiO2_']


    # Normative Orthoclase/potasium metasilicate

    df['Or_p'] = np.where(df['n_Al2O3'] >= df['n_K2O'], df['n_K2O'], df['n_Al2O3']).T

    df['n_Al2O3_'] = np.where(df['n_Al2O3'] >= df['n_K2O'], df['n_Al2O3'] - df['Or_p'], 0).T

    df['n_K2O_'] = np.where(df['n_Al2O3'] >= df['n_K2O'], 0, df['n_K2O'] - df['Or_p']).T

    df['Ks'] = df['n_K2O_']

    df['Y'] = np.where(df['n_Al2O3'] >= df['n_K2O'], df['Y'] +(df['Or_p']*6), df['Y'] + (df['Or_p']*6 + df['Ks'])).T

    df['n_Al2O3'] = df['n_Al2O3_']
    df['n_K2O'] = df['n_K2O_']


    # Normative Albite
    df['Ab_p'] = np.where(df['n_Al2O3'] >= df['n_Na2O'], df['n_Na2O'], df['n_Al2O3']).T

    df['n_Al2O3_'] = np.where(df['n_Al2O3'] >= df['n_Na2O'], df['n_Al2O3'] - df['Ab_p'], 0).T

    df['n_Na2O_'] = np.where(df['n_Al2O3'] >= df['n_Na2O'], 0, df['n_Na2O'] - df['Ab_p']).T

    df['Y'] = df['Y'] + (df['Ab_p'] * 6)

    df['n_Al2O3'] = df['n_Al2O3_']
    df['n_Na2O'] = df['n_Na2O_']


    # Normative Acmite / sodium metasilicate
    df['Ac'] = np.where(df['n_Na2O'] >= df['n_Fe2O3'], df['n_Fe2O3'], df['n_Na2O']).T

    df['n_Na2O_'] = np.where(df['n_Na2O'] >= df['n_Fe2O3'], df['n_Na2O'] - df['Ac'], 0).T

    df['n_Fe2O3_'] = np.where(df['n_Na2O'] >= df['n_Fe2O3'], 0, df['n_Fe2O3'] - df['Ac']).T

    df['Ns'] = df['n_Na2O_']

    df['Y'] = np.where(df['n_Na2O'] >= df['n_Fe2O3'], df['Y'] + (4*df['Ac'] + df['Ns']), df['Y'] + 4*df['Ac']).T

    df['n_Na2O'] = df['n_Na2O_']
    df['n_Fe2O3'] = df['n_Fe2O3_']


    # Normative Anorthite / Corundum
    df['An'] = np.where(df['n_Al2O3'] >= df['n_CaO'], df['n_CaO'], df['n_Al2O3']).T

    df['n_Al2O3_'] = np.where(df['n_Al2O3'] >= df['n_CaO'], df['n_Al2O3'] - df['An'], 0).T

    df['n_CaO_'] = np.where(df['n_Al2O3'] >= df['n_CaO'], 0, df['n_CaO'] - df['An']).T

    df['C'] = df['n_Al2O3_']

    df['n_Al2O3'] = df['n_Al2O3_']

    df['n_CaO'] = df['n_CaO_']

    df['Y'] = df['Y'] + 2*df['An']


    # Normative Sphene / Rutile
    df['Tn_p'] = np.where(df['n_CaO'] >= df['n_TiO2'], df['n_TiO2'], df['n_CaO']).T

    df['n_CaO_'] = np.where(df['n_CaO'] >= df['n_TiO2'], df['n_CaO'] - df['Tn_p'], 0).T

    df['n_TiO2_'] = np.where(df['n_CaO'] >= df['n_TiO2'], 0, df['n_TiO2'] - df['Tn_p']).T

    df['n_CaO'] = df['n_CaO_']
    df['n_TiO2'] = df['n_TiO2_']

    df['Ru'] = df['n_TiO2']

    df['Y'] = df['Y'] + df['Tn_p']


    # Normative Magnetite / Hematite
    df['Mt'] = np.where(df['n_Fe2O3'] >= df['n_FeO'], df['n_FeO'], df['n_Fe2O3']).T

    df['n_Fe2O3_'] = np.where(df['n_Fe2O3'] >= df['n_FeO'], df['n_Fe2O3'] - df['Mt'], 0).T

    df['n_FeO_'] = np.where(df['n_Fe2O3'] >= df['n_FeO'], 0, df['n_FeO'] - df['Mt']).T

    df['n_Fe2O3'] = df['n_Fe2O3_']

    df['n_FeO'] = df['n_FeO_']

    df['Hm'] = df['n_Fe2O3']


    # Subdivision of some normative minerals
    df['n_MgFe_O'] = df['n_MgO'] + df['n_FeO']

    df['MgO_ratio'] = df['n_MgO']/df['n_MgFe_O']
    df['FeO_ratio'] = df['n_FeO']/df['n_MgFe_O']


    # Provisional normative dioside, wollastonite / Hypersthene
    df['Di_p'] = np.where(df['n_CaO'] >= df['n_MgFe_O'], df['n_MgFe_O'], df['n_CaO']).T

    df['n_CaO_']  = np.where(df['n_CaO'] >= df['n_MgFe_O'], df['n_CaO'] - df['Di_p'], 0).T

    df['n_MgFe_O_']  = np.where(df['n_CaO'] >= df['n_MgFe_O'], 0, df['n_MgFe_O'] - df['Di_p']).T

    df['Hy_p']  = df['n_MgFe_O_']

    df['Wo_p']  = np.where(df['n_CaO'] >= df['n_MgFe_O'], df['n_CaO_'], 0).T

    df['Y']  = np.where(df['n_CaO'] >= df['n_MgFe_O'], df['Y'] + (2*df['Di_p'] + df['Wo_p']), df['Y'] + (2*df['Di_p'] + df['Hy_p'])).T

    df['n_CaO'] = df['n_CaO_']
    df['n_MgFe_O'] = df['n_MgFe_O_']


    # Normative quartz / undersaturated minerals
    df['Q'] = np.where(df['n_SiO2'] >= df['Y'], df['n_SiO2'] - df['Y'], 0).T

    df['D'] = np.where(df['n_SiO2'] < df['Y'], df['Y'] - df['n_SiO2'], 0).T

    df['deficit'] = df['D'] > 0

    # Normative Olivine / Hypersthene
    df['Ol_'] = np.where((df['D'] < df['Hy_p']/2), df['D'], df['Hy_p']/2).T

    df['Hy'] = np.where((df['D'] < df['Hy_p']/2), df['Hy_p'] - 2*df['D'], 0).T

    df['D1'] = df['D'] - df['Hy_p']/2

    df['Ol'] = np.where((df['deficit']), df['Ol_'], 0).T

    df['Hy'] = np.where((df['deficit']), df['Hy'], df['Hy_p']).T

    df['deficit'] = df['D1'] > 0


    # Normative Sphene / Perovskite
    df['Tn'] = np.where((df['D1'] < df['Tn_p']), df['Tn_p'] - df['D1'], 0).T

    df['Pf_'] = np.where((df['D1'] < df['Tn_p']), df['D1'], df['Tn_p']).T

    df['D2'] = df['D1'] - df['Tn_p']

    df['Tn'] = np.where((df['deficit']), df['Tn'], df['Tn_p']).T
    df['Pf'] = np.where((df['deficit']), df['Pf_'], 0).T

    df['deficit'] = df['D2'] > 0

    # Normative Nepheline / Albite
    df['Ne_'] = np.where((df['D2'] < 4*df['Ab_p']), df['D2']/4, df['Ab_p']).T

    df['Ab'] = np.where((df['D2'] < 4*df['Ab_p']), df['Ab_p'] - df['D2']/4, 0).T

    df['D3'] = df['D2'] - 4*df['Ab_p']

    df['Ne'] = np.where((df['deficit']), df['Ne_'], 0).T
    df['Ab'] = np.where((df['deficit']), df['Ab'], df['Ab_p']).T

    df['deficit'] = df['D3'] > 0

    # Normative Leucite / Orthoclase
    df['Lc'] = np.where((df['D3'] < 2*df['Or_p']), df['D3']/2, df['Or_p']).T

    df['Or'] = np.where((df['D3'] < 2*df['Or_p']), df['Or_p'] - df['D3']/2, 0).T

    df['D4'] = df['D3'] - 2*df['Or_p']

    df['Lc'] = np.where((df['deficit']), df['Lc'], 0).T
    df['Or'] = np.where((df['deficit']), df['Or'], df['Or_p']).T

    df['deficit'] = df['D4'] > 0

    # Normative dicalcium silicate / wollastonite
    df['Cs'] = np.where((df['D4'] < df['Wo_p']/2), df['D4'], df['Wo_p']/2).T

    df['Wo'] = np.where((df['D4'] < df['Wo_p']/2), df['Wo_p'] - 2*df['D4'], 0).T

    df['D5'] = df['D4'] - df['Wo_p']/2

    df['Cs'] = np.where((df['deficit']), df['Cs'], 0).T
    df['Wo'] = np.where((df['deficit']), df['Wo'], df['Wo_p']).T

    df['deficit'] = df['D5'] > 0

    # Normative dicalcium silicate / Olivine Adjustment
    df['Cs_'] = np.where((df['D5'] < df['Di_p']), df['D5']/2 + df['Cs'], df['Di_p']/2 + df['Cs']).T

    df['Ol_'] = np.where((df['D5'] < df['Di_p']), df['D5']/2 + df['Ol'], df['Di_p']/2 + df['Ol']).T

    df['Di_'] = np.where((df['D5'] < df['Di_p']), df['Di_p'] - df['D5'], 0).T

    df['D6'] = df['D5'] - df['Di_p']

    df['Cs'] = np.where((df['deficit']), df['Cs_'], df['Cs']).T
    df['Ol'] = np.where((df['deficit']), df['Ol_'], df['Ol']).T
    df['Di'] = np.where((df['deficit']), df['Di_'], df['Di_p']).T

    df['deficit'] = df['D6'] > 0

    # Normative Kaliophilite / Leucite
    df['Kp'] = np.where((df['Lc'] >= df['D6']/2), df['D6']/2, df['Lc']).T

    df['Lc_'] = np.where((df['Lc'] >= df['D6']/2), df['Lc'] - df['D6']/2, 0).T

    df['Kp'] = np.where((df['deficit']), df['Kp'], 0).T
    df['Lc'] = np.where((df['deficit']), df['Lc_'], df['Lc']).T

    df['DEFSIO2'] = np.where((df['Lc'] < df['D6']/2) & (df['deficit']), df['D6'] - 2*df['Kp'], 0).T

    # Allocate definite mineral proportions

    # Subdivide Hypersthene, Diopside and Olivine into Mg- and Fe- varieties

    df['Fe-Hy'] = df['Hy'] * df['FeO_ratio']
    df['Fe-Di'] = df['Di'] * df['FeO_ratio']
    df['Fe-Ol'] = df['Ol'] * df['FeO_ratio']

    df['Mg-Hy'] = df['Hy'] * df['MgO_ratio']
    df['Mg-Di'] = df['Di'] * df['MgO_ratio']
    df['Mg-Ol'] = df['Ol'] * df['MgO_ratio']




    weight_adjusted_minerals = ['Fe-Hy', 'Fe-Ol', 'Fe-Di', 'Mt', 'Il']
    subdivided = ['Fe-Hy', 'Fe-Di', 'Fe-Ol', 'Mg-Hy', 'Mg-Di', 'Mg-Ol']
    not_subdivided = ['Hy', 'Di', 'Ol']

    mineral_proportions = pd.DataFrame()
    mineral_pct_mm = pd.DataFrame()
    FREE = pd.DataFrame()
    if not majors_only:
        FREE['FREEO_12b'] = (
            1 + ((0.1)*((mineral_molecular_weights['CaF2-Ap']/328.86918)-1))
            ) * element_AW['O'] * df['FREEO_12b']
        FREE['FREEO_12c'] = (
            1 + ((0.1)*(df['CaF2-Ap']/df['Ap']) * ((mineral_molecular_weights['CaF2-Ap']/328.86918)-1))
            ) * element_AW['O'] * df['FREEO_12c']

        FREE['FREEO_13'] = (
            1+((oxide_molecular_weights['CaO']//56.0774)-1)
            ) * element_AW['O'] * df['FREEO_13']

        FREE['FREEO_14'] = (
            1+(0.5*((oxide_molecular_weights['Na2O']/61.9789)-1))
            ) * element_AW['O'] * df['FREEO_14']

        FREE['FREEO_16'] = (
            1+((oxide_molecular_weights['FeO']//71.8444)-1)
            ) * element_AW['O'] * df['FREEO_16']


        FREE['O'] = FREE[['FREEO_12b','FREEO_12c', 'FREEO_13', 'FREEO_14', 'FREEO_16']].sum(axis=1)

        FREE['CO2'] = df['FREECO2'] * 44.0095

        FREE['P2O5'] = df['FREE_P2O5'] * 141.94452
        FREE['F'] = df['FREE_F'] * 18.9984032
        FREE['Cl'] = df['FREE_Cl'] * 35.4527
        FREE['SO3'] = df['FREE_SO3'] * 80.0642
        FREE['S'] = df['FREE_S'] * 32.066
        FREE['Cr2O3'] = df['FREE_CR2O3'] * 151.990

        FREE['OXIDES'] = FREE[['P2O5', 'F', 'Cl', 'SO3', 'S', 'Cr2O3']].sum(axis=1)
        FREE['DEFSIO2'] = df['DEFSIO2'] * 60.0843
        FREE.drop(['P2O5', 'F', 'Cl', 'SO3', 'S', 'Cr2O3'], axis=1, inplace=True)

    else:
        FREE = 0

    for mineral in mineral_codes:
        try:
            if mineral in not_subdivided:
                pass
            elif mineral == ['Ap']:
                if majors_only:
                    mineral_proportions[mineral] = df[mineral]
                    mineral_pct_mm[mineral] = mineral_proportions[mineral] * mineral_molecular_weights[mineral]
                else:
                    mineral_pct_mm[mineral] = np.where(
                        df['ap_option'] == 2, df[mineral] * mineral_molecular_weights['CaO-Ap'],
                        (df['CaF2-Ap'] * mineral_molecular_weights['CaF2-Ap']) + (df['CaO-Ap'] * mineral_molecular_weights['CaO-Ap']))


            else:
                mineral_proportions[mineral] = df[mineral]
                mineral_pct_mm[mineral] = mineral_proportions[mineral] * mineral_molecular_weights[mineral]
        except:
            continue


    if not subdivide:
        mineral_pct_mm['Hy'] = mineral_pct_mm['Fe-Hy'] + mineral_pct_mm['Mg-Hy']
        mineral_pct_mm['Ol'] = mineral_pct_mm[['Fe-Ol', 'Mg-Ol']].sum(axis=1)
        mineral_pct_mm['Di'] = mineral_pct_mm[['Fe-Di', 'Mg-Di']].sum(axis=1)

        mineral_pct_mm.drop(['Fe-Hy', 'Mg-Hy', 'Fe-Ol', 'Mg-Ol', 'Fe-Di', 'Mg-Di'], axis=1, inplace=True)

        mineral_pct_mm['Pl'] = mineral_pct_mm[['An', 'Ab']].sum(axis=1)
        mineral_pct_mm.drop(['An', 'Ab'], axis=1, inplace=True)

    #eturn(mineral_molecular_weights, mineral_pct_mm.T, FREE[['OXIDES', 'O', 'CO2']])

    mineral_pct_mm.rename(columns=mineral_codes, inplace=True)

    mineral_pct_mm.fillna(0, inplace=True)

    mineral_pct_mm['Sum'] = mineral_pct_mm.sum(skipna=True, axis=1)


    return(mineral_pct_mm)


@st.cache
def fe_correction(df, method='Le Maitre', ig_type='plutonic', constant=None):
    """
        Adjusts FeO Fe2O3 ratio
        -----------
        df : :class:`pandas.DataFrame`
            dataframe containing values for FeO SiO2, Na2O and K2O
        Returns
        --------
        :class:`pandas.DataFrame`
            dataframe with corrected FeO and Fe2O3 values


        References:
        'La Maitre' method uses regressions from Le Maitre (1976)
    """

    df = df.copy(deep=True)

    # replace str values from df with 0
    def unique_strings(df, col):
        return df[col][df[col].map(type) == str].unique().tolist()

    # replace nans with 0

    df.fillna(0, inplace=True)

    for col in ['Fe2O3', 'SiO2', 'Na2O', 'K2O']:
        unique_str = (unique_strings(df, col))
        df[col].replace(unique_str, 0, inplace=True)
    method_values = ['Le Maitre', 'Constant']

    if method not in method_values:
        raise ValueError("Invalid method given. Expecting {}".format(method_values))

    ig_type_values = ['plutonic', 'volcanic']
    if ig_type not in ig_type_values:
        raise ValueError("Invalid ig_type given. Expecting {}".format(ig_type_values))


    df['total_Fe_as_FeO'] = (df['Fe2O3'] / 1.11134) + df['FeO']

    if method == 'Le Maitre':
        if ig_type == 'plutonic':
            fe_adjustment_factor = 0.88 - 0.0016 * df['SiO2'] - 0.027 * (df['Na2O'] + df['K2O'])
        elif ig_type =='volcanic':
            fe_adjustment_factor = 0.93 - 0.0042 * df['SiO2'] - 0.022 * (df['Na2O'] + df['K2O'])
        else:
            fe_adjustment_factor = None

    elif method == 'Constant':
        fe_adjustment_factor = constant
    else:
        fe_adjustment_factor = None


    df['adjusted_Fe2O3'] = df['total_Fe_as_FeO'] * (1 - fe_adjustment_factor) * 1.11134

    df['adjusted_FeO'] = df['total_Fe_as_FeO'] * fe_adjustment_factor


    df['Fe2O3'] = df['adjusted_Fe2O3']
    df['FeO'] = df['adjusted_FeO']

    return df[['FeO', 'Fe2O3']]


def download_df(df):
    xlsx = df.to_csv(index=False)
    b64 = base64.b64encode(
        xlsx.encode()
    ).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="normative_mineralogy.csv">Download results as csv file</a>'


def highlight_greaterthan(s, threshold, column):
    is_max = pd.Series(data=False, index=s.index)
    is_max[column] = s.loc[column] >= threshold
    return ['background-color: yellow' if is_max.any() else '' for v in is_max]

def highlight_lessthan(s, threshold, column):
    is_max = pd.Series(data=False, index=s.index)
    is_max[column] = s.loc[column] <= threshold
    return ['background-color: yellow' if is_max.any() else '' for v in is_max]

