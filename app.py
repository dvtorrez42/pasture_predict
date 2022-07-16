from cmath import nan
import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_folium import folium_static
import folium
from pasture_predict.params import API_URL, SITE_URL
#import os
import pandas as pd
from streamlit.components.v1 import html
from common import set_page_container_style
import requests
from pasture_predict.data import get_data
import plost
import numpy as np

batch = ""
default_index = 0

def get_data_api(batch, predict):

    strbatch = ''.join([s.lower() for s in batch])

    url = f'{API_URL}/predict?batch_name={batch}&predict={predict}'
    response = requests.get(url)
    if response.status_code != 200:
        return ''
    #st.write(response.json())
    data = pd.DataFrame.from_dict(response.json())
    #st.write(strbatch)
    #df = get_data(strbatch, f"predict_{predict}")
    return(data)


def show_batch(batch):

    strbatch = ''.join([s.lower() for s in batch])

    strbatch = strbatch.replace(' ', '')

    df1 = get_data_api(batch, 1)

    #st.write(df1)

    df200 = get_data_api(batch, 200)
    #st.write(df200)
    df200 = df200.dropna()


    df200['ds'] = pd.to_datetime(df200['ds'], format='%d/%m/%Y')


    st.markdown(f"""
                ### Pronóstico acumulado: {round(df1.iloc[0,1],3)} Kg C / Ha
                 """)

    st.markdown(f"""
                 ### Pronóstico
                 """)


    df11 =df200[df200.yhat_lower >0]
    # #df11.columns=['Fecha','Predicción','I.C.Superior','I.C.Inferior']


    df200 = df200[-100:]
    df200['pred']=df200.yhat
    df200.pred[:-11] = np.nan
    df200.yhat[-10:] = np.nan
    df200.yhat_upper[df200.yhat_upper==0] = np.nan
    df200.yhat_lower[df200.yhat_lower==0] = np.nan
    df200.yhat_lower[df200.yhat_lower<0] = 0

    #st.write(df200)

    df200.columns=['Fecha','Prod','L.S.','L.I.','Pred']

    plost.line_chart(
         data=df200,
         x='Fecha',
         y=('Prod','Pred','L.S.','L.I.'))

st.set_page_config(
    page_title="Pleasant pasture", # => Quick reference - Streamlit
    page_icon="??",
    layout="centered", # wide
    initial_sidebar_state="auto") # collapsed

query_params = st.experimental_get_query_params()

with st.sidebar:
    selected = option_menu("", ['Problema', 'Quienes Somos',  'Solución', 'Aspectos técnicos', 'Próximos pasos...'],
        icons=['patch-question','people-fill','activity','tools','signpost'], menu_icon="cast", default_index=default_index,
        styles={
        "container": {"padding": "5!important", "background-color": "#fafafa"},
        "icon": {"color": "green", "font-size": "25px"},
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#02ab21"},
    })

if selected =="Arquitectura":
    st.image("img/logo.png")
    st.header("Aspectos Técnicos")

if selected == 'Quienes Somos':
    st.image("img/logo.png")
    st.markdown(f"{batch}")

    st.header("Quienes Somos")

    col1, col2, col3, col4 = st.columns([4,4,4,4])

    col1.image("img/DaniV.jpg")
    col1.markdown('<div style="text-align: center;">Daniel Villarroel</div>', unsafe_allow_html=True)

    col2.image("img/danib.jpg")
    col2.markdown('<div style="text-align: center;">Daniela Brondino</div>', unsafe_allow_html=True)

    col3.image("img/GabiC.jpg")
    col3.markdown('<div style="text-align: center;">Gabriel Causa</div>', unsafe_allow_html=True)

    col4.image("img/Lucho.jpg")
    col4.markdown('<div style="text-align: center;">Luciano Valente</div>', unsafe_allow_html=True)



if selected == "Próximos pasos...":
    st.image("img/logo.png")
    st.markdown(f"{batch}")

    st.header("Próximos pasos...")
    st.markdown('''
    - Explorar mejor la influencia de variables exógenas.
    - Mejorar la escalabilidad del modelo.
    - Incorporar información sobre las especies de pasturas.
    - ...
    ''' )
    # Antes...
    #  - Dependencia de pronósticos para variables regresoras.
    #  - Un modelo por lote, no regional.
    #  - Agnóstico de la especie de pastura.

else:
    if selected == "Problema":
        st.image("img/logo.png")
        st.title('')
        st.markdown('<h1 style="text-align: center;">¿Es el pasto verde después del invierno?</h1>', unsafe_allow_html=True)


        col1, col2, col3 = st.columns(3)
        col1.image('img/img1.JPG')
        #col2.image('img/img3.JPG')
        col3.image('img/img2.JPG')

    else:
        if selected == "Aspectos técnicos":
            st.image("img/logo.png")

            st.markdown('''
                        # Productividad de pasturas
                        ''')

            col1, col2, col3, col4 = st.columns([4,4,4,4])

            with col1:
                st.markdown('''
                            ### Baseline
                            ### 0,38''')
                #st.header('Baseline')
                #st.metric('','0.38')
            with col2:
                st.markdown('''
                            ### XGBoost
                            ### 0,30''')
                #st.header('XGBoost')
                #st.metric('','0.30')
            with col3:
                st.markdown('''
                        ### Sarimax
                        ### 0,26''')
                #st.header('Sarimax')
                #st.metric('','0.26')
            with col4:
                st.markdown('''
                        ### Prophet
                        ### 0,23''')
                #st.header('Prophet')
                #st.metric('','0.23')

            col5, col6 = st.columns([3,15])

            with col5:
                model_graph = st.radio('Lote', ('Baseline', 'XGBoost', 'Sarimax', 'Prophet'))
            with col6:
                if model_graph == 'Baseline':
                    st.image('img/Baseline.png', caption=None, width=None,
                            use_column_width=None, clamp=False, channels="RGB", output_format="auto")
                elif model_graph == 'XGBoost':
                    st.image('img/ML_Approach.png', caption=None, width=None,
                            use_column_width=None, clamp=False, channels="RGB", output_format="auto")
                elif model_graph == 'Sarimax':
                    st.image('img/sarimax.png', caption=None, width=None,
                            use_column_width=None, clamp=False, channels="RGB", output_format="auto")
                elif model_graph == 'Prophet':
                    st.image('img/Prophet.png', caption=None, width=None,
                            use_column_width=None, clamp=False, channels="RGB", output_format="auto")
                else:
                    pass

            st.markdown("#### Acumulado:")

            col7, col8, col9 = st.columns([5,5,5])

            with col7:
                    st.markdown('''
                                ### Baseline
                                ### 0,22''')
            with col8:
                    st.markdown('''
                                ### Random Forest
                                ### 0,11''')
            with col9:
                    st.markdown('''
                                ### XGBoost
                                ### 0,08''')

            st.markdown('''
                        ##### Horizonte temporal: tres meses.
                        ##### Mean Absolute Percentage Error (MAPE).
                        ''')
        else:
            if selected == "Solución":

                st.image("img/logo.png")

                st.markdown('''
                # Plataforma SaaS
                ''' )


                #col1, col2 = st.columns([3,15])

                #with col1:
                option = st.selectbox('', ('San Luis', 'Vieytes'))
               # with col2:
                if option == 'San Luis':
                    show_batch('sanluis')
                elif option == 'Vieytes':
                    show_batch('vieytes')
                else:
                    pass


                # if batch != "":
                #     strbatch = ''.join([s.lower() for s in batch])

                #     st.markdown(f"""
                #         # {strbatch.upper()}
                #         """)

                #     col1, col2= st.columns([5,15])
                #     col1.markdown(f"""
                #                 ## Título
                #                 ## {df1.iloc[0,0]}
                #                 """)

                #     col2.markdown(f"""
                #                 ## Pronóstico
                #                 """)
                #     col2.line_chart(df11,1500,200)

                #     st.markdown(f"""
                #                 ### Datos historicos
                #                 """)

                #     st.line_chart(df200['prod'])

                #if batch == "":

                    # st.markdown('''
                    # # Plataforma SaaS
                    # ''' )

                m = folium.Map(location=[ -35, -57], zoom_start=5)

                if option == 'Vieytes':
                    data = pd.DataFrame({
                    'lon':[-57.629691],
                    'lat':[-35.200358],
                    'name':['Vieytes'],
                    'value':['vieytes']
                    }, dtype=str)
                else:
                    data = pd.DataFrame({
                    'lon':[-66.09529],
                    'lat':[-34.23537],
                    'name':['San Luis'],
                    'value':['san_luis']
                    }, dtype=str)

                for i in range(0,len(data)):
                    #target='_parent'
                    html=f"<h1><a target='parent' href='{SITE_URL}?batch={data.iloc[i]['name']}'>{data.iloc[i]['name']}</a></h1>"

                    folium.Marker(
                        location=[data.iloc[i]['lat'], data.iloc[i]['lon']],
                        popup=html ,
                    ).add_to(m)



                m.add_child(folium.LatLngPopup())

                map = folium_static(m, height=350, width=700)
