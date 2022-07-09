import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_folium import folium_static
import folium
import os
import pandas as pd
from streamlit.components.v1 import html
from common import set_page_container_style

st.set_page_config(
    page_title='My App',
    layout='wide',
    page_icon=':rocket:'
)

my_js = """
function redirect(batch){
    alert(batch);
}
"""

my_html = f"<script>{my_js}</script>"

html(my_html)

with st.sidebar:
    selected = option_menu("", ['Introduccion', 'Limitaciones', 'Technical approach','Modelo'],
        icons=['patch-question', 'activity','tools','tools'], menu_icon="cast", default_index=0,
        styles={
        "container": {"padding": "5!important", "background-color": "#fafafa"},
        "icon": {"color": "green", "font-size": "25px"},
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#02ab21"},
    })

if selected == "Limitaciones":
    st.image("logo.png")
    st.markdown('''

    # Limitaciones

    - Dependencia de pronósticos para variables regresoras.
    - Un modelo por lote, no regional.
    - Agnóstico de la especie de pastura.

    ''' )

else:
    if selected == "Introduccion":
        st.image("logo.png")
        st.markdown('''

        # Is the grass greener on the other side (of winter)?

        Cattle farmers nedd to know how winter/summer pastures are going to behave in order to make smarter environmental and economics decisions.
        Low temperatures/drought are main controls of pasture growth.

        ''' )

        col1, col2, col3 = st.columns(3)
        col1.image('img/img1.JPG')
        col2.image('img/img3.JPG')
        col3.image('img/img2.JPG')
    else:
        if selected == "Technical approach":
            st.image("logo.png")
            st.markdown('''
            # Technical approach

            - Adquisición de datos de dos APIs distintas.
            - Limpieza de datos:
            -- Interpolar datos vacíos y extremos.
            -- Agregar datos a 8 días según cada variable.
            - Entrenamiento en máquina local (modelos ligeros)

            ''')
        else:
            if selected == "Modelo":


                col1, col2, col3, col4 = st.columns(4)
                col1.metric('Baseline (año pasado)','0.3','Comentario 1')
                col2.metric('Random Forest','0.28','Comentario 2')
                col3.metric('SARIMAX','0.27','Comentario 3')
                col4.metric('Prophet','0.25','Comentario 4')

                m = folium.Map(location=[ -35, -57], zoom_start=5)

                data = pd.DataFrame({
                'lon':[-57.629691, -66.09529],
                'lat':[-35.200358, -34.23537],
                'name':['Vieytes', 'San Luis'],
                'value':['vieytes', 'san_luis']
                }, dtype=str)

                for i in range(0,len(data)):

                    html=f"<h1><a href='http://172.22.234.51:8502?batch={data.iloc[i]['name']}'>{data.iloc[i]['name']}</a></h1>"

                    folium.Marker(
                        location=[data.iloc[i]['lat'], data.iloc[i]['lon']],
                        popup=html ,
                    ).add_to(m)

                #st_data = folium_static(m)

                m.add_child(folium.LatLngPopup())

                map = folium_static(m, height=350, width=700)