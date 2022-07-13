import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_folium import folium_static
import folium
#import os
import pandas as pd
from streamlit.components.v1 import html
from common import set_page_container_style

batch = ""


st.set_page_config(
    page_title="Pleasant pasture", # => Quick reference - Streamlit
    page_icon="🌿",
    layout="centered", # wide
    initial_sidebar_state="auto") # collapsed
    
    
query_params = st.experimental_get_query_params()
if len(query_params)>0:
    batch = query_params["batch"]
    print (batch)

with st.sidebar:
    selected = option_menu("", ['Problema','Solución', 'Aspectos técnicos', 'Próximos pasos...'],
        icons=['patch-question', 'activity','tools','signpost'], menu_icon="cast", default_index=0,
        styles={
        "container": {"padding": "5!important", "background-color": "#fafafa"},
        "icon": {"color": "green", "font-size": "25px"},
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#02ab21"},
    })

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
        st.markdown('''
        # Is the grass greener on the other side (of winter)?
        ''' )

        col1, col2, col3 = st.columns(3)
        col1.image('img/img1.JPG')
        col2.image('img/img3.JPG')
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
                model_graph = st.radio('', ('Baseline', 'XGBoost', 'Sarimax', 'Prophet'))
            with col6:
                if model_graph == 'Baseline':
                    st.image('img/baseline.png', caption=None, width=None, 
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


                st.markdown('''
                # Plataforma SaaS
                ''' )

                m = folium.Map(location=[ -35, -57], zoom_start=5)

                data = pd.DataFrame({
                'lon':[-57.629691, -66.09529],
                'lat':[-35.200358, -34.23537],
                'name':['Vieytes', 'San Luis'],
                'value':['vieytes', 'san_luis']
                }, dtype=str)

                for i in range(0,len(data)):

                    html=f"<h1><a target='_blank' href='http://172.22.234.51:8501?batch={data.iloc[i]['name']}'>{data.iloc[i]['name']}</a></h1>"

                    folium.Marker(
                        location=[data.iloc[i]['lat'], data.iloc[i]['lon']],
                        popup=html ,
                    ).add_to(m)

                #st_data = folium_static(m)

                m.add_child(folium.LatLngPopup())

                map = folium_static(m, height=350, width=700)
