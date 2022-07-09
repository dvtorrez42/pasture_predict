import streamlit as st
import webbrowser

from streamlit.components.v1 import html
import webbrowser

url = 'http://172.22.234.51:8501?batch=San_Luis'

my_js = """
window.location.href = '{url}';
"""

my_html = f"<script>{my_js}</script>"

html(my_html)



if st.button('Open browser'):
    webbrowser.open(url)
