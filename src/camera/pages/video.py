import streamlit as st
from ar import run

detector = run('bottle')
st.title("Visualize It")
detector()