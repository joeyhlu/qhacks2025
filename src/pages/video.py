import streamlit as st
from camera.ObjectDetection import ObjectDetection

detector = ObjectDetection(0, 1280, 720, True)
st.title("Application")
detector()