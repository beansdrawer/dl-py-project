import requests
import streamlit as st
from PIL import Image

st.set_option("deprecation.showfileUploaderEncoding", False)
st.title("Image Classification Model Test")
image = st.file_uploader("Choose an image")

if st.button("Prediction"):
    if image is not None:
        files = {"file": image.getvalue()}
        headers={"Content-Type": "multipart/form-data"}
        res = requests.post(f"http://127.0.0.1:8001/detect_labels",  files=files)
        res = eval(res.text)
        coffee_prob = round(float(res['Coffee']),2)
        monitor_prob = round(float(res['Monitor']),2)
        st.write(f"{{'Coffee':{coffee_prob}, 'Monitor':{monitor_prob}}}")
        st.image(image.getvalue(), width=240)