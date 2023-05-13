import streamlit as st
import pandas as pd
from PIL import Image
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("prediction_service/model.h5")

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # To read file as bytes:
    img = Image.open(uploaded_file)
    img = img.resize((224,224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis =0)
    results = model.predict(img_array)
    argmax_index = np.argmax(results, axis = 1)
    if argmax_index[0] == 0:
        st.image(img, use_column_width="always")
        st.markdown("<h1 style='text-align: center; color: green;'><b>Predicted</b> : Cat</h1>", unsafe_allow_html=True)
    else:
        st.image(img, use_column_width="always")
        st.markdown("<h1 style='text-align: center; color: green;'><b>Predicted</b> : Dog</h1>", unsafe_allow_html=True)
