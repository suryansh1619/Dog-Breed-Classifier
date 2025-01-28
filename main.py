import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from model import predict_breed  


st.title("Dog Breed Classifier")

uploaded_file = st.file_uploader("Upload a Dog Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    st.image(uploaded_file, caption="Uploaded Image",use_container_width=True)
    
    breeds = predict_breed(uploaded_file)
    

    st.write("Top 5 Predicted Breeds:")
    breed_data = pd.DataFrame(breeds)  
    st.table(breed_data)  

    breed_names = breed_data['breed'].tolist()
    probabilities = breed_data['probability'].tolist()

    fig, ax = plt.subplots()
    ax.pie(probabilities, labels=breed_names, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  
    st.pyplot(fig)

    fig, ax = plt.subplots()
    ax.barh(breed_names, probabilities, color='skyblue')
    ax.set_xlabel('Probability')
    ax.set_title('Breed Prediction Probabilities')
    st.pyplot(fig)
