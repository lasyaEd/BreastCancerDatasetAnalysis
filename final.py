import streamlit as st
from sklearn import datasets
import pandas as pd

# Load the Breast Cancer Wisconsin dataset
breastCancer = datasets.load_breast_cancer()

# Convert the dataset into a DataFrame
df = pd.DataFrame(breastCancer.data, columns=breastCancer.feature_names)
st.set_page_config(page_title="Breast Cancer Diagnosis Assistant", page_icon="🎗️")


# Set up Streamlit app layout
st.image("data/pink_ribbon.png", width=200)
st.title("Breast Cancer Diagnosis Assistant")

# Display app description
st.markdown("""
Our app provides interactive tools for analyzing breast cancer data. Explore the characteristics of breast tumors, predict diagnosis outcomes, and gain insights from machine learning models. Understand the features that contribute to tumor classification and empower informed decision-making.

<small><em>Please note: This app is for educational purposes only. Do not use this app for self-diagnosis or medical decision-making.</em></small>
""", unsafe_allow_html=True)
