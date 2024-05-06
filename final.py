import streamlit as st
from sklearn import datasets
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set page configuration and custom CSS for background color
st.set_page_config(page_title="Breast Cancer Diagnosis Assistant", page_icon="🎗️")
# custom_css = f"""
#     <style>
#         body {{
#             background-color: #FFC0CB; /* Breast cancer pink color code */
#         }}
#         .stApp {{
#             background-color: #FFC0CB; /* Apply to entire app container */
#         }}
#     </style>
# """
# st.markdown(custom_css, unsafe_allow_html=True)

# Display title and image
st.title("Breast Cancer Dataset Exploration and Classification")
st.image("data/pink_ribbon.png", width=120)

# Display app description
st.markdown("""
This app provides interactive tools for analyzing breast cancer data. Explore the characteristics of breast tumors, predict diagnosis outcomes, and gain insights from machine learning models. Understand the features that contribute to tumor classification and empower informed decision-making.

<small><em>Please note: This app is for educational purposes only. Do not use this app for self-diagnosis or medical decision-making.</em></small>
""", unsafe_allow_html=True)

# Load breast cancer dataset
breastCancer = datasets.load_breast_cancer()
df = pd.DataFrame(breastCancer.data, columns=breastCancer.feature_names)
df['target'] = breastCancer.target

# Display horizontal tabs for navigation
tabs = ["Data Exploration", "Data Visualization", "Model Training"]
selected_tab = st.selectbox("Select Tab", tabs)

# Define functions for Data Exploration and Model Training
def data_exploration():
    st.write("## Data Exploration")
    st.write("Number of samples:", df.shape[0])
    st.write("Number of features:", df.shape[1])
    st.write("Classes:", breastCancer.target_names)
    st.write("Sample Data:")
    st.write(df.head(7))

def data_visualization():
    # Display correlation heatmap using Seaborn and Matplotlib
    st.write("### Correlation Heatmap")
    st.write("Correlation is a statistical metric that quantifies the association between two variables, ranging from -1 to 1. A positive correlation signifies a direct relationship, whereas a negative correlation indicates an inverse relationship. ")
    corr_matrix = df.corr()
    threshold = st.slider("Select Correlation Threshold", min_value=0.0, max_value=1.0, value=0.75, step=0.01)
    filtre = np.abs(corr_matrix["target"]) > threshold
    corr_features = corr_matrix.columns[filtre].tolist()

    plt.figure(figsize=(14, 12))
    sns.heatmap(df[corr_features].corr(), linewidths=0.1, square=True, linecolor='white', annot=True)
    heatmap_fig = plt.gcf()  # Get current figure
    st.write(f"### Correlation Heatmap (Threshold: {threshold})")
    st.pyplot(heatmap_fig)  # Display the heatmap figure

    # Generate pairplot for selected features with customized settings
    st.write("### Pairplot with KDE Diagonals and Target Variable")
    pairplot_fig = sns.pairplot(df[corr_features], diag_kind="kde", markers="+", hue="target")
    plt.tight_layout()  # Adjust layout for better visualization

    # Display the pairplot using st.pyplot()
    st.pyplot(pairplot_fig.fig)  # Display the pairplot figure

def model_training():
    st.write("## Model Training")
    # Placeholder for model training code

# Render content based on selected tab
if selected_tab == "Data Exploration":
    data_exploration()
elif selected_tab == "Data Visualization":
    data_visualization()
elif selected_tab == "Model Training":
    model_training()
