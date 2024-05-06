import streamlit as st
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

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
st.title("Breast Cancer Dataset Exploration Predictive Analysis")
#st.image("data/pink_ribbon.png", width=200)

# Display app description
st.markdown("""
This app provides interactive tools for analyzing breast cancer data. Explore the characteristics of breast tumors, predict diagnosis outcomes, and gain insights from machine learning models. Understand the features that contribute to tumor classification and empower informed decision-making.

<small><em>Please note: This app is for educational purposes only. Do not use this app for self-diagnosis or medical decision-making.</em></small>
""", unsafe_allow_html=True)

# Load breast cancer dataset
breastCancer = datasets.load_breast_cancer()
df = pd.DataFrame(breastCancer.data, columns=breastCancer.feature_names)
X = breastCancer.data
y = breastCancer.target

# Display horizontal tabs for navigation
tabs = ["Data Exploration", "Model Training"]
selected_tab = st.selectbox("Select Tab", tabs)

# Define functions for Data Exploration and Model Training
def data_exploration():
    st.write("## Data Exploration")
    st.write("Number of samples:", df.shape[0])
    st.write("Number of features:", df.shape[1])
    st.write("Classes:", breastCancer.target_names)
    st.write("Sample Data:")
    st.write(df.head(7))

    # Display correlation heatmap using Seaborn and Matplotlib
    st.write("### Correlation Heatmap")
    plt.figure(figsize=(14, 12))
    sns.heatmap(df.astype(float).corr(), linewidths=0.1, square=True, linecolor='white', annot=True)
    heatmap_fig = plt.gcf()  # Get current figure
    st.pyplot(heatmap_fig)  # Display the figure explicitly

    # Create the countplot using Seaborn and Matplotlib
    fig = plt.figure()
    ax = sns.countplot(x='bland_chromatin', hue='class', data=breastCancer)
    ax.set(xlabel='Bland Chromatin', ylabel='No of cases')
    plt.title("Bland Chromatin w.r.t. Class", y=0.96)

    # Display the plot using Streamlit
    st.pyplot(fig)
    

def model_training():
    st.write("## Model Training")
    # Placeholder for model training code

# Render content based on selected tab
if selected_tab == "Data Exploration":
    data_exploration()
elif selected_tab == "Model Training":
    model_training()
