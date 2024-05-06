import streamlit as st
from sklearn import datasets
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Set page configuration and custom CSS for background color
st.set_page_config(page_title="Breast Cancer Diagnosis Assistant", page_icon="🎗️")
custom_css = f"""
    <style>
        body {{
            background-color: #FFC0CB; /* Breast cancer pink color code */
        }}
        .stApp {{
            background-color: #FFC0CB; /* Apply to entire app container */
        }}
    </style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

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
selected_tab = st.selectbox("Explore options", tabs)

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
    feature_names = breastCancer.feature_names
    df['target'] = breastCancer.target
    # Select pairs of features for pair plot using Streamlit multiselect widget
    selected_pairs = st.multiselect("Select Feature Pairs", options=feature_names, default=feature_names[:4])

    if len(selected_pairs) > 4:
        st.warning("Please select a maximum of 4 feature pairs.")
        selected_pairs = selected_pairs[:4]  # Limit selection to first 8 features


    # Filter DataFrame to include selected feature pairs
    selected_df = df[selected_pairs + ['target']]

    # Pair plot based on selected feature pairs
    if selected_pairs:
        st.write(f"Pair Plot for Selected Feature Pairs: {', '.join(selected_pairs)}")
        pairplot = sns.pairplot(selected_df, hue='target')
        st.pyplot(pairplot)
    else:
        st.write("Please select feature pairs to display the pair plot.")


def model_training():
    X = breastCancer.data
    y = breastCancer.target
    feature_names = breastCancer.feature_names
    st.write("## Explore feature selection and predictive modeling.")
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y

    # Compute correlation matrix
    corr_matrix = df.corr()

    # Define correlation threshold
    threshold = 0.75

    # Find highly correlated features
    corr_pairs = np.where(np.abs(corr_matrix) > threshold)
    corr_features = [(corr_matrix.index[x], corr_matrix.columns[y]) for x, y in zip(*corr_pairs) if x != y and x < y]

    # Drop highly correlated features
    uncorrelated_features = set(feature_names) - set(sum(corr_features, ()))
    filtered_df = df[list(uncorrelated_features) + ['target']]

    # Split data into features (X) and target (y)
    X = filtered_df.drop('target', axis=1)
    y = filtered_df['target']

    # Standardize the feature data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Fit a logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate model performance
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Display results in Streamlit app
    st.title("Model Evaluation with Less Correlated Features")
    st.write(f"Accuracy: {accuracy:.2f}")
    st.write("Classification Report:")
    st.write(report)



# Render content based on selected tab
if selected_tab == "Data Exploration":
    data_exploration()
elif selected_tab == "Data Visualization":
    data_visualization()
elif selected_tab == "Model Training":
    model_training()
