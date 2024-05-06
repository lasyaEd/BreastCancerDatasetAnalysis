import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def main():
    st.title("Breast Cancer Diagnosis App")

    # Create tabs for different sections
    tabs = ["Data Exploration", "Model Training"]
    choice = st.sidebar.selectbox("Select Tab", tabs)

    if choice == "Data Exploration":
        data_exploration()

    elif choice == "Model Training":
        model_training()

def data_exploration():
    st.write("## Data Exploration")
    
    # Load the breast cancer dataset into a DataFrame
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target

    # Display dataset statistics and sample data
    st.write("Number of samples:", df.shape[0])
    st.write("Number of features:", df.shape[1] - 1)  # Exclude target column
    st.write("Classes:", data.target_names)

    # Display the first few rows of the DataFrame
    st.write("Sample Data:")
    st.write(df.head())

    # Visualize class distribution
    st.write("Class Distribution:")
    sns.countplot(x='target', data=df)
    st.pyplot()

    # Display correlation heatmap
    st.write("Correlation Heatmap:")
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
    st.pyplot()

def model_training():
    st.write("## Model Training")

    # Load the breast cancer dataset into a DataFrame
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target

    # Split the data into features (X) and target (y)
    X = df.drop('target', axis=1)
    y = df['target']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Logistic Regression (Logit) classifier
    logit = LogisticRegression(random_state=42)
    logit.fit(X_train, y_train)

    # Make predictions on test data
    y_pred_logit = logit.predict(X_test)

    # Evaluate the Logistic Regression model
    accuracy_logit = accuracy_score(y_test, y_pred_logit)
    st.write("Logistic Regression Accuracy:", accuracy_logit)

    st.write("Logistic Regression Classification Report:")
    st.write(classification_report(y_test, y_pred_logit, target_names=data.target_names))

    # Display confusion matrix for Logistic Regression
    st.write("Confusion Matrix (Logistic Regression):")
    cm_logit = confusion_matrix(y_test, y_pred_logit)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_logit, annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=data.target_names, yticklabels=data.target_names)
    st.pyplot()

    # Train a Decision Tree classifier
    dtree = DecisionTreeClassifier(random_state=42)
    dtree.fit(X_train, y_train)

    # Make predictions on test data
    y_pred_dtree = dtree.predict(X_test)

    # Evaluate the Decision Tree model
    accuracy_dtree = accuracy_score(y_test, y_pred_dtree)
    st.write("Decision Tree Accuracy:", accuracy_dtree)

    st.write("Decision Tree Classification Report:")
    st.write(classification_report(y_test, y_pred_dtree, target_names=data.target_names))

    # Display confusion matrix for Decision Tree
    st.write("Confusion Matrix (Decision Tree):")
    cm_dtree = confusion_matrix(y_test, y_pred_dtree)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_dtree, annot=True, fmt='d', cmap='Greens', cbar=False, 
                xticklabels=data.target_names, yticklabels=data.target_names)
    st.pyplot()

if __name__ == '__main__':
    main()
