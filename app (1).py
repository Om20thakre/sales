from scipy.sparse import data
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix
from scipy.sparse import csr_matrix, coo_matrix, csc_matrix
# Sidebar to choose classifier
def sidebar():
    st.sidebar.title("Choose Classifier")
    classifier = st.sidebar.selectbox("Select Classifier", ("Decision Tree", "Random Forest"))
    return classifier


# Load the data
@st.cache
def load_data(uploaded_file):
    data = pd.read_csv(uploaded_file)
    return data


# Data Exploration and Visualization
def explore_and_visualize_data(data):
    st.title("Supermarket Sales Data Exploration and Visualization")
    explore_data(data)
    visualize_data(data)

# Exploratory Data Analysis
def explore_data(data):
    st.subheader("Data Description")
    st.write(data.describe())
    st.subheader("Data Info")
    st.write(data.info())
    st.subheader("Null Values")
    st.write(data.isnull().sum())
    st.subheader("Data Types")
    st.write(data.dtypes)

# Data Visualization
def visualize_data(data):
    st.subheader("Data Visualization")
    payment_distribution(data)
    city_sales(data)
    product_line_distribution(data)
    histograms(data)
    correlation_heatmap(data)

# Payment Distribution
def payment_distribution(data):
    st.write("Payment Distribution")
    plt.figure(figsize=(6,6))
    sns.histplot(data['Payment'])
    st.pyplot()

# City Sales
def city_sales(data):
    st.write("City Sales")
    plt.figure(figsize=(4,4))
    sns.countplot(x='City', data=data)
    st.pyplot()

# Product Line Distribution
def product_line_distribution(data):
    st.write("Product Line Distribution")
    plt.figure(figsize=(10,6))
    sns.countplot(x='Product line', data=data)
    st.pyplot()

# Histograms
def histograms(data):
    st.write("Histograms")
    plt.figure(figsize=(20,14))
    data.hist()
    st.pyplot()

# Correlation Heatmap
def correlation_heatmap(data):
    st.write("Correlation Heatmap")
    numeric_data = data.select_dtypes(include=['int64', 'float64'])
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm')
    st.pyplot()

# Preprocessing
def preprocessing(data):
    label_encoders = {}
    for column in data.columns:  
        label_encoders[column] = LabelEncoder()
        data[column] = label_encoders[column].fit_transform(data[column])
    return data

# Model training
def train_model(data, classifier):
    data = preprocessing(data)
    X = data.drop(columns=['Gender'])
    y = data['Gender']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if classifier == "Decision Tree":
        st.write("Using Decision Tree Classifier")
        dtree = DecisionTreeClassifier(max_depth=6, random_state=123, criterion='entropy')
        dtree.fit(X_train, y_train)
        y_pred = dtree.predict(X_test)
        st.write("Classification Report:", classification_report(y_test, y_pred))
        st.write("Confusion Matrix:", confusion_matrix(y_test, y_pred))
        st.write("Training Score:", dtree.score(X_train, y_train) * 100)

    elif classifier == "Random Forest":
        st.write("Using Random Forest Classifier")
        rfc = RandomForestClassifier()
        rfc.fit(X_train, y_train)
        y_pred = rfc.predict(X_test)
        st.write("Classification Report:", classification_report(y_test, y_pred))
        st.write("Confusion Matrix:", confusion_matrix(y_test, y_pred))
        st.write("Training Score:", rfc.score(X_train, y_train) * 100)

    return

# Main function
def main():
    st.title("Supermarket Sales Analysis")
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    if uploaded_file is not None:
        data = load_data(uploaded_file)
        explore_and_visualize_data(data)
        classifier = sidebar()
        train_model(data, classifier)

if __name__ == "__main__":
    main()
