import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
import os
import kagglehub
import shutil
import time

# Set page config
st.set_page_config(
    page_title="Financial Inclusion Analysis",
    page_icon="💰",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("💰 Financial Inclusion Analysis Dashboard")
st.markdown("""
This dashboard analyzes financial inclusion data from the World Bank's Global Findex Database.
Explore the data, view visualizations, and understand the factors affecting financial inclusion worldwide.
""")

# Download data function
def download_data():
    st.info("Downloading dataset from KaggleHub. This may take a moment...")
    path = kagglehub.dataset_download("annalie/findex-world-bank")
    src = os.path.join(path, "FINDEXData.csv")
    dst_dir = "data"
    dst = os.path.join(dst_dir, "FINDEXData.csv")
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    if not os.path.exists(dst):
        shutil.copyfile(src, dst)
    st.success("Dataset downloaded and ready!")
    st.rerun()

# Add a download button in the sidebar
if st.sidebar.button("Download/Update Dataset"):
    download_data()

# Sidebar for controls
st.sidebar.title("Controls")

# File uploader for manual data upload
st.sidebar.markdown(
    "You can also upload your own dataset. Make sure it matches the structure of the original FINDEXData.csv."
)
uploaded_file = st.sidebar.file_uploader(
    "Or upload your own FINDEXData.csv file",
    type=["csv"],
    help="Upload a CSV file with the same structure as FINDEXData.csv"
)
if uploaded_file is not None:
    data_path = "data/FINDEXData.csv"
    if not os.path.exists("data"):
        os.makedirs("data")
    with open(data_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    # Wait a moment to ensure file is written
    time.sleep(1)
    st.sidebar.success("File uploaded and saved! The dashboard will now use your uploaded data.")
    st.rerun()

# Load data
@st.cache_data
def load_data():
    data_path = 'data/FINDEXData.csv'
    if not os.path.exists(data_path):
        st.warning("Dataset not found. Please use the 'Download/Update Dataset' button in the sidebar to fetch the data.")
        return None
    df = pd.read_csv(data_path)
    return df

df = load_data()
if df is None:
    st.stop()

# Create pivot table
@st.cache_data
def process_data(df):
    pivoted_df = df.pivot_table(
        index=['Country Name', 'Country Code'],
        columns='Indicator Name',
        values='MRV'
    ).reset_index()
    return pivoted_df

pivoted_df = process_data(df)

# Sidebar for controls
st.sidebar.title("Controls")
target_indicator = st.sidebar.selectbox(
    "Select Financial Indicator",
    options=pivoted_df.columns[2:],  # Exclude Country Name and Code
    index=0
)

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Country Analysis", "Model Performance", "Global Trends"])

with tab1:
    st.header("Data Overview")
    st.info(
        """
        **About the Dataset:** This dataset is sourced from the World Bank's Global Findex Database. It contains financial inclusion indicators for countries worldwide, such as account ownership, borrowing, saving, and more. Each indicator reflects a specific aspect of financial access or usage in the population.
        
        **Selected Indicator:** The indicator you select from the sidebar will be used for all visualizations and analysis on this dashboard. The distribution plot below shows how this indicator varies across countries.
        """
    )
    # Display basic statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Countries", len(pivoted_df))
    with col2:
        st.metric("Total Indicators", len(pivoted_df.columns) - 2)
    with col3:
        st.metric("Average Financial Inclusion", f"{pivoted_df[target_indicator].mean():.2f}%")
    
    # Distribution plot
    st.subheader("Distribution of Financial Inclusion")
    fig = px.histogram(pivoted_df, x=target_indicator, nbins=30,
                      title=f"Distribution of {target_indicator}")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Country Analysis")
    st.info(
        """
        **Top & Bottom Countries:**
        These charts show the countries with the highest and lowest values for the selected financial inclusion indicator. This helps identify global leaders and laggards in financial access or usage.
        """
    )
    # Top and bottom countries
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top 30 Countries")
        top_countries = pivoted_df.nlargest(30, target_indicator)
        fig = px.bar(top_countries, x=target_indicator, y='Country Name',
                    orientation='h', title="Highest Financial Inclusion")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Bottom 30 Countries")
        bottom_countries = pivoted_df.nsmallest(30, target_indicator)
        fig = px.bar(bottom_countries, x=target_indicator, y='Country Name',
                    orientation='h', title="Lowest Financial Inclusion")
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Model Performance")
    st.info(
        """
        **Machine Learning Model:**
        We use a logistic regression model to classify countries as having high or low financial inclusion (relative to the median). Feature importance shows which indicators are most influential in predicting financial inclusion.
        """
    )
    # Prepare data for model
    X = pivoted_df.drop(columns=['Country Name', 'Country Code', target_indicator])
    y = pivoted_df[target_indicator]
    
    # Handle missing values
    X = X.fillna(X.mean())
    y = y.fillna(y.mean())
    
    # Convert target to binary
    y_binary = (y > y.median()).astype(int)
    
    # Split and train model
    X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    # Display metrics
    accuracy = accuracy_score(y_test, y_pred)
    st.metric("Model Accuracy", f"{accuracy:.2%}")
    
    # Feature importance
    st.subheader("Feature Importance")
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.coef_[0]
    }).sort_values('Importance', ascending=False)
    
    fig = px.bar(feature_importance, x='Importance', y='Feature',
                orientation='h', title="Feature Importance for Prediction")
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("Global Trends")
    st.info(
        """
        **Correlation Heatmap:**
        This heatmap shows the relationships between different financial indicators. Strong positive or negative correlations can reveal how different aspects of financial inclusion are related across countries.
        """
    )
    # Correlation heatmap
    st.subheader("Correlation Between Indicators")
    correlation_matrix = X.corr()
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale='RdBu',
        zmid=0
    ))
    fig.update_layout(title="Correlation Heatmap", width=1000, height=900)
    st.plotly_chart(fig, use_container_width=False)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>Data Source: World Bank Global Findex Database</p>
        <p>Last Updated: 2024</p>
    </div>
    """, unsafe_allow_html=True) 