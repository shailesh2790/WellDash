import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import sqlite3
import json
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer

# Use wide mode
st.set_page_config(layout="wide")

# Database functions
@st.cache_resource
def get_database_connection():
    return sqlite3.connect('dashboard_data.db', check_same_thread=False)

def init_db():
    conn = get_database_connection()
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS files
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  filename TEXT,
                  file_type TEXT,
                  data TEXT)''')
    conn.commit()

@st.cache_data(ttl=0)
def get_files_from_db():
    conn = get_database_connection()
    c = conn.cursor()
    c.execute("SELECT id, filename, file_type FROM files")
    files = c.fetchall()
    return files

@st.cache_data
def get_file_data_from_db(file_id):
    conn = get_database_connection()
    c = conn.cursor()
    c.execute("SELECT data FROM files WHERE id=?", (file_id,))
    data = c.fetchone()
    if data:
        return pd.read_json(data[0])
    return None

# Data processing functions
@st.cache_data
def preprocess_data(_df):
    numeric_cols = _df.select_dtypes(include=[np.number]).columns
    imputer = SimpleImputer(strategy='mean')
    imputed_data = imputer.fit_transform(_df[numeric_cols])
    return pd.DataFrame(imputed_data, columns=numeric_cols)

@st.cache_data
def perform_pca(_df):
    preprocessed_df = preprocess_data(_df)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(preprocessed_df)
    pca = PCA()
    pca_result = pca.fit_transform(scaled_data)
    return pca_result, pca.explained_variance_ratio_

@st.cache_data
def perform_kmeans(_df, n_clusters):
    preprocessed_df = preprocess_data(_df)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(preprocessed_df)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(scaled_data)
    return cluster_labels

# Visualization functions
@st.cache_data
def plot_correlation_heatmap(_df):
    numeric_cols = _df.select_dtypes(include=[np.number]).columns
    corr_matrix = _df[numeric_cols].corr()
    fig = px.imshow(corr_matrix, color_continuous_scale='RdBu_r', aspect='auto')
    fig.update_layout(title='Correlation Heatmap')
    return fig

@st.cache_data
def plot_pca_variance(explained_variance_ratio):
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=cumulative_variance_ratio, mode='lines+markers', name='Cumulative Explained Variance'))
    fig.add_trace(go.Bar(y=explained_variance_ratio, name='Explained Variance'))
    fig.update_layout(title='PCA Explained Variance', xaxis_title='Principal Components', yaxis_title='Explained Variance Ratio')
    return fig

def upload_select_data():
    st.header("Upload or Select Data")

    option = st.radio("Choose an option", ("Upload a new file", "Select an existing file"))

    if option == "Upload a new file":
        uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])
        
        if uploaded_file is not None:
            with st.spinner("Processing file..."):
                df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            
            st.success("File successfully uploaded and processed!")
            
            conn = get_database_connection()
            cursor = conn.cursor()

            # Insert the data
            file_type = uploaded_file.name.split('.')[-1]
            data_json = df.to_json(orient='records')
            cursor.execute('''INSERT INTO files (filename, file_type, data)
                              VALUES (?, ?, ?)''',
                           (uploaded_file.name, file_type, data_json))

            conn.commit()
            
            st.success("File saved to database!")
            
            st.session_state['df'] = df
            st.session_state['filename'] = uploaded_file.name

            # Clear the cache to force a refresh of the file list
            get_files_from_db.clear()

    # Select an existing file
    files = get_files_from_db()
    if files:
        file_options = {f"{file[1]} ({file[2]})": file[0] for file in files}
        selected_file = st.selectbox("Select a file", list(file_options.keys()))
        
        if selected_file:
            file_id = file_options[selected_file]
            with st.spinner("Loading data..."):
                df = get_file_data_from_db(file_id)
            if df is not None:
                st.session_state['df'] = df
                st.session_state['filename'] = selected_file
    else:
        st.info("No files in the database. Please upload a file first.")

@st.cache_data
def get_data_summary(_df):
    return _df.describe()

@st.cache_data
def get_missing_data(_df):
    return _df.isnull().sum()

def explore_data():
    st.header("Explore Data")

    if 'df' not in st.session_state:
        st.warning("Please upload or select a file first.")
        return

    df = st.session_state['df']
    filename = st.session_state['filename']

    st.subheader(f"Exploring: {filename}")

    # Display raw data
    st.subheader("Raw Data")
    st.write(df.head())

    # Data summary
    st.subheader("Data Summary")
    st.write(get_data_summary(df))

    # Missing values
    st.subheader("Missing Values")
    missing_data = get_missing_data(df)
    st.write(missing_data[missing_data > 0])

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    st.plotly_chart(plot_correlation_heatmap(df))

    # Data visualization
    st.subheader("Data Visualization")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) >= 2:
        plot_type = st.selectbox("Select plot type", ["Scatter", "Line", "Bar", "Box", "Violin"])
        
        x_col = st.selectbox("Select X-axis column", numeric_cols)
        y_col = st.selectbox("Select Y-axis column", numeric_cols, index=1)
        
        with st.spinner("Generating plot..."):
            if plot_type == "Scatter":
                fig = px.scatter(df, x=x_col, y=y_col)
            elif plot_type == "Line":
                fig = px.line(df, x=x_col, y=y_col)
            elif plot_type == "Bar":
                fig = px.bar(df, x=x_col, y=y_col)
            elif plot_type == "Box":
                fig = px.box(df, y=y_col)
            elif plot_type == "Violin":
                fig = px.violin(df, y=y_col)
            
            st.plotly_chart(fig)
    else:
        st.warning("Not enough numeric columns for visualization.")

def advanced_analysis():
    st.header("Advanced Analysis")

    if 'df' not in st.session_state:
        st.warning("Please upload or select a file first.")
        return

    df = st.session_state['df']
    filename = st.session_state['filename']

    st.subheader(f"Analyzing: {filename}")

    # Display missing values
    st.subheader("Missing Values")
    missing_data = get_missing_data(df)
    st.write(missing_data[missing_data > 0])

    # Data preprocessing
    st.subheader("Data Preprocessing")
    st.write("Handling missing values using mean imputation for numeric columns.")
    with st.spinner("Preprocessing data..."):
        preprocessed_df = preprocess_data(df)
    st.write("Preprocessed Data Sample:")
    st.write(preprocessed_df.head())

    # PCA Analysis
    st.subheader("Principal Component Analysis (PCA)")
    if st.button("Perform PCA"):
        with st.spinner("Performing PCA..."):
            pca_result, explained_variance_ratio = perform_pca(df)
        st.plotly_chart(plot_pca_variance(explained_variance_ratio))

    # K-means Clustering
    st.subheader("K-means Clustering")
    n_clusters = st.slider("Select number of clusters", 2, 10, 3)
    if st.button("Perform K-means Clustering"):
        with st.spinner("Performing K-means Clustering..."):
            cluster_labels = perform_kmeans(df, n_clusters)
        
        numeric_cols = preprocessed_df.columns
        x_col = st.selectbox("Select X-axis column", numeric_cols)
        y_col = st.selectbox("Select Y-axis column", numeric_cols, index=1)
        
        fig = px.scatter(preprocessed_df, x=x_col, y=y_col, color=cluster_labels, title='K-means Clustering Result')
        st.plotly_chart(fig)

    # Time series analysis (if applicable)
    date_cols = df.select_dtypes(include=['datetime64']).columns
    if len(date_cols) > 0:
        st.subheader("Time Series Analysis")
        date_col = st.selectbox("Select date column", date_cols)
        value_col = st.selectbox("Select value column", df.select_dtypes(include=[np.number]).columns)
        
        if st.button("Perform Time Series Analysis"):
            with st.spinner("Performing Time Series Analysis..."):
                df_time = df.set_index(date_col)
                fig = px.line(df_time, y=value_col, title='Time Series Plot')
                st.plotly_chart(fig)

                # Simple moving average
                window_size = st.slider("Select moving average window size", 1, 30, 7)
                df_time['SMA'] = df_time[value_col].rolling(window=window_size).mean()
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_time.index, y=df_time[value_col], mode='lines', name='Original'))
                fig.add_trace(go.Scatter(x=df_time.index, y=df_time['SMA'], mode='lines', name=f'{window_size}-day SMA'))
                fig.update_layout(title='Time Series with Simple Moving Average')
                st.plotly_chart(fig)

def main():
    st.title("Advanced Interactive Data Dashboard")

    init_db()

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Choose the app mode", ["Upload/Select Data", "Explore Data", "Advanced Analysis"])

    if app_mode == "Upload/Select Data":
        upload_select_data()
    elif app_mode == "Explore Data":
        explore_data()
    elif app_mode == "Advanced Analysis":
        advanced_analysis()

if __name__ == "__main__":
    main()
