import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore
import streamlit as st


# Functions for handling missing numbers values
def handle_missing_values(df, strategy):
    numeric_cols = df.select_dtypes(include=['number']).columns
    if strategy == 'drop':
        df = df.dropna()
    elif strategy == 'mean':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif strategy == 'median':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    elif strategy == 'mode':
        for col in df.columns:
            df[col] = df[col].fillna(df[col].mode().iloc[0])
    return df

# Functions for detecting and handling outliers
def detect_outliers(df, method, threshold=1.5):
    numeric_cols = df.select_dtypes(include=['number']).columns

    if method == 'IQR':
        Q1 = df[numeric_cols].quantile(0.25)
        Q3 = df[numeric_cols].quantile(0.75)
        IQR = Q3 - Q1
        df = df[~((df[numeric_cols] < (Q1 - threshold * IQR)) | (df[numeric_cols] > (Q3 + threshold * IQR))).any(axis=1)]
    elif method == 'zscore':
        z_scores = zscore(df.select_dtypes(include=['float64', 'int64']))
        df = df[(z_scores < threshold).all(axis=1)]
    return df

# EDA Function
def automated_eda(df):
    st.write("### Basic Statistics")

    st.write("#### Summary Statistics (Numeric Columns)")
    st.write(df.describe())

    st.write("#### First 5 Rows")
    st.dataframe(df.head())

    st.write("#### Missing Values")
    missing_values = df.isnull().sum()
    st.write(missing_values[missing_values > 0])

    st.write("#### Data Types")
    st.write(df.dtypes)


    # st.write("### Histograms")
    # for col in df.select_dtypes(include=['float64', 'int64']).columns:
    #     st.write(f"Histogram for {col}")
    #     st.bar_chart(df[col])

    st.write("### Correlation Heatmap")
    numeric_cols = df.select_dtypes(include=['number'])  # Select only numeric columns
    if not numeric_cols.empty:
        correlation_matrix = numeric_cols.corr()  # Compute correlation only on numeric columns
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        st.pyplot(plt)
    else:
        st.write("No numeric columns available for correlation heatmap.")

# Streamlit App
def main():
    st.title("Data Cleaning and EDA Tool")

    # File Upload
    uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx", "xls"])
    if uploaded_file:
        # Load the dataset
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.write("### Uploaded Dataset")
        st.dataframe(df.head())

        # Missing Value Handling
        missing_strategy = st.selectbox(
            "How would you like to handle missing values?",
            ['None', 'drop', 'mean', 'median', 'mode']
        )

        if missing_strategy != 'None':
            df = handle_missing_values(df, strategy=missing_strategy)
            st.write(f"Applied missing value handling strategy: {missing_strategy}")

        # Outlier Detection
        outlier_method = st.selectbox(
            "How would you like to handle outliers?",
            ['None', 'IQR', 'zscore']
        )

        if outlier_method != 'None':
            df = detect_outliers(df, method=outlier_method)
            st.write(f"Applied outlier detection method: {outlier_method}")

        # Perform Automated EDA
        if st.button("Run Automated EDA"):
            automated_eda(df)

        # Save Cleaned Dataset
        if st.button("Download Cleaned Dataset"):
            cleaned_csv = df.to_csv(index=False)
            st.download_button(
                label="Download as CSV",
                data=cleaned_csv,
                file_name="cleaned_data.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()
