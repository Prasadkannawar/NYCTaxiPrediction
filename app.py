import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# -----------------------------
# Title and Intro
# -----------------------------
st.set_page_config(page_title="NYC Green Taxi Fare Predictor", layout="wide")
st.title("NYC Green Taxi Fare Predictor")
st.markdown("🚖 Predict `total_amount` using Multiple Linear Regression on NYC Green Taxi Data")

# -----------------------------
# Load and Preprocess Dataset
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_parquet("e:\Semester 6\PA\Project\green_tripdata_2020-07.parquet")

    # Feature Engineering
    df["trip_duration"] = (df["lpep_dropoff_datetime"] - df["lpep_pickup_datetime"]).dt.total_seconds() / 60
    df.fillna(df.median(numeric_only=True), inplace=True)
    df.fillna("Unknown", inplace=True)
    df["weekday"] = df["lpep_dropoff_datetime"].dt.day_name()
    df["hourofday"] = df["lpep_dropoff_datetime"].dt.hour

    # Remove rows with negative total_amount
    df = df[df['total_amount'] >= 0]

    # Encode categorical variables
    df_encoded = pd.get_dummies(df[["store_and_fwd_flag", "RatecodeID", "payment_type", "trip_type", "weekday", "hourofday"]], drop_first=True)

    # Select numeric columns
    numeric_cols = ['trip_distance', 'fare_amount', 'extra', 'mta_tax', 'tip_amount', 
                    'tolls_amount', 'improvement_surcharge', 'congestion_surcharge', 
                    'trip_duration', 'passenger_count']
    
    X = pd.concat([df[numeric_cols], df_encoded], axis=1)
    y = df["total_amount"]
    
    return df, X, y, list(X.columns)

df, X, y, feature_names = load_data()

# -----------------------------
# Train Linear Regression Model
# -----------------------------
@st.cache_resource
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

model = train_model(X, y)

# -----------------------------
# Sidebar for User Inputs
# -----------------------------
st.sidebar.header("🔧 Input Features for Prediction")

# Reasonable default values
default_values = {
    'trip_distance': 2.0,
    'fare_amount': 8.0,
    'extra': 0.5,
    'mta_tax': 0.5,
    'tip_amount': 1.5,
    'tolls_amount': 0.0,
    'improvement_surcharge': 0.3,
    'congestion_surcharge': 2.5,
    'trip_duration': 10.0,
    'passenger_count': 1.0,
}

user_input = {}
for col in feature_names:
    val = st.sidebar.number_input(col, value=default_values.get(col, 0.0))
    user_input[col] = val

input_df = pd.DataFrame([user_input])

# -----------------------------
# Predict Button
# -----------------------------
if st.button("🔮 Predict Total Amount"):
    prediction = model.predict(input_df)[0]
    prediction = max(0, prediction)  # Prevent negative predictions
    st.success(f"💰 Predicted Total Amount: ${prediction:.2f}")

# -----------------------------
# Dynamic Visualization Section
# -----------------------------
st.subheader("📊 Explore Total Amount with Dynamic Graph")

# Dropdown for graph type
plot_type = st.selectbox("Select Visualization Type", ["Histogram", "Boxplot", "Density Curve"])

fig, ax = plt.subplots()

if plot_type == "Histogram":
    sns.histplot(df['total_amount'], bins=50, kde=False, ax=ax)
    ax.set_title("Histogram of Total Amount")
    ax.set_xlabel("Total Amount")
    ax.set_ylabel("Frequency")

elif plot_type == "Boxplot":
    sns.boxplot(x=df['total_amount'], ax=ax)
    ax.set_title("Boxplot of Total Amount")
    ax.set_xlabel("Total Amount")

elif plot_type == "Density Curve":
    sns.kdeplot(df['total_amount'], fill=True, ax=ax)
    ax.set_title("Density Curve of Total Amount")
    ax.set_xlabel("Total Amount")

# Show the selected plot
st.pyplot(fig)
