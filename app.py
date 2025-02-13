import pandas as pd
import sklearn
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.feature_selection import VarianceThreshold

# Load Dataset
df = pd.read_csv("hyderabad_eta_data.csv")

# Custom CSS Styles
st.markdown(
    """
    <style>
        .title {
            text-align: center;
            font-size: 32px;
            color: #00BFFF; /* Sky Blue */
            font-weight: bold;
        }
        .custom-label {
            font-size: 18px;
            font-weight: bold;
            color: #FFFFFF; /* White */
        }
        .selected-value {
            font-size: 20px;
            font-weight: bold;
            color: #FFD700; /* Gold */
        }
        .stSlider > div > div > div {
            background-color: #2196F3; /* Blue Slider Track */
        }
        .stSlider > div > div > div > div {
            background-color: #FFFFFF !important; /* White Handle */
        }
        .stButton > button {
            background-color: #007BFF;
            color: white;
            font-size: 18px;
            border-radius: 10px;
            padding: 10px;
            width: 100%;
        }
        .footer {
            text-align: center;
            font-size: 16px;
            margin-top: 30px;
            color: #FFFFFF;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<h1 class="title">🚗 Estimated Time of Arrival (ETA) Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="footer">Developed by Amarendra Nayak</p>', unsafe_allow_html=True)

# Display Logo
st.image("innomatics-footer-logo.png", width=700)

st.markdown("""
### 🛣️ **How It Works:**
This tool estimates the travel time between two locations based on **distance, traffic, weather, and time of day**.
Simply enter your trip details, and our AI model will predict your estimated time of arrival in minutes. 📍
""")

# Mapping weather & days to numerical values
weather_map = {'rainy': 0, 'clear': 1, 'foggy': 2}
day_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}

df['weather_condition'] = df['weather_condition'].map(weather_map)
df['day_of_week'] = df['day_of_week'].map(day_map)

# Splitting data
X = df.drop('ETA', axis=1)
y = df['ETA']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)

# Handling missing values
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Feature Selection
selector = VarianceThreshold(threshold=0)
X_train_selected = selector.fit_transform(X_train_imputed)
X_test_selected = selector.transform(X_test_imputed)

if X_train_selected.shape[1] == 0:
    st.error("All features were removed due to zero variance. Try adjusting preprocessing!")
    st.stop()

# Scaling Data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_selected)
X_test_scaled = scaler.transform(X_test_selected)

# Training the Model
@st.cache_resource
def train_model():
    dt = DecisionTreeRegressor()
    dt.fit(X_train_scaled, y_train)
    return dt

dt_model = train_model()

# User Input Section
st.subheader("📌 Enter Route Details:")

start_lat = st.slider('🌍 **Select Start Latitude**', -90.0, 90.0, 40.7128, 0.0001)
start_lng = st.slider('🧭 **Select Start Longitude**', -180.0, 180.0, -74.0060, 0.0001)
end_lat = st.slider('🌍 **Select End Latitude**', -90.0, 90.0, 34.0522, 0.0001)
end_lng = st.slider('🧭 **Select End Longitude**', -180.0, 180.0, -118.2437, 0.0001)
distance_km = st.slider('📏 **Select Distance (km)**', 0.0, 5000.0, 100.0, 0.1)
traffic_density = st.slider('🚦 **Traffic Level (0-10)**', 0, 10, 5)
weather_condition = st.selectbox("🌦️ **Select the Weather Condition**", ['rainy', 'clear', 'foggy'])
day_of_week = st.selectbox("📅 **Select the Day of the Week**", list(day_map.keys()))
hour_of_day = st.slider("⏰ **Select the Time (Hours)**", 0, 23, 5, step=1)

# Creating Input DataFrame
input_data = pd.DataFrame({
    'start_lat': [start_lat],
    'start_lng': [start_lng],
    'end_lat': [end_lat],
    'end_lng': [end_lng],
    'distance_km': [distance_km],
    'traffic_density': [traffic_density],
    'weather_condition': [weather_map[weather_condition]],
    'day_of_week': [day_map[day_of_week]],
    'hour_of_day': [hour_of_day]
})

# Handle Missing Values in User Input
input_imputed = imputer.transform(input_data)
input_selected = selector.transform(input_imputed)
input_scaled = scaler.transform(input_selected)

# Predict ETA
if st.button("🚀 Predict Estimated Time of Arrival"):
    eta_prediction = dt_model.predict(input_scaled)
    st.success(f'⏳ Estimated Time of Arrival: {eta_prediction[0]:.2f} minutes')
    
    # Evaluate Model Performance
    y_test_pred = dt_model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    st.info(f'📊 Model RMSE: {rmse:.2f} minutes')
