# -----------------------------------------
# SMART DISASTER AI - STREAMLIT INTERFACE
# -----------------------------------------
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import random
import requests

# ---------------------------
# TITLE
# ---------------------------
st.set_page_config(page_title="Smart Disaster AI", page_icon="🌪️", layout="centered")
st.title("🌍 Smart Disaster AI")
st.write("Predict natural hazards and get personalized household safety advice!")

# ---------------------------
# STEP 1: TRAINING DATA
# ---------------------------
# Generate realistic dataset
rows = []
random.seed(42)
for _ in range(80):
    rainfall = random.randint(0, 120)
    wind = random.randint(0, 70)
    temperature = random.randint(15, 45)

    # Smarter labeling logic
    if rainfall > 85 and wind < 40:
        hazard = "flood"
    elif wind > 45:
        hazard = "storm"
    elif temperature > 38 and rainfall < 20:
        hazard = "wildfire"
    elif rainfall > 60 and wind > 40:
        hazard = "storm"
    else:
        hazard = "none"

    rows.append([rainfall, wind, temperature, hazard])

data = pd.DataFrame(rows, columns=['rainfall','wind_speed','temperature','hazard_type'])

# Train Random Forest
X = data[['rainfall','wind_speed','temperature']]
y = data['hazard_type']

model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
model.fit(X, y)

# ---------------------------
# STEP 2: USER INPUT
# ---------------------------
st.header("📥 Enter Weather Conditions")

use_api = st.radio("Use live weather data?", ('No', 'Yes'))

if use_api == 'Yes':
    api_key = st.text_input("Enter OpenWeatherMap API Key", type="password")
    city = st.text_input("Enter your city")
    rainfall, wind, temperature = 0, 0, 0
    if st.button("Fetch Weather"):
        try:
            url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
            response = requests.get(url).json()
            rainfall = response.get('rain', {}).get('1h', 0)
            wind = response['wind']['speed'] * 3.6
            temperature = response['main']['temp']
            st.success(f"Live weather fetched: Rainfall={rainfall} mm, Wind={round(wind,1)} km/h, Temp={temperature} °C")
        except:
            st.error("Failed to fetch weather. Please enter manually.")
            use_api = 'No'

if use_api == 'No':
    rainfall = st.slider("Rainfall (mm)", 0, 120, 10)
    wind = st.slider("Wind Speed (km/h)", 0, 70, 10)
    temperature = st.slider("Temperature (°C)", 15, 45, 25)

st.header("🏠 Household Information")
children = st.checkbox("Children in home?")
elderly = st.checkbox("Elderly in home?")
basement = st.checkbox("Basement available?")
pets = st.checkbox("Pets in home?")

# ---------------------------
# STEP 3: PREDICTION
# ---------------------------
input_df = pd.DataFrame([[rainfall, wind, temperature]],
                        columns=['rainfall','wind_speed','temperature'])

probs = model.predict_proba(input_df)[0]
classes = model.classes_
prediction = classes[probs.argmax()]
highest_prob = max(probs)

# Severity
if highest_prob < 0.4:
    severity = "LOW"
elif highest_prob < 0.7:
    severity = "MEDIUM"
else:
    severity = "HIGH"

# ---------------------------
# STEP 4: DISPLAY RESULTS
# ---------------------------
st.header("⚠️ Risk Analysis")
for hazard, prob in zip(classes, probs):
    st.write(f"{hazard.capitalize()}: {round(prob*100,1)}%")

st.subheader(f"Most Likely Hazard: {prediction.upper()}")
st.subheader(f"Risk Severity Level: {severity}")

# ---------------------------
# STEP 5: PERSONALIZED ADVICE
# ---------------------------
st.header("🛡️ Safety Recommendations")

if prediction == "flood":
    st.write("- Move to higher ground immediately.")
    st.write("- Prepare emergency evacuation kit.")
    if basement: st.write("- DO NOT shelter in basement.")
    if children: st.write("- Prepare fast evacuation plan for children.")
    if elderly: st.write("- Ensure medication and mobility aids ready.")
    if pets: st.write("- Bring pets indoors and prepare carriers.")
elif prediction == "storm":
    st.write("- Stay indoors.")
    st.write("- Shelter in interior room away from windows.")
    if basement: st.write("- Basement is a safe shelter location.")
    if children: st.write("- Keep children away from windows.")
    if pets: st.write("- Keep pets indoors.")
elif prediction == "wildfire":
    st.write("- Prepare for evacuation immediately.")
    st.write("- Close windows, doors, and vents.")
    if elderly: st.write("- Arrange early evacuation assistance.")
    if pets: st.write("- Prepare pet carriers.")
    if children: st.write("- Keep children ready for evacuation.")
elif prediction == "none":
    st.write("- No immediate hazard detected.")
    st.write("- Continue monitoring weather updates.")

st.info("Stay alert and follow official emergency services guidance.")
