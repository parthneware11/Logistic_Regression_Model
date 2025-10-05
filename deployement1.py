import streamlit as st
import numpy as np
import pickle

# -------------------------------
# Load the saved dictionary model
# -------------------------------
with open("titanic_model.pkl","rb") as f:
    loaded = pickle.load(f)
    
model = loaded['model']   
# Extract the actual model
# If you ever need label encoders, you can use: encoders = loaded['label_encoders']

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="🚢 Titanic Survival Predictor", layout="centered")

st.title("🚢 Titanic Survival Prediction App")
st.write("Enter the passenger details below to predict survival outcome:")

# -------------------------------
# Input fields
# -------------------------------
pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", min_value=0, max_value=100, value=30)
sibsp = st.number_input("Siblings/Spouses Aboard (SibSp)", min_value=0, max_value=10, value=0)
parch = st.number_input("Parents/Children Aboard (Parch)", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=32.0)
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

# -------------------------------
# Encode categorical variables
# -------------------------------
sex_num = 1 if sex == "male" else 0
embarked_map = {"C": 0, "Q": 1, "S": 2}
embarked_num = embarked_map[embarked]

# Combine inputs into an array (match training order)
features = np.array([[pclass, sex_num, age, sibsp, parch, fare, embarked_num]])

# -------------------------------
# Prediction
# -------------------------------
if st.button("🔎 Predict Survival"):
    try:
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]
        
        if prediction == 1:
            st.success(f"🎉 The passenger is LIKELY TO SURVIVE (Probability: {probability:.2f})")
        else:
            st.error(f"💀 The passenger is UNLIKELY TO SURVIVE (Probability: {probability:.2f})")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.markdown("---")
st.caption("Model: Logistic Regression | Built with ❤️ using Streamlit")




