"""  
import streamlit as st
import joblib
import numpy as np

# Load the model
model = joblib.load("best_model.pkl")

st.title("💼 Employee Salary Prediction App")

st.write("Enter the employee details below to predict income class:")

# Collect input features
age = st.number_input("Age", min_value=18, max_value=75, value=30)
workclass = st.selectbox("Workclass", ['Private', 'Self-emp', 'Government', 'Others'])  # map to int
marital_status = st.selectbox("Marital Status", ['Married', 'Single', 'Divorced'])  # map to int
occupation = st.selectbox("Occupation", ['Tech', 'Sales', 'Admin', 'Others'])  # map to int
relationship = st.selectbox("Relationship", ['Husband', 'Wife', 'Own-child', 'Not-in-family', 'Unmarried'])
race = st.selectbox("Race", ['White', 'Black', 'Asian', 'Other'])
gender = st.selectbox("Gender", ['Male', 'Female'])
hours_per_week = st.slider("Hours per week", 1, 100, 40)
capital_gain = st.number_input("Capital Gain", value=0)
capital_loss = st.number_input("Capital Loss", value=0)
educational_num = st.slider("Educational Number", 1, 16, 10)
native_country = st.selectbox("Native Country", ['United-States', 'Others'])  # map to int

# Example encodings (you must ensure these match what LabelEncoder did)
def encode(value, mapping):
    return mapping.get(value, 0)

if st.button("Predict"):
    # Encoding - dummy maps (replace with your actual mappings)
    workclass_map = {'Private': 0, 'Self-emp': 1, 'Government': 2, 'Others': 3}
    marital_map = {'Married': 0, 'Single': 1, 'Divorced': 2}
    occupation_map = {'Tech': 0, 'Sales': 1, 'Admin': 2, 'Others': 3}
    relationship_map = {'Husband': 0, 'Wife': 1, 'Own-child': 2, 'Not-in-family': 3, 'Unmarried': 4}
    race_map = {'White': 0, 'Black': 1, 'Asian': 2, 'Other': 3}
    gender_map = {'Male': 0, 'Female': 1}
    country_map = {'United-States': 0, 'Others': 1}

    # Convert to encoded format
    input_data = [
        age,
        encode(workclass, workclass_map),
        encode(marital_status, marital_map),
        encode(occupation, occupation_map),
        encode(relationship, relationship_map),
        encode(race, race_map),
        encode(gender, gender_map),
        capital_gain,
        capital_loss,
        hours_per_week,
        educational_num,
        encode(native_country, country_map)
    ]

    prediction = model.predict([input_data])[0]

    result = ">50K" if prediction == 1 else "<=50K"
    st.success(f"💰 Predicted Salary Class: **{result}**")  """

#%%writefile app.py

import streamlit as st
import pickle
import numpy as np

def load_model():
    with open('best_model.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()
regressor = data['model']
le_occupation = data['le_occupation']

def show_predict_page():
    st.title("Salary Prediction")

    st.write("""### Please enter the required information to predict the salary""")

    education_map = {
        'Preschool': 1,
        '1st-4th': 2,
        '5th-6th': 3,
        '7th-8th': 4,
        '9th': 5,
        '10th': 6,
        '11th': 7,
        '12th': 8,
        'HS-grad': 9,
        'Some-college': 10,
        'Assoc-voc': 11,
        'Assoc-acdm': 12,
        'Bachelors': 13,
        'Masters': 14,
        'Prof-school': 15,
        'Doctorate': 16
    }

    Occupation = (
        "Tech-support", "Craft-repair", "Other-service", "Sales",
        "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct",
        "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv",
        "Protective-serv", "Armed-Forces"
    )
    education_raw = st.selectbox("Education Level", list(education_map.keys()))
    educational_num = education_map[education_raw]
    occupation = st.selectbox("Occupation", Occupation)
    age = st.slider("Age", 18, 65, 30)
    hours_per_week = st.slider("Hours per week", 1, 80, 40)
    #experience = st.slider("Years of Experience", 0, 40, 5)

    ok = st.button("Calculate Salary")
    if ok:
        input_data = np.array([[age, educational_num, occupation, hours_per_week]])

        # Transform occupation
        input_data[:, 2] = le_occupation.transform(input_data[:, 2])

        input_data = input_data.astype(float)

        prediction = regressor.predict(input_data)

        proba = regressor.predict_proba(input_data)

        if prediction[0] == 1:
            confidence = proba[0][1] * 100
            st.success(f"Predicted: >50K (Confidence: {confidence:.2f}%)")
        else:
            confidence = proba[0][0] * 100
            st.info(f"Predicted: <=50K (Confidence: {confidence:.2f}%)")






show_predict_page()
