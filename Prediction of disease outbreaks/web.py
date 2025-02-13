import pickle
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
from sklearn.preprocessing import StandardScaler

# Load models
diabetes_model= pickle.load(open(r"C:\Users\AMBUJ\OneDrive\Desktop\Prediction of disease outbreaks\Training_Models\diabetes_model.sav",'rb'))
heart_disease_model=pickle.load(open(r"C:\Users\AMBUJ\OneDrive\Desktop\Prediction of disease outbreaks\Training_Models\heart_disease.sav",'rb'))
parkinsons_model= pickle.load(open(r"C:\Users\AMBUJ\OneDrive\Desktop\Prediction of disease outbreaks\Training_Models\parkinson_model.sav",'rb'))

# Set up sidebar menu
with st.sidebar:
    selected = option_menu('Disease Prediction System',
                           ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction'],
                           menu_icon='hospital-fill',
                           icons=['activity', 'heart', 'person'],
                           default_index=0)

# ===== Diabetes Prediction =====
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using ML')
    
    col1, col2, col3 = st.columns(3)
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies', '0')
    with col2:
        Glucose = st.text_input('Glucose Level', '0')
    with col3:
        BloodPressure = st.text_input('Blood Pressure Value', '0')
    with col1:
        SkinThickness = st.text_input('Skin Thickness Value', '0')
    with col2:
        Insulin = st.text_input('Insulin Level', '0')
    with col3:
        BMI = st.text_input('BMI Value', '0')
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function Value', '0')
    with col2:
        Age = st.text_input('Age of the Person', '0')
    
    if st.button('Diabetes Test Result'):
        user_input = np.array([float(Pregnancies), float(Glucose), float(BloodPressure),
                               float(SkinThickness), float(Insulin), float(BMI),
                               float(DiabetesPedigreeFunction), float(Age)]).reshape(1, -1)
        result = diabetes_model.predict(user_input)
        st.success('Diabetic' if result[0] == 1 else 'Not Diabetic')

# ===== Heart Disease Prediction =====
if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction using ML')
    
    features = ['Age', 'Sex', 'Chest Pain Type', 'Resting BP', 'Cholesterol', 'Fasting Blood Sugar',
                'Resting ECG', 'Max Heart Rate', 'Exercise-Induced Angina', 'Oldpeak',
                'Slope', 'Number of Major Vessels', 'Thal']
    user_input = []
    
    for feature in features:
        user_input.append(st.text_input(feature, '0'))
    
    if st.button('Heart Disease Test Result'):
        user_input = np.array([float(x) for x in user_input]).reshape(1, -1)
        result = heart_model.predict(user_input)
        st.success('Heart Disease Detected' if result[0] == 1 else 'No Heart Disease')

# ===== Parkinson's Disease Prediction =====
if selected == "Parkinsons Prediction":
    st.title("Parkinson's Disease Prediction using ML")
    
    features = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)',
                'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)',
                'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR',
                'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE']
    user_input = []
    
    for feature in features:
        user_input.append(st.text_input(feature, '0'))
    
    if st.button("Parkinson's Test Result"):
        user_input = np.array([float(x) for x in user_input]).reshape(1, -1)
        result = parkinsons_model.predict(user_input)
        st.success("Parkinson's Disease Detected" if result[0] == 1 else "No Parkinson's Disease")
