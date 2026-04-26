
import pickle
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

#load model
model = pickle.load(open('model_rf.pkl', 'rb'))

#scaling
scaler = MinMaxScaler()

#give title
st.title('HR Attrition prediction')

age = st.number_input('Age', min_value=18, max_value=100, value=30)
salary = st.number_input('Salary', min_value=0, max_value=100000, value=5000)
job_role = st.selectbox('Job Role', ['Sales', 'Research', 'HR', 'Developer'])
job_satisfaction = st.number_input('Job Satisfaction', min_value=1, max_value=5, value=3)
work_experience = st.number_input('Work Experience', min_value=0, max_value=50, value=5)
overtime = st.selectbox('Overtime', ['Yes', 'No'])
work_life_balance = st.number_input('Work-Life Balance', min_value=1, max_value=5, value=3)
department = st.selectbox('Department', ['Sales', 'R&D', 'HR'])

#encoding

#overtime
overtime = 1 if overtime == 'Yes' else 0

#job_role
role_dict = {'Sales': 0, 'Research': 1, 'HR': 2, 'Developer': 3}
job_role = role_dict[job_role]

#department
dept_dict = {'Sales': 0, 'R&D': 1, 'HR': 2}
department = dept_dict[department]

#dataframe
input_features = pd.DataFrame({
    'Age': [age],
    'Salary': [salary],
    'JobRole': [job_role],
    'JobSatisfaction': [job_satisfaction],
    'WorkExperience': [work_experience],
    'Overtime': [overtime],
    'WorkLifeBalance': [work_life_balance],
    'Department': [department]
})

input_features[['Age', 'Salary', 'WorkExperience']] = scaler.fit_transform(input_features[['Age', 'Salary', 'WorkExperience']])

#predictions
if st.button('Predict'):
    prediction = model.predict(input_features)
    if prediction[0] == 1:
        st.error("Prediction: Employee will Leave")
    else:
        st.success("Prediction: Employee will Stay")