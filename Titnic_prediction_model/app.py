import streamlit as st
import pandas as pd
import joblib
model = joblib.load("titanic_model.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

st.title("Titanic survival prediction by Rishav")
st.markdown("provide the following details")
age = st.slider('Age',0,90,40)
sex = st.selectbox('Sex',['M','F'])
pclass = st.selectbox('PClass',[1,2,3])
fare = st.number_input('Fare',0,700,20)
sibsp = st.number_input('Sibsp',0,8,3)
embarked = st.selectbox('Embarked',['S','C','Q'])
alone = st.selectbox("Alone",['Y','N'])

if st.button("Prediction"):
     
    sex = 1 if sex == "M" else 0
    embarked_map = {"S": 0, "C": 1, "Q": 2}
    embarked = embarked_map[embarked]

    raw_input = {
        "Age": age,
        "Sex": sex,
        "PClass": pclass,
        "Fare": fare,
        "Sibsp": sibsp,
        "Embarked": embarked,
        "Alone": 1 if alone == "Y" else 0
    }
input_df=pd.DataFrame([raw_input])
    
for col in expected_columns:
    if col not in input_df.columns:
        input_df[col]=0
input_df=input_df[expected_columns]  
prediction=model.predict(input_df)[0]

if prediction[0] == 1:
        st.success("Survived")
else:
        st.error("Did not survive")