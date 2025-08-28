import streamlit as st
import joblib,pandas as pd

st.title("Titanic Age-Group Prediction")
st.write("Enter the passengers details to predict Age Group")

pclass=st.selectbox("Please select the Passenger Class",[1,2,3])
sex=st.selectbox("Please select the Gender",["male","female"])
sibsp=st.number_input("Please enter the number of Sibling or Spouses",0,8,0)
parch=st.number_input("Please enter the number of Parents/Children",0,6,0)
fare=st.slider("Please enter theticket fare",0,600,32)
embarked=st.selectbox("Please select the Port of Embarkation",['S','C','Q'])

family=sibsp+parch+1
IsAlone=1 if family==1 else 0

row=pd.DataFrame([{ 'Pclass':pclass,
                    'Sex':sex,
                    'SibSp':sibsp,
                    'Parch':parch,
                    'Fare':fare,
                    'Embarked':embarked,
                    'Family_Size':family,
                    'IsAlone':IsAlone,
                    }])

if st.button("Submit"):
    pipe=joblib.load("titanic_age_group_pipeline.joblib")
    pred=pipe.predict(row)[0]
    
    age_group={
        0: "G1 (Age<15)",
        1: "G2 (Age 15-30)",
        2: "G3 (Age 31-55)",
        3: "G4 (Age 56-80)",
        4: "G5 (Age >80)"
    }
    st.success(f"The predicted Age Group is:{age_group.get(pred,pred)}")
