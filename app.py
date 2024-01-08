import pandas as pd
import numpy as np
import streamlit as st
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

st.write(
    """
    # Penguin Prediction App
    
    This app predicts the Palmer Penguin species.
    """
)

st.sidebar.header("User Input Features")

def user_input_features():
    
    island = st.sidebar.selectbox('Island', ('Biscoe','Dream','Torgersen'))
    sex = st.sidebar.selectbox('Sex', ('male','female'))
    bill_length = st.sidebar.slider('Bill Length (mm)', 32.1, 59.6, 44.0)
    bill_depth = st.sidebar.slider("Bill Depth (mm)", 13.1, 21.5, 17.0)
    flipper_length = st.sidebar.slider('Flipper Length (mm)', 172.0, 231.0, 200.0)
    body_mass = st.sidebar.slider('Body Mass (g)', 2700.0, 6300.0, 4200.0)
    
    x_cat = [[sex,island]]
    x_num = [[bill_length,bill_depth,flipper_length,body_mass]]
    
    df = {
        "island":island,
        "sex":sex,
        "bill_length":bill_length,
        "bill_depth":bill_depth,
        "flipper_length":flipper_length,
        "body_mass":body_mass,
    }
    df = pd.DataFrame(df, index=[0])
    
    return (x_cat,x_num,df)
x_cat, x_num, df = user_input_features()

st.subheader("Input Features")
st.write(df)

def read_model():
    
    with open('models/encoder.pkl','rb') as f:
        encoder = pickle.load(f)
    with open('models/clf.pkl','rb') as f:
        clf = pickle.load(f)
    
    return (encoder,clf)
encoder, clf = read_model()

species = clf.classes_

x_cat = encoder.transform(x_cat)

x = np.hstack([x_cat,x_num])
ypred = clf.predict(x)
yprob = clf.predict_proba(x)

st.write("Prediction probability")
pred_df = pd.DataFrame(data=zip(species,yprob[0]), columns=['Species','Probability'])
st.write(pred_df)

st.write(f"Predicted Value: {ypred[0]}")

# st.write(ypred)
# st.write(yprob)