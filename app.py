import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
from plotly import graph_objs as go

data = pd.read_csv("data//Salary_Data.csv")
st.title("Salary Predictor")
x = np.array(data['YearsExperience']).reshape(-1,1)
lr = LinearRegression()
lr.fit(x,np.array(data['Salary']))
navigation = st.sidebar.radio("Navigate Here",["Home","Prediction","Add Salary"])

if (navigation == "Home"):
    st.image("data//sal.jpg", width=800)
    if (st.checkbox("Show Table")):
        st.table(data)
    graph = st.selectbox("Select the graph",["Non-Interactive","Interactive"])
    val = st.slider("Filter data using years",0,20)
    data = data.loc[data["YearsExperience"]>= val]
    if (graph == "Non-Interactive"):
        plt.figure(figsize=(10,5))
        plt.scatter(data["YearsExperience"],data["Salary"])
        plt.ylim(0)
        plt.xlabel("Years of Experience")
        plt.ylabel("Salary")
        plt.tight_layout()
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
    if (graph == "Interactive"):
        layout = go.Layout(
            xaxis = dict(range=[0,16]),
            yaxis = dict(range=[0,210000])
        )
        figure = go.Figure(data=go.Scatter(x=data["YearsExperience"], y=data["Salary"], mode='markers'),layout = layout)
        st.plotly_chart(figure)

if (navigation == "Prediction"):
    st.header("Know your Salary")
    val = st.number_input("Enter you exp",0.00,20.00,step = 0.25)
    val = np.array(val).reshape(1,-1)
    pred =lr.predict(val)[0]

    if st.button("Predict"):
        st.success("Your predicted salary is {}".format(round(pred)))
if (navigation == "Add Salary"):
    st.header("Contribute to our dataset")
    ex = st.number_input("Enter your Experience",0.0,20.0)
    sal = st.number_input("Enter your Salary",0.00,1000000.00,step = 1000.0)
    if st.button("Submit"):
        to_add = {"YearsExperience":[ex],"Salary":[sal]}
        to_add = pd.DataFrame(to_add)
        to_add.to_csv("data//Salary_Data.csv",mode='a',header = False,index= False)
        st.success("Submitted")