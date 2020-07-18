#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 20:04:45 2020

@author: ravivarma
"""

## Core
import streamlit as st
import streamlit.components.v1 as components
from streamlit_pandas_profiling import st_profile_report

## Custome Components
import sweetviz as sv

def st_display_sweetviz(report_html, width = 1000, height = 500):
    report_file = codecs.open(report_html, 'r')
    page = report_file.read()
    components.html(page,width=width,height=height,scrolling=True)

## EDA
import pandas as pd
import numpy as np
import codecs
from pandas_profiling import ProfileReport

## Data Visualization
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
# COLOR = "black"
# BACKGROUND_COLOR = "#fff"
import seaborn as sns

## Machine Learning
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


def main():
    """AutoML Web App Tool with Streamlit"""
    
    st.title("AutoML WebApp")
    st.text("Version(Beta): 0.2")
    
    menu = ["Home", "Pandas Profile", "Sweetviz", "EDA", "Plot", "Model Building", "About"]
    choice = st.sidebar.selectbox("Menu", menu)
        
    if choice == 'Home':
        st.markdown('**Data Analysis, Visualization** and Machine Learning **Model Building** in an interactive **WebApp** for Data Scientist/Data Engineer/Business Analyst.  \n\nThe purpose of this app is to create a **quick Business Insights**.  \n\nAutoML WebApp built with **Streamlit framework** using **Pandas** and **Numpy** for Data Analysis, **Matplotlib** and **Seaborn** for Data Visualization, **SciKit-Learn** for Machine Learning Model.')
#         st.markdown('**Demo URL**: https://automlwebapp.herokuapp.com/')
        st.header("Silent Features")
        st.markdown('* User can browse or upload file(Dataset) in .csv or .txt format.  \n* User can get the details of dataset like No. of rows & Columns, Can View Column list, Select Columns with rows to show, Dataset Summary like count, mean, std, min and max values.  \n* Several Data Visualizations like Correlation with HeatMap, PieChart and Plots like Area, Bar, Line, Box, KDE.  \n* User can built Models like LogisticRegression, LinearDiscriminantAnalysis, KNeighborsClassifier, DecisionTreeClassifier, GaussianNB, SVC.  \n* Model Evaluation with Accuracy, Mean and Standard Deviation.')

    
           
    if choice == 'Pandas Profile':
        st.subheader("Automated EDA with Pandas Profile")
        
        data = st.file_uploader("Upload Dataset", type=["csv", "txt"]) 
        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head())
            profile = ProfileReport(df)
            st_profile_report(profile)
    
    if choice == 'Sweetviz':
        st.subheader("Automated EDA with Sweetviz")
        
        data = st.file_uploader("Upload Dataset", type=["csv", "txt"]) 
        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head())
            if st.button("Generate Sweetviz Report"):
            
            ## WorkFlow
                report = sv.analyze(df)
                report.show_html()
                st_display_sweetviz("SWEETVIZ_REPORT.html")
    
    
    if choice == 'EDA':
        st.subheader("Exploratory Data Analysis")
        
        data = st.file_uploader("Upload Dataset", type=["csv", "txt"]) 
        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head())
            
            if st.checkbox("Show Shape"):
                st.write(df.shape)
            
            if st.checkbox("Show Columns"):
                all_columns = df.columns.to_list()
                st.write(all_columns)
            
            if st.checkbox("Select Columns To Show"):
                selected_columns = st.multiselect("Select Columns", all_columns)
                new_df = df[selected_columns]
                st.dataframe(new_df)
                             
            if st.checkbox("Show Summary"):
                st.write(df.describe())
            
            if st.checkbox("Show Value Counts"):
                st.write(df.iloc[:,-1].value_counts())
            
            if st.checkbox("Correlation with Seaborn"):
                st.write(sns.heatmap(df.corr(),annot=True))
                st.pyplot()
        
            if st.checkbox("Pie Chart"):
                all_columns = df.columns.to_list()
                columns_to_plot = st.selectbox("Select 1 Column", all_columns)
                pie_plot = df[columns_to_plot].value_counts().plot.pie(autopct="%1.1f%%")
                st.write(pie_plot)
                st.pyplot()
                
                
    
    elif choice == 'Plot':
        st.subheader("Data Visualization")
        
        data = st.file_uploader("Upload Dataset", type=["csv", "txt"]) 
        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head())
        
        if st.checkbox("Correlation with Seaborn"):
                st.write(sns.heatmap(df.corr(),annot=True))
                st.pyplot()
        
        if st.checkbox("Pie Chart"):
            all_columns = df.columns.to_list()
            columns_to_plot = st.selectbox("Select 1 Column", all_columns)
            pie_plot = df[columns_to_plot].value_counts().plot.pie(autopct="%1.1f%%")
            st.write(pie_plot)
            st.pyplot()
            
        all_columns_names = df.columns.tolist()
        type_of_plot = st.selectbox("Select Type of Plot", ["area", "bar", "line", "hist", "box", "kde"])
        selected_columns_names = st.multiselect("Select Columns To Plot", all_columns_names)
         
        if st.button("Generate Plot"):
            st.success("Generating Customizable Plot of {} for {}".format(type_of_plot,selected_columns_names))
            
            ## Plot By Streamlit
            if type_of_plot == 'area':
                cust_data = df[selected_columns_names]
                st.area_chart(cust_data)
            
            elif type_of_plot == 'bar':
                cust_data = df[selected_columns_names]
                st.bar_chart(cust_data)
            
            elif type_of_plot == 'line':
                cust_data = df[selected_columns_names]
                st.line_chart(cust_data)
            
            ## Custom Plot
            elif type_of_plot:
                cust_plot=df[selected_columns_names].plot(kind=type_of_plot)
                st.write(cust_plot)
                st.pyplot()
                
            
        
    elif choice == 'Model Building':
        st.subheader("Building Ml Model")
        
        data = st.file_uploader("Upload Dataset", type=["csv", "txt"]) 
        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head())
            
            
            ## Model Building
            X = df.iloc[:,0:-1]
            Y = df.iloc[:,-1]
            seed = 7
            
            ## Model
            models = [] 
            models.append(("LR", LogisticRegression()))
            models.append(("LDA", LinearDiscriminantAnalysis()))
            models.append(("KNN", KNeighborsClassifier()))
            models.append(("CART", DecisionTreeClassifier()))
            models.append(("NB", GaussianNB()))
            models.append(("SVM", SVC()))
            ## Evaluate each model in turn

            ## List
            model_names = []
            model_mean = []
            model_std = []
            all_models = []
            scoring = 'accuracy'
            
            for name, model in models:
                kfold = model_selection.KFold(n_splits=10, random_state=seed)
                cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
                model_names.append(name)
                model_mean.append(cv_results.mean())
                model_std.append(cv_results.std())
                
                accuracy_results = {"model_name":name,"model_accuracy":cv_results.mean(),"standard_deviation":cv_results.std()}
                all_models.append(accuracy_results)
            
            if st.checkbox ("Metrics as Table"):
                st.dataframe(pd.DataFrame(zip(model_names, model_mean, model_std),columns=["Model Name", "Model Accuracy", "Standard Deviation"]))                
            
            if st.checkbox("Metrics as JSON"):
                st.json(all_models)

    elif choice == "About":
        st.header("About Author")
        st.markdown("Hi, there! I'm **Ravi Varma**. I'm passionate about using data to extract decision making insight and help machines learn to make the world a better place. If you liked what you saw, want to have a chat with me about the **Data Science** or **Machine Learning Projects**, **Work Opportunities**, or collaboration, shoot an **email** at **ravivarmathotakura@gmail.com**")
        st.markdown('**Portfolio**: https://ravivarmathotakura.github.io/portfolio/')
        st.markdown('**Follow Me**: [@LinkedIn](https://www.linkedin.com/in/ravivarmathotakura/ "LinkedIn"), [@GitHub](https://github.com/ravivarmathotakura "GitHub")')
        st.subheader("Note")
        st.text("The author is not responsible for any misuse the program. \nAny contribution or suggestions are most welcome.")
        
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
