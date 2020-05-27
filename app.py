# -*- coding: utf-8 -*-
"""
Created on Wed May 27 07:42:21 2020

@author: susitha
"""



import numpy as np
from flask import Flask,request,render_template,url_for 

from keras.models import load_model 

import pickle 

import tensorflow as tf 

global graph 
graph=tf.compat.v1.get_default_graph()

with open(r'CounterVectorizer.pkl','rb') as file: 

    cv = pickle.load(file) 

    print("\ncv loaded\n") 

cla=load_model('mymodel.h5') 

cla.compile(optimizer='adam',loss='binary_crossentropy') 

  

app=Flask(__name__) 

  

@app.route('/') 

def index(): 

    return render_template('index.html') 

@app.route('/predict',methods=['GET','POST']) 

def page2(): 
    if request.method=='GET': 

        img_url=url_for('static',filename='style/4.png') 

        return render_template('index.html',url=img_url) 

    if request.method=='POST': 

        topic=request.form['tweet'] 

        print("Hey"+topic) 

        topic=cv.transform([topic]) 

        print("/n"+str([topic.shape])+"\n") 

        with graph.as_default(): 

            y_pred=cla.predict(topic) 

            print("pred is"+str(y_pred)) 

            if(y_pred>0.5): 

                img_url=url_for('static',filename='style/1_2.png') 

                topic="positive Tweet"

            elif( y_pred<0.5): 

 

 

                img_url=url_for('static',filename='style/1_3.png') 

                topic="Negative Tweet" 

            else: 

                img_url=url_for('static',filename='style/happy.png') 

                topic="Neutral Tweet"
            
                 

if(__name__=='__main__'): 

    app.run(debug=True) 