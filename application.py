from flask import Flask,request,jsonify,render_template 
import pickle 
import numpy as np , pandas as pd 
from sklearn.preprocessing import StandardScaler

application=Flask(__name__)
app=application

ridge_model=pickle.load(open('models/reegressor.pkl','rb'))
Scaler_model=pickle.load(open('models/scaaler.pkl','rb'))






@app.route ('/')
def index():
    return render_template('index.html')
@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        FWI=float(request.form.get('FWI'))
        Classes  = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))  


        new_data_scaled=Scaler_model.transform([[ RH, Ws,Rain ,FFMC,DMC,ISI,FWI,Classes  ,Region]])
        result=ridge_model.predict(new_data_scaled)

        return render_template('home.html',result=result[0])
    else :
        return render_template('home.html')


if __name__=='__main__':
    app.run(host='0.0.0.0')